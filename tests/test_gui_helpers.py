"""Tests for the GUI helper layer (no Streamlit needed). They check the GUI delegates to the real modules."""
import dataclasses
import numpy as np
import pandas as pd
import pytest

from gui import helpers as h
from csp import StudyCSP, CSPConfig, capacity_ok
from integration.planner import plan
from search import ALGOS


@pytest.fixture(scope="module")
def ctx():
    return h.load_context()


@pytest.fixture(scope="module")
def sc(ctx):
    return h.resolve_scenario(ctx, {}, True)


def test_context_uses_saved_model_and_heldout_split(ctx):
    assert ctx.model_path.exists() and len(ctx.profiles) >= 30
    assert ctx.order[-1] in set(ctx.test.code_presentation) and ctx.order[-1] not in set(ctx.train.code_presentation)
    assert [p.student_id for p in ctx.profiles] == sorted(p.student_id for p in ctx.profiles)


def test_missing_data_and_model_give_clear_errors(tmp_path):
    with pytest.raises(h.GuiError, match="Processed data not found"):
        h.load_context.__wrapped__(42, str(tmp_path / "nope.csv"))
    with pytest.raises(h.GuiError, match="No trained model"):
        h.load_context.__wrapped__(42, None, str(tmp_path / "nope.npz"))


def test_nn_on_off_use_the_right_predictors(ctx, sc):
    on, off = h.run_plan(ctx, sc), h.run_plan(ctx, dataclasses.replace(sc, nn_on=False))
    assert on.predictor_name == "NN" and off.predictor_name == "AvgMark"
    assert ctx.predictor(True) is ctx.nn and ctx.predictor(False) is ctx.avg
    assert all(len(set(row)) == 1 for row in off.result.table)            # baseline ignores effort
    assert any(len(set(row)) > 1 for row in on.result.table)             # NN reacts to effort
    # same target for both, taken from the NN rule
    assert on.result.target == off.result.target == sc.target


def test_run_plan_matches_direct_planner_call(ctx, sc):
    for algo in ALGOS:
        out = h.run_plan(ctx, sc, algo)
        direct = plan(ctx.profile(sc.student_id), ctx.nn, sc.target, sc.spw, algo, sc.goal, h.csp_config("Backtracking + Forward Checking + AC-3"))
        assert out.result.status == direct.status and out.result.allocation == direct.allocation
        assert out.result.search.expanded == direct.search.expanded and out.result.search.generated == direct.search.generated


def test_search_comparison_matches_planner_counts(ctx, sc):
    outs = h.compare_algorithms(ctx, sc)
    summ = h.search_summary(outs).set_index("key")
    for a, o in outs.items():
        assert summ.loc[a, "Expanded"] == o.result.search.expanded and summ.loc[a, "Generated"] == o.result.search.generated
    costs = {a: o.result.cost for a, o in outs.items() if o.ok}
    assert costs.get("ucs") == costs.get("astar")                         # both optimal
    assert outs["astar"].result.search.expanded <= outs["ucs"].result.search.expanded


def test_trace_frame_marks_goal_and_rejections(ctx):
    ex = h.find_csp_rejection_example(ctx)
    assert ex is not None
    tr = h.trace_frame(ex["result"])
    assert (tr.Outcome == "NN says enough → CSP rejected").sum() == ex["n_rejected"]
    assert len(tr) == len(ex["result"].trace) and tr.Step.is_monotonic_increasing


def test_timetable_grid_roundtrip(ctx, sc):
    out = h.run_plan(ctx, sc)
    sched = out.shown["schedule"]
    g = h.timetable_grid(sched, list(sc.weeks), sc.spw)
    assert g.ids.shape == (max(sc.weeks), sc.spw) and int((g.ids > 0).sum()) == len(sched) == sum(out.shown["allocation"])
    assert h.timetable_frame(sched, list(sc.weeks), sc.spw).shape == g.ids.shape


def test_csp_runs_match_solver_and_capacity_rule():
    for counts, weeks, spw in [((2, 3, 2), (2, 4, 6), 2), ((4, 5, 8), (3, 5, 8), 2), ((0, 0, 0), (1, 1, 1), 1), ((3, 3, 3), (1, 2, 3), 3)]:
        runs = h.run_csp_modes(counts, weeks, spw)
        cap = h.capacity_report(counts, weeks, spw)
        assert bool(cap["Enough slots"].all()) == capacity_ok(counts, weeks, spw)
        for mode, d in runs.items():
            direct = StudyCSP(counts, weeks, spw).solve(h.csp_config(mode))
            assert (d["result"].status, d["result"].stats) == (direct.status, direct.stats)
            assert d["valid"] == (d["result"].status == "solved")
        assert runs["Backtracking + Forward Checking + AC-3"]["result"].status in ("solved", "no_solution")


def test_fc_and_ac3_reduce_work_on_stress_example():
    e = h.STRESS_EXAMPLE
    s = h.csp_summary(h.run_csp_modes(e["counts"], e["weeks"], e["spw"]), e["counts"], e["weeks"], e["spw"]).set_index("Mode")
    bt, fc, ac = s.loc["Backtracking only"], s.loc["Backtracking + Forward Checking"], s.loc["Backtracking + Forward Checking + AC-3"]
    assert fc["Consistency checks"] < bt["Consistency checks"] and fc["FC prunings"] > 0 and bt["FC prunings"] == 0
    assert ac["Assignments"] < bt["Assignments"] and ac["AC-3 prunings"] > 0


def test_rejection_example_is_genuine(ctx):
    ex = h.find_csp_rejection_example(ctx)
    s, sc2 = ex["state"], ex["scenario"]
    table = ex["result"].table
    marks = [table[i][s[i]] for i in range(3)]
    ok = all(m >= sc2.target for m in marks) if sc2.goal == "each" else sum(marks) / 3 >= sc2.target - 1e-9
    assert ok                                                            # NN accepts ...
    assert not capacity_ok(s, list(sc2.weeks), sc2.spw)                  # ... CSP capacity test fails
    assert all(d["result"].status != "solved" for d in ex["runs"].values())


@pytest.mark.parametrize("kw,msg", [({"target": 150.0}, "between 0 and 100"), ({"target": float("nan")}, "must be a number"),
                                    ({"spw": 0}, "slots per week"), ({"goal": "xx"}, "Unknown goal"),
                                    ({"weeks": (0, 4, 4)}, "deadline"), ({"weeks": (4, 4)}, "Expected 3")])
def test_invalid_inputs_rejected(ctx, sc, kw, msg):
    with pytest.raises(h.GuiError, match=msg):
        h.run_plan(ctx, dataclasses.replace(sc, **kw))


def test_unknown_student_and_algo(ctx, sc):
    with pytest.raises(h.GuiError):
        h.run_plan(ctx, dataclasses.replace(sc, student_id=-1))
    with pytest.raises(h.GuiError):
        h.run_plan(ctx, sc, algo="dfs")
    with pytest.raises(h.GuiError):
        h.run_csp_modes([1, 2], [3], 2)


def test_unreachable_target_gives_no_solution_message(ctx, sc):
    out = h.run_plan(ctx, dataclasses.replace(sc, target=100.0, spw=1, weeks=(1, 1, 1)))
    assert out.result.status == "NO_SOLUTION" and "NO SOLUTION" in out.result.message
    assert out.shown["best_effort"] and len(h.per_assignment_table(out)) == 3


def test_zero_effort_when_target_already_met(ctx, sc):
    out = h.run_plan(ctx, dataclasses.replace(sc, target=0.0))
    assert out.ok and out.result.allocation == [0, 0, 0] and out.result.schedule == []


def test_resolve_scenario_reads_state(ctx):
    sid = ctx.profiles[3].student_id
    state = {"sel_student": sid, f"target_{sid}": 71, "sel_spw": 4, "sel_goal": "mean", f"weeks_{sid}_1": 9}
    s = h.resolve_scenario(ctx, state, False)
    assert (s.student_id, s.target, s.spw, s.goal, s.weeks[1], s.nn_on) == (sid, 71.0, 4, "mean", 9, False)


def test_nn_metrics_use_eval_module(ctx):
    from eval.metrics import nn_vs_baselines
    live = h.nn_live_metrics(ctx).set_index("Model")
    ref = nn_vs_baselines(ctx.model, ctx.train, ctx.test)
    assert np.isclose(live.loc["NN (12-16-8-1)", "RMSE"], ref["NN (12-16-8-1)"]["RMSE"])
    df = h.nn_prediction_frame(ctx)
    assert len(df) == len(ctx.test) and {"Actual", "NN", "Average baseline"} <= set(df.columns)


def test_load_results_handles_missing_and_corrupt(tmp_path):
    (tmp_path / "exp1_search.csv").write_text("\x00\x01 not,a\ncsv")
    r = h.load_results(tmp_path)
    assert set(r) == set(h.RESULT_FILES) and all(v is None or isinstance(v, pd.DataFrame) for v in r.values())
    assert r["exp2"] is None and h.stored_seed_count(r) is None


def test_stored_results_have_at_least_five_seeds_and_expected_columns():
    r = h.load_results()
    assert h.stored_seed_count(r) >= 5
    assert {"expanded_mean", "expanded_std", "cost_mean"} <= set(r["exp1"].columns)
    assert set(r["exp2"].slots_per_week) == {2, 3, 4} and set(r["exp2"].solver) == {"BT", "BT+FC", "BT+FC+AC3"}
    assert {"reaches_target_mean", "reaches_target_std"} <= set(r["exp3"].columns)
    t = h.mean_std_table(r["exp1"], ["algo"], ["expanded"])
    assert t.expanded.str.contains("±").all()
