"""The three experiments + NN evaluation. Run from the project root:
   python -m eval.experiments --seeds 5     -> results/*.csv|md|png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from config import *
from data.dataset_loader import load_processed, split_by_presentation, make_profiles
from nn.train import fit_model
from nn.baseline import AvgMarkBaseline
from nn.predict import NNPredictor, AvgPredictor
from csp import StudyCSP, CSPConfig, capacity_ok
from integration.planner import plan
from eval.metrics import nn_vs_baselines, monotonicity_report
from eval import plots


def target_rule(table) -> float:
    """GENERATED target mark: 3 marks above the student's WEAKEST predicted assignment (no extra effort),
    clipped to [55, 90]. Every student therefore needs some effort on at least one assignment, and
    "each" goal = all three assignments must reach this mark."""
    return float(np.clip(round(min(r[0] for r in table) + 3), 55, 90))


def planner_batch(profiles, predictor, spw=SLOTS_PER_WEEK, algo="astar", goal="each", ref_predictor=None):
    """Planner output (allocation, predicted marks; timetable inside plan()) for a list of student profiles."""
    rows = []
    for p in profiles:
        tgt = target_rule((ref_predictor or predictor).marks_table(p))
        r = plan(p, predictor, tgt, spw, algo, goal)
        rows.append({"student": p.student_id, "target": tgt, "status": r.status, "weeks_avail": p.weeks_avail,
                     "allocation": r.allocation, "total_sessions": r.cost,
                     "pred_marks": [round(m, 1) for m in (r.marks or r.best_effort["marks"])],
                     "timetable": r.schedule, "expanded": r.search.expanded, "csp_rejected": r.search.csp_rejected})
    return pd.DataFrame(rows)


def _profiles_for_required_planner(df_test, n_profiles, seed):
    profiles = make_profiles(df_test, n_profiles, seed)
    if len(profiles) < n_profiles:
        raise ValueError(f"Need {n_profiles} eligible held-out student profiles, but found {len(profiles)}. Choose a course/presentation with at least {n_profiles} students having three scored assessments.")
    return profiles


def exp1_search(df_train, df_test, seeds, n_profiles=30, goal="each"):
    """UCS vs Greedy vs A*: nodes expanded and plan cost."""
    out = []
    for s in seeds:
        pred = NNPredictor(fit_model(df_train, seed=s)); profs = _profiles_for_required_planner(df_test, n_profiles, s)
        for algo in ["ucs", "greedy", "astar"]:
            b = planner_batch(profs, pred, algo=algo, goal=goal); ok = b[b.status == "OK"]
            out.append({"seed": s, "algo": algo, "expanded": b.expanded.mean(),
                        "expanded_solved_only": ok.expanded.mean() if len(ok) else np.nan,
                        "cost": ok.total_sessions.mean() if len(ok) else np.nan, "solved_rate": len(ok) / len(b)})
    return pd.DataFrame(out)


def exp2_csp(seeds, n_inst=10, max_steps=20000):
    """BT vs BT+FC vs BT+FC+AC3 with 2, 3, 4 slots per week on random instances."""
    cfgs = [CSPConfig(False, False, max_steps), CSPConfig(True, False, max_steps), CSPConfig(True, True, max_steps)]
    out = []
    for s in seeds:
        rng = np.random.default_rng(s)
        for spw in [2, 3, 4]:
            insts = [(rng.integers(2, 9, 3).tolist(), np.sort(rng.integers(2, 12, 3)).tolist()) for _ in range(n_inst)]
            for cfg in cfgs:
                a = b = c = fc_removed = ac3_removed = cut = solved = 0
                for n, w in insts:
                    r = StudyCSP(n, w, spw).solve(cfg)
                    a += r.stats.assignments; b += r.stats.backtracks; c += r.stats.checks
                    # Keep FC and AC-3 pruning separate so the effect of each switch is visible.
                    fc_removed += r.stats.fc_prunings; ac3_removed += r.stats.ac3_prunings
                    cut += r.status == "cutoff"; solved += r.status == "solved"
                    if r.status != "cutoff":
                        assert (r.status == "solved") == capacity_ok(n, w, spw)   # cross-check vs exact test
                out.append({"seed": s, "slots_per_week": spw, "solver": cfg.name, "assignments": a / n_inst,
                            "backtracks": b / n_inst, "checks": c / n_inst,
                            "fc_prunings": fc_removed / n_inst, "ac3_prunings": ac3_removed / n_inst,
                            "cutoff_rate": cut / n_inst, "solved_rate": solved / n_inst})
    return pd.DataFrame(out)


def exp3_nn_vs_avg(df_train, df_test, seeds, n_profiles=30, goal="each"):
    """NN versus the simple average-mark predictor inside the planner (same target per student)."""
    out = []
    for s in seeds:
        nn = NNPredictor(fit_model(df_train, seed=s)); avg = AvgPredictor(AvgMarkBaseline.fit(df_train))
        profs = _profiles_for_required_planner(df_test, n_profiles, s)
        for name, pr in [("NN", nn), ("AvgMark", avg)]:
            b = planner_batch(profs, pr, goal=goal, ref_predictor=nn); ok = b[b.status == "OK"]
            out.append({"seed": s, "predictor": name, "reaches_target": len(ok) / len(b),
                        "sessions_when_solved": ok.total_sessions.mean() if len(ok) else np.nan,
                        "plans_needing_effort": float((ok.total_sessions > 0).mean()) if len(ok) else 0.0})
    return pd.DataFrame(out)


def summarize(d, by, cols):
    g = d.groupby(by)[cols].agg(["mean", "std"]); g.columns = [f"{c}_{s}" for c, s in g.columns]; return g.reset_index().round(3)


def save_table(d, name, R):
    d.to_csv(R / f"{name}.csv", index=False); cols = list(d.columns)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(map(str, r)) + " |" for r in d.values]
    (R / f"{name}.md").write_text("\n".join(md) + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/processed/oulad_processed.csv")
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--goal", default="each", choices=["each", "mean"])
    ap.add_argument("--out", default="results"); a = ap.parse_args()
    R = Path(a.out); R.mkdir(exist_ok=True); seeds = list(range(a.seeds))
    df = load_processed(a.data); tr, te, order = split_by_presentation(df)
    print(f"train presentations={order[:-1]} test={order[-1]} | train rows={len(tr)} test rows={len(te)}")
    rows, mono = [], []
    for s in seeds:
        m = fit_model(tr, seed=s)
        for name, mt in nn_vs_baselines(m, tr, te).items(): rows.append({"seed": s, "model": name, **mt})
        mono.append({"seed": s, **monotonicity_report(m, _profiles_for_required_planner(te, 30, s))})
    save_table(summarize(pd.DataFrame(rows), "model", ["MAE", "RMSE", "R2"]), "nn_results", R)
    save_table(pd.DataFrame(mono).round(3), "monotonicity", R)
    e1 = exp1_search(tr, te, seeds, goal=a.goal)
    save_table(summarize(e1, "algo", ["expanded", "expanded_solved_only", "cost", "solved_rate"]), "exp1_search", R)
    plots.bar_with_err(e1, "algo", "algo", "expanded", R / "exp1_nodes.png", "Exp1: nodes expanded", "nodes expanded")
    plots.bar_with_err(e1.dropna(subset=["cost"]), "algo", "algo", "cost", R / "exp1_cost.png", "Exp1: plan cost", "sessions")
    e2 = exp2_csp(seeds)
    save_table(summarize(e2, ["slots_per_week", "solver"], ["assignments", "backtracks", "checks", "fc_prunings", "ac3_prunings", "cutoff_rate"]), "exp2_csp", R)
    plots.bar_with_err(e2, "slots_per_week", "solver", "assignments", R / "exp2_assignments.png", "Exp2: assignments", "assignments")
    plots.bar_with_err(e2, "slots_per_week", "solver", "checks", R / "exp2_checks.png", "Exp2: consistency checks", "checks")
    plots.bar_with_err(e2, "slots_per_week", "solver", "backtracks", R / "exp2_backtracks.png", "Exp2: backtracks", "backtracks")
    plots.bar_with_err(e2, "slots_per_week", "solver", "fc_prunings", R / "exp2_fc_prunings.png", "Exp2: Forward Checking prunings", "values removed")
    plots.bar_with_err(e2, "slots_per_week", "solver", "ac3_prunings", R / "exp2_ac3_prunings.png", "Exp2: AC-3 prunings", "values removed")
    e3 = exp3_nn_vs_avg(tr, te, seeds, goal=a.goal)
    save_table(summarize(e3, "predictor", ["reaches_target", "sessions_when_solved", "plans_needing_effort"]), "exp3_nn_vs_avg", R)
    plots.bar_with_err(e3, "predictor", "predictor", "reaches_target", R / "exp3_reach.png", "Exp3: targets reached", "fraction")
    for n in ["nn_results", "monotonicity", "exp1_search", "exp2_csp", "exp3_nn_vs_avg"]:
        print("\n==", n, "==\n" + (R / f"{n}.md").read_text())


if __name__ == "__main__":
    main()
