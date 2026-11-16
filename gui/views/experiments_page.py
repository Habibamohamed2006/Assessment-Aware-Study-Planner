"""Page 5 - Experiments and results (reads the project's stored result files)."""
from __future__ import annotations

import streamlit as st

from gui import charts, helpers as h, ui
from gui.helpers import RESULTS_DIR

CMD = "python -m eval.experiments --seeds 5 --goal each"
SOLVER = {"BT": "BT", "BT+FC": "BT+FC", "BT+FC+AC3": "BT+FC+AC-3"}


def _missing(name):
    st.info(f"`results/{name}` was not found. Generate it from the project root with `{CMD}`.")


def render(ctx, nn_on: bool) -> None:
    ui.page_header("Experiments & results", "Tables and charts read from the project's stored result files in results/. "
                   "Bars show the mean, whiskers one standard deviation across seeds.")
    res = h.load_results()
    k = h.stored_seed_count(res)
    ui.chips([f"Seeds: {k if k else 'unknown'}", "Source: results/*.csv", f"Regenerate: {CMD}"])
    if k is not None and k < 5:
        st.warning(f"Only {k} seeds found; the assignment requires at least 5.")
    tabs = st.tabs(["1 · Search", "2 · CSP", "3 · Predictors"])
    with tabs[0]:
        _exp1(ctx, res)
    with tabs[1]:
        _exp2(res)
    with tabs[2]:
        _exp3(res)


def _png(name, caption):
    p = RESULTS_DIR / name
    if p.exists():
        with st.expander(f"Original project figure: {name}"):
            st.image(str(p), caption=caption)


def _exp1(ctx, res):
    st.markdown("#### Experiment 1: UCS vs Greedy vs A*")
    d = res["exp1"]
    if d is None:
        _missing("exp1_search.csv"); return
    ui.show_table(h.mean_std_table(d, ["algo"], ["expanded", "expanded_solved_only", "cost", "solved_rate"], 3))
    st.caption("`expanded` averages over all profiles (including unsolvable ones, where A* stops immediately); "
               "`expanded_solved_only` averages only profiles that were solved. Cost is averaged over solved profiles.")
    a, b = st.columns(2)
    names = {"ucs": "UCS", "greedy": "Greedy", "astar": "A*"}
    with a:
        ui.show_chart(charts.mean_std_bars(d, "algo", None, "expanded_mean", "expanded_std", "Nodes expanded (all profiles)", "nodes",
                                           charts.ALGO_COLORS, names), "e1_exp")
    with b:
        ui.show_chart(charts.mean_std_bars(d, "algo", None, "expanded_solved_only_mean", "expanded_solved_only_std",
                                           "Nodes expanded (solved profiles only)", "nodes", charts.ALGO_COLORS, names), "e1_exp_ok")
    a, b = st.columns(2)
    with a:
        ui.show_chart(charts.mean_std_bars(d, "algo", None, "cost_mean", "cost_std", "Solution cost (sessions)", "sessions",
                                           charts.ALGO_COLORS, names), "e1_cost")
    with b:
        ui.show_chart(charts.mean_std_bars(d, "algo", None, "solved_rate_mean", "solved_rate_std", "Fraction of profiles solved", "fraction",
                                           charts.ALGO_COLORS, names), "e1_solved")
    m = d.set_index("algo")
    if {"ucs", "astar"} <= set(m.index) and m.loc["astar", "expanded_solved_only_mean"] > 0:
        st.success(f"On solved profiles A* expanded {m.loc['astar', 'expanded_solved_only_mean']:.1f} nodes on average vs "
                   f"{m.loc['ucs', 'expanded_solved_only_mean']:.1f} for UCS "
                   f"({m.loc['ucs', 'expanded_solved_only_mean'] / m.loc['astar', 'expanded_solved_only_mean']:.1f}× more), at the same mean cost.")
    st.markdown("##### Generated nodes")
    ui.note("The stored exp1_search.csv has no generated-node column (the experiment script only records expanded nodes and cost). "
            "The button below re-runs the same experiment live through the project's own planner and adds generated nodes. "
            "It takes roughly a minute and writes nothing to disk.")
    if st.button("Compute generated nodes (5 seeds)"):
        with st.spinner("Running Experiment 1 for 5 seeds…"):
            st.session_state["exp1_live"] = h.exp1_with_generated(ctx.seed, 5)
    live = st.session_state.get("exp1_live")
    if live:
        per_seed, summ = live
        ui.show_table(h.mean_std_table(summ, ["algo"], ["expanded", "generated", "cost", "solved_rate"], 3))
        ui.show_chart(charts.mean_std_bars(summ, "algo", None, "generated_mean", "generated_std", "Nodes generated (live, 5 seeds)", "nodes",
                                           charts.ALGO_COLORS, names), "e1_gen")
        same = all(abs(summ.set_index("algo").loc[a, "expanded_mean"] - d.set_index("algo").loc[a, "expanded_mean"]) < 0.01
                   for a in d["algo"] if a in set(summ["algo"]))
        st.caption("Check: live expanded-node means " + ("match" if same else "DIFFER from") + " the stored file.")
        with st.expander("Per-seed values"):
            ui.show_table(per_seed.round(3))
    _png("exp1_nodes.png", "Project figure")


def _exp2(res):
    st.markdown("#### Experiment 2: BT vs BT+FC vs BT+FC+AC-3 (2, 3, 4 slots per week)")
    d = res["exp2"]
    if d is None:
        _missing("exp2_csp.csv"); return
    shown = d.copy(); shown["solver"] = shown["solver"].map(SOLVER).fillna(shown["solver"])
    ui.show_table(h.mean_std_table(shown, ["slots_per_week", "solver"], ["assignments", "backtracks", "checks", "fc_prunings", "ac3_prunings", "cutoff_rate"], 2))
    spw_sel = st.multiselect("Slots per week", sorted(d["slots_per_week"].unique()), default=sorted(d["slots_per_week"].unique()), key="e2_spw")
    s = shown[shown["slots_per_week"].isin(spw_sel)]
    colors = {"BT": charts.GREY, "BT+FC": charts.PURPLE, "BT+FC+AC-3": charts.NAVY}
    for (metric, title, log), col in zip([("assignments", "Assignments", False), ("backtracks", "Backtracks", False),
                                          ("checks", "Consistency checks (log scale)", True), ("fc_prunings", "Forward-checking prunings", False)],
                                         st.columns(2) * 2):
        with col:
            ui.show_chart(charts.mean_std_bars(s, "slots_per_week", "solver", f"{metric}_mean", f"{metric}_std", title, title.split(" (")[0].lower(),
                                               colors, None, log), f"e2_{metric}")
    st.caption("Plain BT and BT+FC make the same assignments and backtracks (forward checking only cuts work per step); AC-3 prunes first. "
               "With only 2 slots/week many instances are infeasible and BT can hit the step cut-off: see cutoff_rate in the table. "
               "Standard deviations are across seeds; each seed averages over 10 random instances.")
    _png("exp2_ac3_prunings.png", "Project figure")


def _exp3(res):
    st.markdown("#### Experiment 3: Neural network vs average predictor inside the planner")
    d, nn = res["exp3"], res["nn"]
    if d is None:
        _missing("exp3_nn_vs_avg.csv")
    else:
        names = {"NN": "NN", "AvgMark": "Average predictor"}
        ui.show_table(h.mean_std_table(d, ["predictor"], ["reaches_target", "sessions_when_solved", "plans_needing_effort"], 3))
        cols = st.columns(3)
        for col, (m, t) in zip(cols, [("reaches_target", "Targets reached (fraction)"), ("sessions_when_solved", "Sessions when solved"),
                                      ("plans_needing_effort", "Plans that needed effort")]):
            with col:
                ui.show_chart(charts.mean_std_bars(d, "predictor", None, f"{m}_mean", f"{m}_std", t, "", {"NN": charts.PURPLE, "AvgMark": charts.AMBER}, names), f"e3_{m}")
        st.caption("Targets come from the NN's own rule, and the same target is given to both predictors. The average predictor ignores effort, "
                   "so it can only 'reach' a target that its flat prediction already meets with 0 sessions (see plans_needing_effort = 0).")
    st.markdown("##### Prediction performance (held-out presentation)")
    if nn is None:
        _missing("nn_results.csv")
    else:
        ui.show_table(h.mean_std_table(nn, ["model"], ["MAE", "RMSE", "R2"], 3))
        cols = st.columns(3)
        for col, m in zip(cols, ["MAE", "RMSE", "R2"]):
            with col:
                ui.show_chart(charts.mean_std_bars(nn, "model", None, f"{m}_mean", f"{m}_std", m, m,
                                                   {x: (charts.PURPLE if x.startswith("NN") else charts.GREY) for x in nn["model"]}), f"nn_{m}")
    if res["mono"] is not None:
        with st.expander("Raw NN monotonicity per seed"):
            ui.show_table(res["mono"])
    _png("exp3_reach.png", "Project figure")
