"""Page 2 - Search algorithms visualisation."""
from __future__ import annotations

import streamlit as st

from config import K_MAX
from gui import charts, helpers as h, ui
from gui.views import scenario

PRIORITY = {"ucs": "f = g  (sessions used so far). Optimal but blind: it expands every cheaper plan first.",
            "greedy": "f = h  (total marks still missing below the target). Fast, but ignores cost so far and is not guaranteed optimal.",
            "astar": "f = g + h  (h = sessions still needed per assignment). Optimal when predictions are monotone, and far fewer expansions."}


def render(ctx, nn_on: bool) -> None:
    ui.page_header("Search algorithms", "How UCS, Greedy Best-First and A* explore session allocations. A state is the number of "
                   "sessions on each assignment; each action adds one session (cost 1).")
    sc = scenario.current(ctx, nn_on)
    csp_mode = st.session_state.get("plan_csp", "Backtracking + Forward Checking + AC-3")
    try:
        outs = h.compare_algorithms(ctx, sc, csp_mode)
    except h.GuiError as e:
        st.error(str(e)); return
    summ = h.search_summary(outs)

    st.markdown("#### Algorithm comparison on this scenario")
    show = summ.drop(columns=["key"]).copy()
    show["Allocation"] = show["Allocation"].astype(str)
    ui.show_table(show)
    ui.show_chart(charts.search_metric_bars(summ), "search_bars")
    ok = summ[summ.Status == "OK"]
    if len(ok) and ok["Solution cost (sessions)"].nunique() == 1:
        st.caption(f"All algorithms that solved it found the same optimal cost ({int(ok['Solution cost (sessions)'].iloc[0])} sessions); "
                   "they differ in how many nodes they had to expand and generate.")
    elif len(ok):
        st.caption("Costs differ: Greedy is not guaranteed optimal.")
    if not nn_on:
        ui.banner("warn", "NN is OFF: the average-mark predictor does not react to extra sessions, so the search typically "
                          "either succeeds at 0 sessions or cannot reach the target.")

    st.markdown("#### Step-by-step trace")
    algo = st.radio("Algorithm to trace", list(h.ALGO_LABELS), horizontal=True, format_func=h.ALGO_LABELS.get, key="search_algo")
    out = outs[algo]; res = out.result
    st.markdown(f"**Priority:** {PRIORITY[algo]}")
    c = st.columns(5)
    ui.metric_card(c[0], "Status", "Solved" if out.ok else "No solution", "", "good" if out.ok else "bad")
    ui.metric_card(c[1], "Solution cost", res.cost if res.cost is not None else "—", "sessions in total", "accent")
    ui.metric_card(c[2], "Expanded", res.search.expanded, "nodes taken off the frontier")
    ui.metric_card(c[3], "Generated", res.search.generated, "nodes put on the frontier")
    ui.metric_card(c[4], "NN ok but CSP rejected", res.search.csp_rejected, f"{res.csp_calls} CSP calls", "bad" if res.search.csp_rejected else "")
    st.write("")
    if out.ok:
        st.success(f"Final allocation {res.allocation} → predicted marks {[round(m, 1) for m in res.marks]} (cost {res.cost}).")
    else:
        st.error(res.message)

    trace = h.trace_frame(res)
    if trace.empty:
        ui.note("The search stopped before expanding any node: A* proved from its heuristic that the target is unreachable "
                "for at least one assignment (h = ∞ at the start state), so there is nothing to trace.")
        return
    if res.search.expanded > len(trace):
        st.caption(f"Trace limited to the first {len(trace):,} of {res.search.expanded:,} expansions (limit set in search/__init__.py).")
    n = len(trace)
    step = st.slider("Step", 1, n, n, key=f"step_{algo}_{n}") if n > 1 else 1
    row = trace.iloc[step - 1]
    ui.note(f"Step {step}: expanded state {row['State (sessions per assignment)']} · g = {row['g (sessions so far)']} · "
            f"priority = {row['Priority f']} · predicted marks {row['Predicted marks']} · {row['Outcome']}.")
    st.write("")
    upto = trace[trace["Step"] <= step]
    tabs = st.tabs(["Search space", "Slice view", "Progress", "Trace table"])
    with tabs[0]:
        if len(res.table) == 3:
            ui.show_chart(charts.search_space_3d(upto, K_MAX), f"s3d_{algo}")
            st.caption("Each dot is an expanded state (x/y/z = sessions on A1/A2/A3), shaded by expansion order. "
                       "Generated-but-unexpanded nodes are counted in the cards above; the project does not record them individually.")
        else:
            st.info("The 3-D view needs exactly three assignments.")
    with tabs[1]:
        if len(res.table) == 3:
            k3 = st.select_slider("Sessions on A3", list(range(K_MAX + 1)), value=int(row["A3"]), key=f"k3_{algo}_{n}_{step}")
            ui.show_chart(charts.search_slice_heatmap(upto, k3, K_MAX), f"slice_{algo}")
        else:
            st.info("The slice view needs exactly three assignments.")
    with tabs[2]:
        ui.show_chart(charts.search_progress(trace, step), f"prog_{algo}")
    with tabs[3]:
        ui.show_table(upto.drop(columns=["A1", "A2", "A3"], errors="ignore").astype({"Predicted marks": str}), height=320)
