"""Page 3 - CSP scheduling visualisation."""
from __future__ import annotations

import streamlit as st

from config import K_MAX
from gui import charts, helpers as h, ui
from gui.views import scenario

SOURCES = ["Planner allocation (current scenario)", "Custom session counts", "Stress example (infeasible, shows pruning)"]


def _instance(ctx, sc, nn_on):
    src = st.radio("Sessions to schedule", SOURCES, horizontal=True, key="csp_source")
    weeks, spw = list(sc.weeks), sc.spw
    if src == SOURCES[0]:
        algo = st.session_state.get("plan_algo", "astar")
        out = h.run_plan(ctx, sc, algo, st.session_state.get("plan_csp", "Backtracking + Forward Checking + AC-3"))
        counts = out.shown["allocation"]
        if out.shown["best_effort"]:
            st.warning("The planner found no plan that reaches the target, so the best-effort allocation is scheduled here.")
    elif src == SOURCES[1]:
        cols = st.columns(len(weeks))
        counts = [cols[i].number_input(f"A{i + 1} sessions", 0, K_MAX, 2, key=f"csp_cnt_{i}") for i in range(len(weeks))]
    else:
        e = h.STRESS_EXAMPLE
        counts, weeks, spw = list(e["counts"]), list(e["weeks"]), e["spw"]
        ui.note(f"Fixed illustrative instance: sessions {counts}, deadlines {weeks} weeks, {spw} slots/week. "
                "Total capacity is not enough, so every solver must fail; plain backtracking explores a huge tree before it learns that.")
    return [int(c) for c in counts], weeks, spw


def render(ctx, nn_on: bool) -> None:
    ui.page_header("CSP scheduling", "Variables are study sessions, domains are weekly slots that exist before the assignment's deadline. "
                   "Constraints: one session per slot, and sessions of one assignment stay in order.")
    sc = scenario.current(ctx, nn_on)
    try:
        counts, weeks, spw = _instance(ctx, sc, nn_on)
        runs = h.run_csp_modes(counts, weeks, spw)
    except h.GuiError as e:
        st.error(str(e)); return
    summ = h.csp_summary(runs, counts, weeks, spw)

    mode = st.radio("CSP method", list(h.CSP_MODES), horizontal=True, key="csp_mode")
    d = runs[mode]; r = d["result"]; stt = r.stats
    cap = h.capacity_report(counts, weeks, spw)
    if r.status == "solved":
        ui.banner("ok", f"Valid schedule found for {sum(counts)} sessions (checked by the independent validator: {d['valid']}).")
    elif r.status == "cutoff":
        ui.banner("warn", f"Stopped after {h.GUI_CSP_MAX_STEPS:,} assignments without an answer ({r.message}). Try a method with pruning.")
    else:
        ui.banner("fail", r.message or "No valid schedule exists.")
    c = st.columns(5)
    ui.metric_card(c[0], "Assignments", len(counts), f"{sum(counts)} sessions: {counts}", "accent")
    ui.metric_card(c[1], "Search steps (assignments)", stt.assignments, "search effort")
    ui.metric_card(c[2], "Backtracks", stt.backtracks, "assignments undone", "bad" if stt.backtracks else "good")
    ui.metric_card(c[3], "Valid schedule", "Yes" if r.status == "solved" else "No", r.status, "good" if r.status == "solved" else "bad")
    ui.metric_card(c[4], "Values pruned", stt.fc_prunings + stt.ac3_prunings, f"FC {stt.fc_prunings} · AC-3 {stt.ac3_prunings}")
    st.write("")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("##### Weekly slots and assigned sessions")
        if d["timetable"]:
            ui.show_chart(charts.timetable_heatmap(h.timetable_grid(d["timetable"], weeks, spw)), f"csp_tt_{mode}")
        else:
            empty = h.timetable_grid([], weeks, spw)
            ui.show_chart(charts.timetable_heatmap(empty), f"csp_tt_{mode}")
            st.caption("No timetable: this method did not produce a schedule. All slots are shown empty.")
    with right:
        st.markdown("##### Deadlines and capacity")
        ui.show_table(cap)
        ui.show_chart(charts.capacity_chart(cap), "csp_cap")
        st.caption("Exact feasibility rule: by each deadline, cumulative sessions must not exceed cumulative slots (csp.constraints.capacity_ok).")

    st.markdown("##### Removed / pruned domains")
    if h.CSP_MODES[mode][1]:
        dom = h.domain_frame(r)
        ui.show_chart(charts.csp_domain_bars(dom), "csp_dom")
        with st.expander("Domain table"):
            ui.show_table(dom)
        st.caption(f"AC-3 removed {stt.ac3_prunings} slot values in {stt.ac3_revisions} arc revisions before any search."
                   + (" It emptied a domain, which proves there is no schedule without any backtracking." if r.status == "no_solution" and stt.assignments == 0 else ""))
    else:
        ui.note("AC-3 is not active in this mode, so no domain is pruned in advance." +
                (f" Forward checking removed {stt.fc_prunings} values dynamically during search." if h.CSP_MODES[mode][0] else ""))

    st.markdown("#### How Forward Checking and AC-3 reduce the search")
    ui.show_table(summ[["Mode", "Status", "Assignments", "Backtracks", "Consistency checks", "FC prunings", "AC-3 prunings"]])
    a, b = st.columns([3, 2])
    with a:
        ui.show_chart(charts.csp_mode_comparison(summ), "csp_cmp")
    with b:
        ui.show_chart(charts.csp_pruning_bars(summ), "csp_prune")
    st.caption("Forward checking keeps the same assignments/backtracks as plain backtracking but needs far fewer consistency checks; "
               "AC-3 prunes before search and can cut assignments and backtracks. Consistency-check axis is logarithmic. "
               "Stored results for 5 seeds are on the Experiments page.")

    st.markdown("---")
    _rejection_example(ctx)


def _rejection_example(ctx) -> None:
    st.markdown("### Example: the NN says “target reached”, the CSP says “impossible”")
    st.write("The NN only looks at marks. It does not know that sessions need free slots before the deadline. "
             "Here a real student is given very tight deadlines and 2 slots per week.")
    c1, c2, c3, c4 = st.columns(4)
    w = (c1.number_input("A1 deadline (weeks)", 1, h.MAX_WEEKS, 1, key="rej_w0"), c2.number_input("A2 deadline", 1, h.MAX_WEEKS, 6, key="rej_w1"),
         c3.number_input("A3 deadline", 1, h.MAX_WEEKS, 10, key="rej_w2"))
    spw = c4.number_input("Slots per week", 1, h.MAX_SPW, 2, key="rej_spw")
    ex = h.cached_rejection_example(ctx.seed, tuple(int(x) for x in w), int(spw))
    if ex is None:
        ui.note("No rejection case was found among the first 300 profiles for these deadlines/slots. "
                "Try shorter deadlines or fewer slots. (tests/test_integration.py also holds a deterministic CSP-rejection case.)")
        return
    sc, s, marks, res = ex["scenario"], ex["state"], ex["marks"], ex["result"]
    c = st.columns(4)
    ui.metric_card(c[0], "Student", sc.student_id, f"goal: {sc.goal}", "accent")
    ui.metric_card(c[1], "NN-proposed sessions", str(list(s)), f"total {sum(s)}")
    ui.metric_card(c[2], "NN-predicted marks", str([round(m, 1) for m in marks]), f"target {sc.target:g} → NN says reached", "good")
    ui.metric_card(c[3], "CSP verdict", "Rejected", f"{ex['n_rejected']} plan(s) rejected during search", "bad")
    st.write("")
    a, b = st.columns(2)
    with a:
        ui.show_table(ex["capacity"])
        st.caption("Rows with “Enough slots = False” are the violated deadline capacities.")
    with b:
        ui.show_chart(charts.capacity_chart(ex["capacity"]), "rej_cap")
    ui.show_table(h.csp_summary(ex["runs"], s, sc.weeks, sc.spw)[["Mode", "Status", "Assignments", "Backtracks", "Consistency checks", "Message"]])
    if res.status == "OK":
        st.success(f"The planner recovered: it kept searching and found a schedulable allocation {res.allocation} "
                   f"(predicted marks {[round(m, 1) for m in res.marks]}).")
    else:
        st.error(res.message)
