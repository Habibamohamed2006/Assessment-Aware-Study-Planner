"""Page 1 - Study Planner Dashboard."""
from __future__ import annotations

import streamlit as st

from gui import charts, helpers as h, ui
from config import K_MAX


def _reset_target(sid, value):
    st.session_state[f"target_{sid}"] = value


def render(ctx, nn_on: bool) -> None:
    ui.hero("Assessment-Aware Study Planner",
            "Neural-network predictions, search and constraint satisfaction working together: the NN estimates how many "
            "extra study sessions each assignment needs to reach your target mark, Search finds the cheapest allocation, "
            "and the CSP checks that those sessions fit into your weekly slots before every deadline.")
    ss = st.session_state
    ids = [p.student_id for p in ctx.profiles]
    ss.setdefault("sel_student", h.pick_default_student(ctx))
    ss.setdefault("sel_spw", 3)
    ss.setdefault("sel_goal", "each")
    ss.setdefault("plan_algo", "astar")
    ss.setdefault("plan_csp", "Backtracking + Forward Checking + AC-3")

    st.markdown("#### 1 · Student, target and capacity")
    c1, c2, c3 = st.columns([2, 1, 1])
    sid = c1.selectbox("Student profile", ids, key="sel_student", format_func=lambda x: f"Student {x}",
                       help=f"{len(ids):,} real held-out students (presentation {ctx.order[-1]}) with three scored assessments.")
    profile = ctx.profile(sid)
    auto_t = h.reference_target(ctx, profile)
    ss.setdefault(f"target_{sid}", auto_t)
    c2.number_input("Target mark", min_value=0.0, max_value=100.0, step=1.0, key=f"target_{sid}",
                    help="Auto value = project target rule: 3 marks above the weakest NN-predicted assignment, clipped to [55, 90].")
    c3.number_input("Study slots per week", min_value=1, max_value=h.MAX_SPW, step=1, key="sel_spw",
                    help="How many study sessions fit in one week (a design assumption, not an OULAD observation).")
    d1, d2 = st.columns([2, 1])
    d1.selectbox("Goal", list(h.GOAL_LABELS), key="sel_goal", format_func=h.GOAL_LABELS.get)
    d2.button("Reset target to rule", on_click=_reset_target, args=(sid, auto_t),
              help=f"Rule target for this student: {auto_t:g}")

    st.markdown("#### 2 · Assignments and deadlines")
    cols = st.columns(len(profile.weeks_avail))
    for i, (col, w) in enumerate(zip(cols, profile.weeks_avail)):
        ss.setdefault(f"weeks_{sid}_{i}", int(w))
        col.number_input(f"A{i + 1}: weeks until deadline", min_value=1, max_value=h.MAX_WEEKS, step=1, key=f"weeks_{sid}_{i}")
    sc = h.resolve_scenario(ctx, ss, nn_on)
    ui.show_table(h.assignment_info(ctx, sc))
    st.caption("Deadlines default to the real course deadlines, measured from 4 weeks before the first one. "
               "The actual mark column is shown for reference only; the planner never sees it.")

    with st.expander("Advanced: search algorithm and CSP method used by the planner"):
        a1, a2 = st.columns(2)
        a1.selectbox("Search algorithm", list(h.ALGO_LABELS), key="plan_algo", format_func=h.ALGO_LABELS.get)
        a2.selectbox("CSP method", list(h.CSP_MODES), key="plan_csp")

    if not nn_on:
        ui.banner("warn", "Neural network is OFF: the average-mark predictor is used. It ignores study effort, so extra sessions "
                          "cannot raise predicted marks. The target is still derived from the NN so both modes chase the same goal.")

    if st.button("Generate Study Plan", type="primary"):
        try:
            ss["plan_out"] = h.run_plan(ctx, sc, ss["plan_algo"], ss["plan_csp"])
        except h.GuiError as e:
            ss.pop("plan_out", None)
            st.error(str(e))
        except Exception as e:                        # never show a traceback to the end user
            ss.pop("plan_out", None)
            st.error(f"The planner failed unexpectedly: {e}")

    out = ss.get("plan_out")
    if out is None:
        ui.note("Choose a student and click “Generate Study Plan”. Everything shown afterwards is computed live by the project's "
                "Search, NN and CSP modules.")
        return
    if out.scenario != sc or out.algo != ss["plan_algo"] or out.csp_mode != ss["plan_csp"]:
        st.info("Inputs changed since this plan was generated. Click “Generate Study Plan” to refresh.")
    _results(out)


def _results(out: h.PlanOutcome) -> None:
    r, sc, shown = out.result, out.scenario, out.shown
    st.markdown("---")
    st.markdown("### Study plan")
    ui.chips([f"Student {sc.student_id}", f"Predictor: {out.predictor_name}", f"Search: {h.ALGO_LABELS[out.algo]}",
              f"CSP: {out.csp_mode}", f"{sc.spw} slots/week"])
    tbl = h.per_assignment_table(out)
    marks = shown["marks"]
    if out.ok:
        ui.banner("ok", f"Target achieved: predicted marks {[round(m, 1) for m in marks]} satisfy the goal "
                        f"“{h.GOAL_LABELS[sc.goal]}” (target {sc.target:g}), and the CSP found a valid timetable.")
    else:
        ui.banner("fail", r.message)

    c = st.columns(5)
    ui.metric_card(c[0], "Target achieved", "Yes" if out.ok else "No", f"target {sc.target:g}", "good" if out.ok else "bad")
    ui.metric_card(c[1], "Total sessions", sum(shown["allocation"]), "best-effort (not a solution)" if shown["best_effort"] else
                   f"max {K_MAX} per assignment", "accent")
    ui.metric_card(c[2], "Mean predicted mark", f"{sum(marks) / len(marks):.1f}", f"min {min(marks):.1f}", "accent")
    ui.metric_card(c[3], "Search nodes expanded", r.search.expanded, f"{r.search.generated} generated", "")
    ui.metric_card(c[4], "CSP checks", r.csp_calls, f"{r.search.csp_rejected} rejected by CSP", "")
    st.write("")
    if shown["best_effort"]:
        ui.note("Because no plan reaches the target, the table and timetable below show the planner's BEST-EFFORT allocation "
                "(highest predicted marks that fit the slot capacity). It does not meet the target.")
    left, right = st.columns([3, 2])
    with left:
        ui.show_table(tbl)
    with right:
        ui.show_chart(charts.sessions_bar(tbl), "dash_sessions")
    l2, r2 = st.columns(2)
    with l2:
        ui.show_chart(charts.marks_vs_target(tbl), "dash_marks")
    with r2:
        ui.show_chart(charts.whatif_curves(h.whatif_frame(r.table), sc.target, shown["allocation"]), "dash_whatif")

    st.markdown("#### Weekly study timetable")
    if not shown["schedule"]:
        ui.note("No study sessions are needed: the predicted marks already meet the target." if out.ok else
                "No sessions could be scheduled.")
    else:
        grid = h.timetable_grid(shown["schedule"], out.profile.weeks_avail, sc.spw)
        t1, t2 = st.columns([3, 2])
        with t1:
            ui.show_chart(charts.timetable_heatmap(grid), "dash_timetable")
        with t2:
            st.dataframe(h.timetable_frame(shown["schedule"], out.profile.weeks_avail, sc.spw))
            st.caption("A#·S# = assignment number · session number. Deadlines are shown in the week label.")
    with st.expander("Search and CSP counters"):
        st.json({"search": {"expanded": r.search.expanded, "generated": r.search.generated, "goal_tests": r.search.goal_tests,
                            "csp_rejected": r.search.csp_rejected}, "csp_total": r.csp_stats, "csp_calls": r.csp_calls})
