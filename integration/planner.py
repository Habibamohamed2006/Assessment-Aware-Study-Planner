"""Integration: Search proposes a plan -> NN says if marks are high enough -> CSP checks it can be scheduled."""
from __future__ import annotations
import itertools
from dataclasses import dataclass, field
from config import *
from csp import StudyCSP, CSPConfig, capacity_ok
from search import build_problem, run_search, SearchStats


@dataclass
class PlanResult:
    status: str                     # "OK" | "NO_SOLUTION"
    message: str
    algo: str
    target: float
    goal: str
    allocation: list | None
    marks: list | None
    schedule: list | None           # (assignment, session, week, slot)
    search: SearchStats
    csp_stats: dict
    csp_calls: int
    trace: list
    table: list
    weeks_avail: list
    spw: int
    best_effort: dict | None = None
    cost: int | None = None


def _best_effort(table, weeks, spw, goal, k_max):
    best, best_key = None, None
    for s in itertools.product(*[range(k + 1) for k in k_max]):
        if not capacity_ok(s, weeks, spw):
            continue
        m = [table[i][s[i]] for i in range(len(s))]
        key = (sum(m) / len(m), min(m), -sum(s)) if goal == "mean" else (min(m), sum(m) / len(m), -sum(s))
        if best_key is None or key > best_key:
            best, best_key = (s, m), key
    s, m = best
    sol = StudyCSP(s, weeks, spw).solve(CSPConfig())
    return {"allocation": list(s), "marks": m, "schedule": StudyCSP(s, weeks, spw).timetable(sol.solution or {})}


def plan(profile, predictor, target, spw=SLOTS_PER_WEEK, algo="astar", goal="each",
         csp_cfg: CSPConfig | None = None, k_max=K_MAX) -> PlanResult:
    csp_cfg = csp_cfg or CSPConfig()
    table = predictor.marks_table(profile, k_max)
    n = len(table); kmax = [k_max] * n
    marks, nn_ok, h_g, h_a = build_problem(table, target, goal)
    agg = {"assignments": 0, "backtracks": 0, "checks": 0, "ac3_revisions": 0, "ac3_prunings": 0, "fc_prunings": 0}
    calls = [0]

    def feasible(state):
        calls[0] += 1
        csp = StudyCSP(state, profile.weeks_avail, spw)
        r = csp.solve(csp_cfg)
        for k in agg:
            agg[k] += getattr(r.stats, k)
        return r.status == "solved", (csp, r)

    sr = run_search(algo, kmax, nn_ok, feasible, h_g, h_a)
    base = dict(algo=algo, target=target, goal=goal, search=sr.stats, csp_stats=agg, csp_calls=calls[0],
                trace=sr.trace, table=table, weeks_avail=profile.weeks_avail, spw=spw)
    if sr.status == "solved":
        csp, r = sr.payload
        return PlanResult("OK", "Plan found.", allocation=list(sr.state), marks=marks(sr.state),
                          schedule=csp.timetable(r.solution), cost=sr.cost, **base)
    be = _best_effort(table, profile.weeks_avail, spw, goal, kmax)
    why = ("the NN predicts the target cannot be reached even with the maximum sessions"
           if sr.stats.csp_rejected == 0 else
           f"{sr.stats.csp_rejected} plan(s) met the target by NN but the CSP rejected every schedule")
    msg = (f"NO SOLUTION: target {target} is unreachable at this capacity ({spw} slots/week): {why}. "
           f"Best predicted marks possible: {[round(x, 1) for x in be['marks']]} "
           f"with sessions {be['allocation']}.")
    return PlanResult("NO_SOLUTION", msg, allocation=None, marks=None, schedule=None, best_effort=be, **base)


def describe(res: PlanResult, student_id=None) -> str:
    lines = [f"[{res.algo.upper()}] target={res.target} ({res.goal}), slots/week={res.spw}, weeks before deadlines={res.weeks_avail}"]
    if res.status == "OK":
        lines.append(f"Allocation (sessions per assignment): {res.allocation}  total={res.cost}")
        lines.append(f"Predicted marks: {[round(m, 1) for m in res.marks]}")
        lines.append("Timetable (week, slot -> assignment/session):")
        for a, j, w, s in res.schedule:
            lines.append(f"   week {w:>2} slot {s}: assignment {a + 1}, session {j}")
    else:
        lines.append(res.message)
    st = res.search
    lines.append(f"Search: expanded={st.expanded} generated={st.generated} csp_rejected={st.csp_rejected} | "
                 f"CSP: calls={res.csp_calls} {res.csp_stats}")
    return "\n".join(lines)
