"""Search package: UCS, Greedy Best-First, A* over 'sessions per assignment' states.

State  = tuple of session counts, one per assignment (start = all zeros).
Action = add ONE session to ONE assignment (cost 1)  ->  path cost = total number of sessions.
Goal   = predicted marks reach the target (NN) AND the CSP can schedule the sessions (integration).
The three algorithms differ only in the priority (ucs.py, greedy.py, astar.py); the shared loop lives here.
"""
from __future__ import annotations
import heapq, itertools, math
from dataclasses import dataclass, field


@dataclass
class SearchStats:
    expanded: int = 0
    generated: int = 0
    goal_tests: int = 0
    csp_rejected: int = 0


@dataclass
class SearchResult:
    status: str                  # "solved" | "no_solution"
    state: tuple | None
    cost: int | None
    stats: SearchStats
    payload: object = None       # what the feasibility check returned for the accepted state (the schedule)
    trace: list = field(default_factory=list)


def build_problem(table, target, goal="each"):
    """Returns (marks_fn, nn_ok_fn, h_greedy, h_astar).  table[i][k] = predicted mark.
    goal 'each': every assignment >= target.  goal 'mean': average mark >= target.
    h_astar is admissible and consistent when the table is monotone (each session raises one mark only)."""
    n = len(table)
    marks = lambda s: [table[i][s[i]] for i in range(n)]
    if goal == "each":
        needs = [next((k for k, v in enumerate(row) if v >= target), math.inf) for row in table]
        ok = lambda s: all(m >= target for m in marks(s))
        h_a = lambda s: sum(max(0, needs[i] - s[i]) for i in range(n))
        h_g = lambda s: sum(max(0.0, target - m) for m in marks(s))
    else:
        gmax = max((row[k + 1] - row[k] for row in table for k in range(len(row) - 1)), default=0.0)
        ok = lambda s: sum(marks(s)) / n >= target - 1e-9
        def h_a(s):
            short = n * target - sum(marks(s))
            if short <= 1e-9:
                return 0
            return math.inf if gmax <= 1e-9 else math.ceil(short / gmax - 1e-9)
        h_g = lambda s: max(0.0, target - sum(marks(s)) / n)
    return marks, ok, h_g, h_a


def best_first(priority, k_max, nn_ok, feasible, h_greedy, h_astar, prune_inf=False, trace_limit=3000) -> SearchResult:
    """Generic best-first graph search. priority(g, state, h_greedy, h_astar) -> number (lower = first).
    feasible(state) -> (bool, payload) is the CSP check, called only on states the NN accepts."""
    n = len(k_max)
    start = tuple([0] * n)
    stats, trace = SearchStats(), []
    if prune_inf and math.isinf(h_astar(start)):
        return SearchResult("no_solution", None, None, stats, trace=trace)   # target unreachable for some assignment
    cnt = itertools.count()
    frontier = [(priority(0, start, h_greedy, h_astar), 0, next(cnt), start)]
    seen = {start}
    stats.generated = 1
    while frontier:
        f, g, _, s = heapq.heappop(frontier)
        stats.expanded += 1
        if len(trace) < trace_limit:
            trace.append({"order": stats.expanded, "state": s, "g": g, "f": f, "rejected": False})
        if nn_ok(s):
            stats.goal_tests += 1
            ok, payload = feasible(s)
            if ok:
                return SearchResult("solved", s, g, stats, payload, trace)
            stats.csp_rejected += 1          # NN says "enough" but the CSP cannot schedule it: discard, continue
            if trace and trace[-1]["state"] == s:
                trace[-1]["rejected"] = True
            continue                          # supersets of an unschedulable plan are unschedulable too
        for i in range(n):
            if s[i] < k_max[i]:
                t = s[:i] + (s[i] + 1,) + s[i + 1:]
                if t in seen or (prune_inf and math.isinf(h_astar(t))):
                    continue
                seen.add(t)
                stats.generated += 1
                heapq.heappush(frontier, (priority(g + 1, t, h_greedy, h_astar), g + 1, next(cnt), t))
    return SearchResult("no_solution", None, None, stats, trace=trace)


from search.ucs import ucs                 # noqa: E402  (after the shared code they build on)
from search.greedy import greedy           # noqa: E402
from search.astar import astar             # noqa: E402

ALGOS = {"ucs": ucs, "greedy": greedy, "astar": astar}


def run_search(algo, k_max, nn_ok, feasible, h_greedy, h_astar) -> SearchResult:
    return ALGOS[algo](k_max, nn_ok, feasible, h_greedy, h_astar)
