from tests import *
from search import build_problem, run_search

TABLE = [[40 + 5 * k for k in range(9)], [50 + 5 * k for k in range(9)], [30 + 5 * k for k in range(9)]]  # target 55 -> needs (3,1,5)

def run(algo, table=TABLE, target=55, feasible=lambda s: (True, None), goal="each"):
    _, ok, hg, ha = build_problem(table, target, goal)
    return run_search(algo, [len(r) - 1 for r in table], ok, feasible, hg, ha)

def test_known_answer_ucs():
    r = run("ucs"); assert r.status == "solved" and r.state == (3, 1, 5) and r.cost == 9
def test_known_answer_astar():
    r = run("astar"); assert r.state == (3, 1, 5) and r.cost == 9
def test_known_answer_greedy():
    r = run("greedy"); assert r.state == (3, 1, 5) and r.cost == 9
def test_astar_expands_fewer_than_ucs():
    assert run("astar").stats.expanded < run("ucs").stats.expanded
def test_greedy_expands_fewer_than_ucs_and_counters_positive():
    g, u = run("greedy"), run("ucs"); assert g.stats.expanded < u.stats.expanded and g.stats.generated >= g.stats.expanded
def test_mean_goal_known_answer():
    r = run("astar", goal="mean")      # need 165 total marks, have 120, +5 per session -> 9 sessions
    assert r.status == "solved" and r.cost == 9
# ---- 3 edge
def test_impossible_nn_cannot_reach():
    for a in ["ucs", "greedy", "astar"]:
        assert run(a, target=99).status == "no_solution"
def test_impossible_csp_rejects_everything():
    r = run("ucs", feasible=lambda s: (False, None)); assert r.status == "no_solution" and r.stats.csp_rejected >= 1
def test_csp_rejection_then_search_continues_in_mean_goal():
    # reject plans putting >4 sessions on assignment 0; mean goal still has other solutions
    r = run("ucs", goal="mean", feasible=lambda s: (s[0] <= 4, None), target=60)
    assert r.status == "solved" and r.state[0] <= 4
