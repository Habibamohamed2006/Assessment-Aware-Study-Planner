"""Edge cases for search, CSP and integration (collected here)."""
from tests import *
import numpy as np
from search import build_problem, run_search
from csp import StudyCSP, CSPConfig
from integration.planner import plan

TABLE = [[40 + 5 * k for k in range(9)], [50 + 5 * k for k in range(9)], [30 + 5 * k for k in range(9)]]

def run(algo, table=TABLE, target=55, feasible=lambda s: (True, None), goal="each"):
    _, ok, hg, ha = build_problem(table, target, goal)
    return run_search(algo, [len(r) - 1 for r in table], ok, feasible, hg, ha)

def test_edge_target_already_met():
    r = run("ucs", target=30); assert r.state == (0, 0, 0) and r.cost == 0 and r.stats.expanded == 1
def test_edge_target_exactly_at_last_level():
    r = run("astar", target=40 + 5 * 8, table=[TABLE[0]]); assert r.state == (8,)
def test_edge_single_assignment():
    r = run("greedy", table=[TABLE[0]], target=55); assert r.state == (3,)
# ---- impossible
def test_edge_zero_sessions():
    r = StudyCSP([0, 0, 0], [2, 3, 4], 3).solve(CSPConfig()); assert r.status == "solved" and r.solution == {}
def test_edge_exactly_full_capacity():
    r = StudyCSP([2, 2], [2, 2], 2).solve(CSPConfig()); assert r.status == "solved" and sorted(r.solution.values()) == [0, 1, 2, 3]
def test_edge_single_slot_single_session():
    r = StudyCSP([1], [1], 1).solve(CSPConfig()); assert r.solution == {(0, 0): 0}
# ---- impossible
def test_edge_target_already_met_gives_empty_plan():
    r = plan(P([4, 8, 12]), FakePredictor([60, 60, 60], [1, 1, 1]), 55); assert r.allocation == [0, 0, 0] and r.schedule == []
def test_edge_exactly_enough_slots():
    r = plan(P([2, 2, 2]), FakePredictor([50, 50, 50], [5, 5, 5]), 55, 3); assert r.status == "OK" and r.allocation == [1, 1, 1]
# ---- impossible
