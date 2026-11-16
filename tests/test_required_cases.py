"""Explicitly categorized minimum cases required by the project brief."""
import numpy as np
import pytest
from tests import P, FakePredictor
from csp import StudyCSP, CSPConfig, check_solution
from search import build_problem, run_search
from integration.planner import plan


@pytest.mark.normal
def test_normal_search_known_optimal_solution():
    table = [[40 + 5*k for k in range(9)], [50 + 5*k for k in range(9)], [30 + 5*k for k in range(9)]]
    _, ok, hg, ha = build_problem(table, 55, "each")
    r = run_search("ucs", [8, 8, 8], ok, lambda s: (True, None), hg, ha)
    assert r.status == "solved" and r.state == (3, 1, 5) and r.cost == 9


@pytest.mark.normal
def test_normal_csp_returns_valid_timetable():
    csp = StudyCSP([2, 1], [3, 4], 2)
    r = csp.solve(CSPConfig(True, True))
    assert r.status == "solved" and check_solution([2, 1], [3, 4], 2, r.solution)


@pytest.mark.normal
def test_normal_plan_integrates_nn_search_and_csp():
    r = plan(P([4, 8, 12]), FakePredictor([40, 50, 30], [5, 5, 5]), 55, 3, "astar")
    assert r.status == "OK" and r.allocation == [3, 1, 5] and len(r.schedule) == 9


@pytest.mark.normal
def test_normal_zero_effort_is_valid_when_target_already_met():
    r = plan(P([4, 8, 12]), FakePredictor([60, 60, 60], [1, 1, 1]), 55)
    assert r.status == "OK" and r.cost == 0 and r.schedule == []


@pytest.mark.normal
def test_normal_search_algorithms_find_a_goal():
    table = [[40 + 5*k for k in range(9)]]
    _, ok, hg, ha = build_problem(table, 55)
    for algo in ("ucs", "greedy", "astar"):
        r = run_search(algo, [8], ok, lambda s: (True, None), hg, ha)
        assert r.status == "solved" and r.state == (3,)


@pytest.mark.edge
def test_edge_exact_capacity_is_schedulable():
    r = StudyCSP([2, 2], [2, 2], 2).solve(CSPConfig(True, True))
    assert r.status == "solved" and check_solution([2, 2], [2, 2], 2, r.solution)


@pytest.mark.edge
def test_edge_empty_session_set_is_solved():
    r = StudyCSP([0, 0, 0], [2, 3, 4], 3).solve()
    assert r.status == "solved" and r.solution == {}


@pytest.mark.edge
def test_edge_single_session_has_one_available_slot():
    r = StudyCSP([1], [1], 1).solve()
    assert r.status == "solved" and r.solution == {(0, 0): 0}


@pytest.mark.impossible
def test_impossible_deadline_capacity_returns_no_solution():
    r = StudyCSP([3], [1], 2).solve(CSPConfig(True, True))
    assert r.status == "no_solution" and "NO SOLUTION" in r.message


@pytest.mark.impossible
def test_impossible_target_returns_best_effort_and_clear_message():
    r = plan(P([1, 1, 1]), FakePredictor([20, 30, 40], [0, 0, 0]), 90, 2)
    assert r.status == "NO_SOLUTION" and "NO SOLUTION" in r.message and r.best_effort is not None
