from tests import *
import numpy as np
from csp import StudyCSP, CSPConfig, capacity_ok, check_solution, ac3

CFGS = [CSPConfig(False, False), CSPConfig(True, False), CSPConfig(False, True), CSPConfig(True, True)]

def test_all_configs_solve_and_validate():
    for c in CFGS:
        r = StudyCSP([3, 2, 2], [2, 4, 5], 2).solve(c)
        assert r.status == "solved" and check_solution([3, 2, 2], [2, 4, 5], 2, r.solution)
def test_ordering_within_assignment():
    r = StudyCSP([4], [3], 2).solve(CSPConfig()); s = [r.solution[(0, j)] for j in range(4)]; assert s == sorted(s)
def test_deadline_respected():
    r = StudyCSP([2, 3], [1, 3], 3).solve(CSPConfig())
    assert all(v < 3 for (i, j), v in r.solution.items() if i == 0)
def test_forward_checking_does_less_work():
    n, w = [4, 4, 4], [4, 5, 6]
    a, b = StudyCSP(n, w, 3).solve(CSPConfig(False, False)).stats, StudyCSP(n, w, 3).solve(CSPConfig(True, False)).stats
    assert b.checks < a.checks and b.assignments <= a.assignments
def test_ac3_prunes_domains():
    r = StudyCSP([4, 2], [2, 6], 2).solve(CSPConfig(False, True))
    assert sum(r.domain_sizes_after_ac3.values()) < sum(r.domain_sizes_initial.values()) and r.stats.ac3_prunings > 0
def test_matches_exact_capacity_test_on_random_cases():
    rng = np.random.default_rng(0)
    for _ in range(40):
        n = rng.integers(1, 5, 3).tolist(); w = rng.integers(1, 4, 3).tolist(); spw = int(rng.integers(1, 3))
        for c in [CSPConfig(True, True), CSPConfig(False, False)]:
            r = StudyCSP(n, w, spw).solve(c); assert (r.status == "solved") == capacity_ok(n, w, spw)
# ---- 3 edge
def test_impossible_too_many_sessions_clear_message():
    for c in CFGS:
        r = StudyCSP([3], [1], 2).solve(c); assert r.status == "no_solution" and "NO SOLUTION" in r.message
def test_impossible_deadlines_overlap():
    r = StudyCSP([2, 2], [1, 1], 3).solve(CSPConfig(True, True)); assert r.status == "no_solution"
