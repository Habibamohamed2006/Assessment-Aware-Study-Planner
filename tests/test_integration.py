from tests import *
from integration.planner import plan

def test_plan_known_answer_all_algos():
    for a in ["ucs", "greedy", "astar"]:
        r = plan(P([4, 8, 12]), FakePredictor([40, 50, 30], [5, 5, 5]), 55, 3, a)
        assert r.status == "OK" and r.allocation == [3, 1, 5] and len(r.schedule) == 9
def test_schedule_respects_deadlines():
    r = plan(P([3, 6, 9]), FakePredictor([40, 50, 30], [5, 5, 5]), 55, 3, "astar")
    for a, j, w, s in r.schedule: assert w <= [3, 6, 9][a]
def test_all_csp_configs_give_same_plan():
    from csp import CSPConfig
    res = [plan(P([4, 8, 12]), FakePredictor([40, 50, 30], [5, 5, 5]), 55, 3, "astar", csp_cfg=c).allocation
           for c in [CSPConfig(False, False), CSPConfig(True, False), CSPConfig(True, True)]]
    assert res[0] == res[1] == res[2]
def test_impossible_target_message_and_best_effort():
    r = plan(P([4, 8, 12]), FakePredictor([40, 50, 30], [1, 1, 1]), 90, 3)
    assert r.status == "NO_SOLUTION" and "NO SOLUTION" in r.message and "unreachable" in r.message and r.best_effort["marks"]
def test_nn_enough_but_csp_rejects_demo_case():
    # needs 6 sessions on assignment 1 but only 2 weeks * 2 slots before its deadline -> CSP rejects, clear message
    r = plan(P([2, 8, 12]), FakePredictor([25, 50, 50], [5, 5, 5]), 55, 2, "ucs")
    assert r.status == "NO_SOLUTION" and r.search.csp_rejected >= 1 and "CSP rejected" in r.message
