"""Backtracking search with switchable Forward Checking and AC-3 pre-processing."""
from csp.variables import CSPStats, CSPResult, CSPConfig
from csp.ac3 import ac3
from csp.forward_checking import forward_check


class _Cutoff(Exception):
    pass


def solve(csp, cfg: CSPConfig) -> CSPResult:
    stats = CSPStats()
    domains = {v: list(d) for v, d in csp.init_domains.items()}
    res = CSPResult("no_solution", None, stats, domain_sizes_initial={v: len(d) for v, d in domains.items()})
    if not csp.vars:
        res.status, res.solution, res.message = "solved", {}, "no sessions to schedule"
        return res
    if cfg.use_ac3:
        ok = ac3(csp, domains, stats)
        res.domain_sizes_after_ac3 = {v: len(d) for v, d in domains.items()}
        if not ok:
            res.message = "NO SOLUTION: AC-3 emptied a domain (sessions cannot fit before their deadlines)."
            return res
    assignment: dict = {}

    def bt(domains):
        if len(assignment) == len(csp.vars):
            return True
        x = next(v for v in csp.vars if v not in assignment)       # static variable order
        for val in domains[x]:
            consistent = True
            for y, vy in assignment.items():
                stats.checks += 1
                if not csp.ok(x, val, y, vy):
                    consistent = False
                    break
            if not consistent:
                continue
            stats.assignments += 1
            if stats.assignments > cfg.max_steps:
                raise _Cutoff()
            assignment[x] = val
            nd = forward_check(csp, x, val, assignment, domains, stats) if cfg.use_fc else domains
            if nd is not None and bt(nd):
                return True
            del assignment[x]
            stats.backtracks += 1
        return False

    try:
        found = bt(domains)
    except _Cutoff:
        res.status, res.message = "cutoff", f"stopped after {cfg.max_steps} assignments"
        return res
    if found:
        res.status, res.solution = "solved", dict(assignment)
    else:
        res.message = "NO SOLUTION: no schedule satisfies all deadline / slot / ordering rules."
    return res
