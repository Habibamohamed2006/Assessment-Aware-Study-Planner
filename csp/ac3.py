"""AC-3 arc consistency: trims slots that can never be part of a solution (run before search)."""
from collections import deque


def ac3(csp, domains, stats) -> bool:
    """Prunes `domains` in place. Returns False if some domain becomes empty (=> no solution)."""
    q = deque((x, y) for x in csp.vars for y in csp.vars if x != y)
    while q:
        x, y = q.popleft()
        stats.ac3_revisions += 1
        removed = [vx for vx in domains[x] if not any(csp.ok(x, vx, y, vy) for vy in domains[y])]
        if removed:
            domains[x] = [v for v in domains[x] if v not in removed]
            stats.ac3_prunings += len(removed)
            if not domains[x]:
                return False
            q.extend((z, x) for z in csp.vars if z != x and z != y)
    return True
