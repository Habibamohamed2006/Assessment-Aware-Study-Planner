"""Forward checking: after x=val, remove inconsistent slots from the domains of unassigned sessions."""


def forward_check(csp, x, val, assignment, domains, stats):
    """Returns new domains, or None on a domain wipe-out (dead end detected early)."""
    nd = dict(domains)
    for y in csp.vars:
        if y in assignment:
            continue
        new = [vy for vy in domains[y] if csp.ok(x, val, y, vy)]
        stats.fc_prunings += len(domains[y]) - len(new)
        if not new:
            return None
        nd[y] = new
    return nd
