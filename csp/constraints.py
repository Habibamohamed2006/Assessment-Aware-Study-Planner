"""Binary constraints + independent validators (used by the solvers and by the tests)."""


def consistent(x, vx, y, vy) -> bool:
    """x=(i,j), y=(i',j') are session variables, vx/vy slot indices.
    (1) one session per slot; (2) sessions of the same assignment are in order (j < j' => slot < slot')."""
    if vx == vy:
        return False
    if x[0] == y[0]:
        return vx < vy if x[1] < y[1] else vx > vy
    return True


def capacity_ok(counts, weeks_avail, spw) -> bool:
    """Hall-type test (exact for this model): by every deadline, sessions due <= slots available."""
    total = 0
    for w, c in sorted(zip(weeks_avail, counts)):
        total += c
        if total > w * spw:
            return False
    return True


def check_solution(counts, weeks_avail, spw, solution) -> bool:
    """Independent validator: all sessions placed, distinct slots, before deadline, in order."""
    slots = list(solution.values())
    if len(set(slots)) != len(slots) or len(solution) != sum(counts):
        return False
    for (i, j), s in solution.items():
        if not (0 <= s < weeks_avail[i] * spw):
            return False
        if j > 0 and solution[(i, j - 1)] >= s:
            return False
    return True
