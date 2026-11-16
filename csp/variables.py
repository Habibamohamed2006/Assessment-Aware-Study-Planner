"""CSP model: variables = study sessions, domains = weekly slots, config / stats / result containers.

Variables : (i, j) = j-th session of assignment i.
Domain    : slot indices 0 .. weeks_avail[i]*slots_per_week-1   (slot s -> week s//spw + 1, slot s%spw + 1);
            only slots BEFORE the deadline exist, which encodes the deadline rule.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from csp.constraints import consistent


@dataclass
class CSPConfig:
    use_fc: bool = True
    use_ac3: bool = True
    max_steps: int = 200_000          # cutoff on assignments so plain BT cannot run forever

    @property
    def name(self):
        return "BT" + ("+FC" if self.use_fc else "") + ("+AC3" if self.use_ac3 else "")


@dataclass
class CSPStats:
    assignments: int = 0      # assignments actually made (value consistent with earlier ones)
    checks: int = 0           # pairwise consistency checks (work done, for honesty)
    backtracks: int = 0       # assignments undone
    ac3_revisions: int = 0
    ac3_prunings: int = 0
    fc_prunings: int = 0


@dataclass
class CSPResult:
    status: str                       # "solved" | "no_solution" | "cutoff"
    solution: dict | None
    stats: CSPStats
    message: str = ""
    domain_sizes_initial: dict = field(default_factory=dict)
    domain_sizes_after_ac3: dict = field(default_factory=dict)


class StudyCSP:
    ok = staticmethod(consistent)

    def __init__(self, counts, weeks_avail, slots_per_week):
        self.counts, self.weeks, self.spw = list(counts), list(weeks_avail), slots_per_week
        self.vars = [(i, j) for i, c in enumerate(self.counts) for j in range(c)]
        self.init_domains = {v: list(range(self.weeks[v[0]] * self.spw)) for v in self.vars}

    def solve(self, cfg: CSPConfig = CSPConfig()) -> CSPResult:
        from csp.backtracking import solve      # lazy: backtracking imports this module
        return solve(self, cfg)

    def timetable(self, solution):
        """list of (assignment, session, week, slot_in_week) sorted by time."""
        rows = [(i, j + 1, s // self.spw + 1, s % self.spw + 1) for (i, j), s in solution.items()]
        return sorted(rows, key=lambda r: (r[2], r[3]))
