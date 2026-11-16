"""CSP package: Backtracking, Forward Checking and AC-3 as separate, switchable parts."""
from csp.constraints import consistent, capacity_ok, check_solution
from csp.variables import CSPConfig, CSPStats, CSPResult, StudyCSP
from csp.ac3 import ac3
from csp.forward_checking import forward_check
from csp.backtracking import solve
