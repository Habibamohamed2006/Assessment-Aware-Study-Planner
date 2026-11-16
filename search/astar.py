"""A*: priority = g + h, h = sessions still needed per assignment (admissible and consistent for monotone predictions).
States with h = infinity (target unreachable for an assignment) are pruned."""
import search


def astar(k_max, nn_ok, feasible, h_greedy, h_astar):
    return search.best_first(lambda g, s, hg, ha: g + ha(s), k_max, nn_ok, feasible, h_greedy, h_astar, prune_inf=True)
