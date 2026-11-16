"""Greedy Best-First: priority = h (total marks still missing below the target). Fast, not guaranteed optimal."""
import search


def greedy(k_max, nn_ok, feasible, h_greedy, h_astar):
    return search.best_first(lambda g, s, hg, ha: hg(s), k_max, nn_ok, feasible, h_greedy, h_astar)
