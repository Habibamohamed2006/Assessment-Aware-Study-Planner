"""Uniform Cost Search: priority = g (number of sessions so far). Optimal, but blind."""
import search


def ucs(k_max, nn_ok, feasible, h_greedy, h_astar):
    return search.best_first(lambda g, s, hg, ha: g, k_max, nn_ok, feasible, h_greedy, h_astar)
