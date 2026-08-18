import numpy as np

# Algorithm 10.1: convert a discrete system to a directed weighted graph.
# WALL: Julia uses Graphs.jl (WeightedGraph); reimplemented inline here with an
# adjacency structure keyed by states. env.states() lists the states; sys.successors(s)
# is system-specific and returns (next_states, weights).


class WeightedGraph:
    def __init__(self, states):
        self.states = list(states)
        self._index = {s: i for i, s in enumerate(self.states)}
        self._succ = {s: {} for s in self.states}   # s -> {s': w} outgoing
        self._pred = {s: {} for s in self.states}    # s -> {s': w} incoming

    def add_edge(self, s, sp, w):
        self._succ[s][sp] = w
        self._pred[sp][s] = w


def to_graph(sys):
    S = sys.env.states()
    g = WeightedGraph(S)
    for s in S:
        Sp, ws = sys.successors(s)  # Julia successors(sys, s)
        for sp, w in zip(Sp, ws):
            g.add_edge(s, sp, w)
    return g