import numpy as np

# Algorithm 10.5: finite-horizon probabilistic reachability for discrete systems.
# Initializes the reach probability to 1 on the target set and back-propagates it
# through the out-neighbors (eq 10.3), then weights by the initial distribution.
# Self-sufficient: includes the graph (10.1). psi.set is the target set.


class WeightedGraph:
    def __init__(self, states):
        self.states = list(states)
        self._index = {s: i for i, s in enumerate(self.states)}
        self._succ = {s: {} for s in self.states}
        self._pred = {s: {} for s in self.states}

    def add_edge(self, s, sp, w):
        self._succ[s][sp] = w
        self._pred[sp][s] = w


def outneighbors(g, s):
    return list(g._succ[s].keys())


def get_weight(g, s, sp):
    return g._succ[s][sp]


def to_graph(sys):
    S = sys.env.states()
    g = WeightedGraph(S)
    for s in S:
        Sp, ws = sys.successors(s)
        for sp, w in zip(Sp, ws):
            g.add_edge(s, sp, w)
    return g


class ReachabilityAlgorithm:
    pass


class ProbabilisticFiniteHorizon(ReachabilityAlgorithm):
    def __init__(self, h):
        self.h = h


def reachable(alg, sys, psi):
    S = sys.env.states()
    g = to_graph(sys)
    dist = sys.env.initial_distribution()
    ST = psi.set
    R = {s: (1.0 if s in ST else 0.0) for s in S}
    for d in range(2, alg.h + 1):  # Julia 2:h
        R = {s: (1.0 if s in ST else
                 sum(get_weight(g, s, sp) * R[sp] for sp in outneighbors(g, s)))
             for s in S}
    return sum(R[s] * dist.pdf(s) for s in S)