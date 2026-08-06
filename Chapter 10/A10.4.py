import numpy as np

# Algorithm 10.4: probability of occupancy for discrete systems. Starts from the
# initial distribution and pushes it forward through the graph via eq 10.2, using the
# in-neighbors' weighted contributions. Self-sufficient: includes the graph (10.1).
# dist = env.initial_distribution(); pdf(dist, s) is the state probability.


class WeightedGraph:
    def __init__(self, states):
        self.states = list(states)
        self._index = {s: i for i, s in enumerate(self.states)}
        self._succ = {s: {} for s in self.states}
        self._pred = {s: {} for s in self.states}

    def add_edge(self, s, sp, w):
        self._succ[s][sp] = w
        self._pred[sp][s] = w


def inneighbors(g, s):
    return list(g._pred[s].keys())


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


class SetCategorical:
    def __init__(self, P):
        self.P = P  # dict: state -> probability


class ReachabilityAlgorithm:
    pass


class ProbabilisticOccupancy(ReachabilityAlgorithm):
    def __init__(self, h):
        self.h = h


def reachable(alg, sys):
    S = sys.env.states()
    g = to_graph(sys)
    dist = sys.env.initial_distribution()
    P = {s: dist.pdf(s) for s in S}
    for t in range(2, alg.h + 1):  # Julia 2:h
        P = {s: sum(get_weight(g, sp, s) * P[sp] for sp in inneighbors(g, s)) for s in S}
    return SetCategorical(P)