import numpy as np

# Algorithm 10.3: backward reachability for discrete systems. At each depth it grows
# the set of states that can reach the target using in-neighbors, stopping on
# convergence. Self-sufficient: includes the graph (10.1). psi.set is the target set.


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


class DiscreteBackward(ReachabilityAlgorithm):
    def __init__(self, h):
        self.h = h


def backward_reachable(alg, sys, psi):
    g = to_graph(sys)
    S = set(psi.set)
    B = set(S)
    for d in range(2, alg.h + 1):  # Julia 2:h
        S = set(sp for s in S for sp in inneighbors(g, s))
        if B == (B | S):
            break
        B = B | S
    return B