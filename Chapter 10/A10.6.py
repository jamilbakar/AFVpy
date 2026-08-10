import numpy as np

# Algorithm 10.6: infinite-horizon probabilistic reachability for discrete systems.
# Builds the transition matrix, makes target states absorbing (zero their rows), and
# solves (I - TR) R_inf = R1 (eq 10.11), then weights by the initial distribution.
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


def index(g, s):
    return g._index[s]


def state(g, i):
    return g.states[i]


def to_matrix(g):
    n = len(g.states)
    TR = np.zeros((n, n))
    for s in g.states:
        for sp, w in g._succ[s].items():
            TR[g._index[s], g._index[sp]] = w
    return TR


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


class ProbabilisticInfiniteHorizon(ReachabilityAlgorithm):
    pass


def reachable(alg, sys, psi):
    S = sys.env.states()
    g = to_graph(sys)
    dist = sys.env.initial_distribution()
    STi = [index(g, s) for s in psi.set]
    R1 = np.array([1.0 if i in STi else 0.0 for i in range(len(S))])
    TR = to_matrix(g)
    TR[STi, :] = 0
    Rinf = np.linalg.solve(np.eye(len(S)) - TR, R1)
    return sum(Rinf[i] * dist.pdf(state(g, i)) for i in range(len(S)))