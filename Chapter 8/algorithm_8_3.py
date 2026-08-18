import numpy as np

# Algorithm 8.3: linear forward reachability via set propagation. Self-sufficient:
# includes the one-step propagation (8.2). WALL: LazySets.jl has no drop-in Python
# equivalent; union (∪) and the set ops assume a set library. S_1 (initial state set)
# and disturbance_set are system-specific: env.S1() and sys.disturbance_set().


def linear_map(M, S):
    return M @ S


def minkowski_sum(A, B):
    return A + B


class UnionSetArray:       # pycvxset has no union type; keep the convex pieces
    def __init__(self, sets):
        self.sets = list(sets)


def union(A, B):           # LazySets: A ∪ B
    if isinstance(A, UnionSetArray):
        return UnionSetArray(A.sets + [B])
    return UnionSetArray([A, B])


def get_matrices(sys):
    return sys.env.Ts(), sys.env.Ta(), sys.agent.Pi_o(), sys.sensor.Os()


def linear_set_propagation(sys, S, X):
    Ts, Ta, Pio, Os = get_matrices(sys)
    return minkowski_sum(minkowski_sum(minkowski_sum(
        linear_map(Ts + Ta @ Pio @ Os, S), linear_map(Ta @ Pio, X.xo)),
        linear_map(Ta, X.xa)), X.xs)


class ReachabilityAlgorithm:
    pass


class SetPropagation(ReachabilityAlgorithm):
    def __init__(self, h):
        self.h = h  # time horizon


def reachable(alg, sys):
    h = alg.h
    S = sys.env.S1()
    X = sys.disturbance_set()
    R = S
    for t in range(h):  # Julia 1:h
        S = linear_set_propagation(sys, S, X)
        R = union(R, S)
    return R