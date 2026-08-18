import numpy as np

# Algorithm 8.4: check whether a system satisfies an avoid-set spec via set
# propagation: compute the reachable set and test whether it intersects the avoid set.
# Self-sufficient: includes the set-propagation reachability (8.3). WALL: LazySets.jl
# has no drop-in Python equivalent; intersection (∩) and is_empty assume a set library.


def linear_map(M, S):
    return M @ S


def minkowski_sum(A, B):
    return A + B


class UnionSetArray:       # pycvxset has no union type; keep the convex pieces
    def __init__(self, sets):
        self.sets = list(sets)


def union(A, B):
    if isinstance(A, UnionSetArray):
        return UnionSetArray(A.sets + [B])
    return UnionSetArray([A, B])


def intersection(A, B):    # LazySets A ∩ B -> pycvxset .intersection
    return A.intersection(B)


def is_empty(S):           # LazySets isempty(S) -> pycvxset .is_empty (property)
    return S.is_empty


def negate(psi):           # ¬(ψ::AvoidSetSpecification) = ψ.set
    return psi.set


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
        self.h = h


def reachable(alg, sys):
    h = alg.h
    S = sys.env.S1()
    X = sys.disturbance_set()
    R = S
    for t in range(h):
        S = linear_set_propagation(sys, S, X)
        R = union(R, S)
    return R


def satisfies(alg, sys, psi):
    R = reachable(alg, sys)
    return not is_empty(intersection(R, negate(psi)))