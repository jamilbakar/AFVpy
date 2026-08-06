import numpy as np

# Algorithm 8.5: overapproximate linear forward reachability. Like 8.3, but every
# `freq` steps it replaces the current set with an epsilon-close overapproximation.
# Self-sufficient: includes the one-step propagation (8.2). WALL: LazySets.jl has no
# drop-in Python equivalent; union and overapproximate assume a set library.


def linear_map(M, S):
    return M @ S


def minkowski_sum(A, B):
    return A + B


def union(A, B):
    return A | B


def overapproximate(S, eps):  # LazySets: overapproximate(S, eps)
    return S.overapproximate(eps)


def get_matrices(sys):
    return sys.env.Ts(), sys.env.Ta(), sys.agent.Pi_o(), sys.sensor.Os()


def linear_set_propagation(sys, S, X):
    Ts, Ta, Pio, Os = get_matrices(sys)
    return minkowski_sum(minkowski_sum(minkowski_sum(
        linear_map(Ts + Ta @ Pio @ Os, S), linear_map(Ta @ Pio, X.xo)),
        linear_map(Ta, X.xa)), X.xs)


class ReachabilityAlgorithm:
    pass


class OverapproximateSetPropagation(ReachabilityAlgorithm):
    def __init__(self, h, freq, eps):
        self.h = h          # time horizon
        self.freq = freq    # overapproximation frequency
        self.eps = eps      # overapproximation tolerance


def reachable(alg, sys):
    h, freq, eps = alg.h, alg.freq, alg.eps
    S = sys.env.S1()
    X = sys.disturbance_set()
    R = S
    for t in range(1, h + 1):  # Julia 1:h (t used for the freq test)
        S = linear_set_propagation(sys, S, X)
        R = union(R, S)
        S = overapproximate(S, eps) if t % freq == 0 else S
    return R