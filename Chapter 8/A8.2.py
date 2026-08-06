import numpy as np

# Algorithm 8.2: one-step reachable set for a linear system (eq 8.12).
# WALL: Julia uses LazySets.jl for the set operations. There is no drop-in Python
# equivalent, so the linear map (M * S) and Minkowski sum (A ⊕ B) are expressed via
# the helpers below, which assume a set library whose sets support `@` (left linear
# map) and `+` (Minkowski sum). X is a Disturbance-set with .xo / .xa / .xs.
# Component accessors: env.Ts(), env.Ta(), agent.Pi_o(), sensor.Os().


def linear_map(M, S):      # LazySets: M * S
    return M @ S


def minkowski_sum(A, B):   # LazySets: A ⊕ B
    return A + B


def get_matrices(sys):
    return sys.env.Ts(), sys.env.Ta(), sys.agent.Pi_o(), sys.sensor.Os()


def linear_set_propagation(sys, S, X):
    Ts, Ta, Pio, Os = get_matrices(sys)
    term1 = linear_map(Ts + Ta @ Pio @ Os, S)
    term2 = linear_map(Ta @ Pio, X.xo)
    term3 = linear_map(Ta, X.xa)
    return minkowski_sum(minkowski_sum(minkowski_sum(term1, term2), term3), X.xs)