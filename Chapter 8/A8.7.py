import numpy as np
import cvxpy as cp

# Algorithm 8.7: linear forward reachability using linear programming. For each depth
# it builds the constrained model (8.6) and evaluates the support function in every
# direction, then unions the resulting polytope into the reachable set.
# Self-sufficient: includes constrained_model / rho (8.6). WALLs: JuMP.jl -> cvxpy;
# LazySets.jl (HPolytope, HalfSpace, union, Ab) assume a set library.


def dim(S):
    return S.dim


def Ab(P):
    return P.A, P.b


def union(A, B):
    return A | B


def HalfSpace(direction, value):     # LazySets: HalfSpace(d, ρ)
    raise NotImplementedError


def HPolytope(halfspaces):           # LazySets: HPolytope([...])
    raise NotImplementedError


def get_matrices(sys):
    return sys.env.Ts(), sys.env.Ta(), sys.agent.Pi_o(), sys.sensor.Os()


class _Model:
    def __init__(self):
        self.constraints = []
        self.obj_dict = {}
        self.value = None


def constrained_model(sys, d, S, X):
    model = _Model()
    s = cp.Variable((dim(S), d)); model.obj_dict['s'] = s
    xo = cp.Variable((dim(X.xo), d)); model.obj_dict['xo'] = xo
    xs = cp.Variable((dim(X.xs), d)); model.obj_dict['xs'] = xs
    xa = cp.Variable((dim(X.xa), d)); model.obj_dict['xa'] = xa
    As, bs = Ab(S)
    (Axo, bxo), (Axs, bxs), (Axa, bxa) = Ab(X.xo), Ab(X.xs), Ab(X.xa)
    model.constraints.append(As @ s[:, 0] <= bs)
    for i in range(d):
        model.constraints.append(Axo @ xo[:, i] <= bxo)
        model.constraints.append(Axs @ xs[:, i] <= bxs)
        model.constraints.append(Axa @ xa[:, i] <= bxa)
    Ts, Ta, Pio, Os = get_matrices(sys)
    for i in range(d - 1):
        model.constraints.append(
            (Ts + Ta @ Pio @ Os) @ s[:, i] + Ta @ Pio @ xo[:, i]
            + Ta @ xa[:, i] + xs[:, i] == s[:, i + 1])
    return model


def rho(model, direction, depth):
    s = model.obj_dict['s']
    prob = cp.Problem(cp.Maximize(direction @ s[:, depth - 1]), model.constraints)
    model.value = prob.solve()
    return model.value


class ReachabilityAlgorithm:
    pass


class LinearProgramming(ReachabilityAlgorithm):
    def __init__(self, h, D, tol):
        self.h = h      # time horizon
        self.D = D      # directions to evaluate the support function
        self.tol = tol  # tolerance for checking satisfaction (used by 8.8)


def reachable(alg, sys):
    h, D = alg.h, alg.D
    S = sys.env.S1()
    X = sys.disturbance_set()
    R = S
    for d in range(2, h + 1):  # Julia 2:h
        model = constrained_model(sys, d, S, X)
        rhos = [rho(model, direction, d) for direction in D]
        R = union(R, HPolytope([HalfSpace(direction, r) for direction, r in zip(D, rhos)]))
    return R