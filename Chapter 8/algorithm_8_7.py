import numpy as np
import cvxpy as cp
from pycvxset import Polytope

# Algorithm 8.7 (pycvxset backend): linear forward reachability via linear programming.
# LazySets -> pycvxset mapping:
#   Ab(P)                 -> (P.A, P.b)              (H-rep A x <= b)
#   HalfSpace(dir, val)   -> (dir, val) pair         (dir . x <= val)
#   HPolytope(halfspaces) -> Polytope(A=..., b=...)
#   union(A, B)           -> UnionSetArray (a union of convex sets is not convex, so
#                            the pieces are kept, as LazySets' UnionSetArray does)
# The optimization model uses cvxpy. Install: pip install cvxpy pycvxset.


class UnionSetArray:
    def __init__(self, sets):
        self.sets = list(sets)


def union(A, B):
    if isinstance(A, UnionSetArray):
        return UnionSetArray(A.sets + [B])
    return UnionSetArray([A, B])


def Ab(P):
    return P.A, P.b


def HalfSpace(direction, value):
    return (np.asarray(direction, dtype=float), float(value))  # direction . x <= value


def HPolytope(halfspaces):
    A = np.array([h[0] for h in halfspaces], dtype=float)
    b = np.array([h[1] for h in halfspaces], dtype=float)
    return Polytope(A=A, b=b)


def dim(S):
    return S.dim


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
        self.h = h
        self.D = D
        self.tol = tol


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