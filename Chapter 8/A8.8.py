import numpy as np
import cvxpy as cp

# Algorithm 8.8: check whether a system could reach a convex avoid set via convex
# programming. For each depth it builds the constrained model (8.6), adds a variable
# for the avoid set, and minimizes the squared distance between the reachable set and
# the avoid set; a zero distance (within tolerance) at any depth means the spec is not
# satisfied. Self-sufficient: includes constrained_model (8.6). WALLs: JuMP.jl -> cvxpy;
# LazySets.jl (Ab, avoid set) assume a set library.


def dim(S):
    return S.dim


def Ab(P):
    return P.A, P.b


def negate(psi):     # ¬(ψ::AvoidSetSpecification) = ψ.set
    return psi.set


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


class ReachabilityAlgorithm:
    pass


class LinearProgramming(ReachabilityAlgorithm):
    def __init__(self, h, D, tol):
        self.h = h
        self.D = D
        self.tol = tol


def satisfies(alg, sys, psi):
    S = sys.env.S1()
    X = sys.disturbance_set()
    for d in range(1, alg.h + 1):  # Julia 1:h
        model = constrained_model(sys, d, S, X)
        u = cp.Variable(dim(S))
        Au, bu = Ab(negate(psi))
        constraints = model.constraints + [Au @ u <= bu]
        s = model.obj_dict['s']
        objective = cp.Minimize(cp.sum([(s[i, d - 1] - u[i]) ** 2 for i in range(dim(S))]))
        value = cp.Problem(objective, constraints).solve()
        if np.isclose(value, 0.0, atol=alg.tol):
            return False
    return True
