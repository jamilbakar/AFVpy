import numpy as np
import cvxpy as cp

# Algorithm 8.6: support function of a reachable set at depth d.
# WALL 1 (JuMP.jl): the JuMP model maps to cvxpy. `Model(SCS.Optimizer)` becomes a
# small container of variables + constraints; @variable -> cp.Variable, @constraint ->
# append to a list, @objective + optimize! -> cp.Problem(...).solve(),
# objective_value -> problem value, model.obj_dict[:s] -> model.obj_dict['s'].
# WALL 2 (LazySets.jl): Ab(P) = tosimplehrep(constraints_list(P)) returns the
# H-representation (A, b) with A x <= b; sets are assumed to expose .A / .b and .dim.
# NOTE: ρ takes (model, direction, depth); the source reuses the name `d` for both the
# direction vector and the depth (bold vs plain in the book).


def dim(S):
    return S.dim


def Ab(P):
    return P.A, P.b


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
