import numpy as np
from collections import namedtuple
from Interval_ad import Interval, jacobian, hessian, mid

# Algorithm 9.5: nonlinear forward reachability using conservative linearization,
# concretizing the reachable set at each step. Backend: interval_ad. Includes
# conservative_linearization (9.3). sets(sys, d) is system-specific.

Transition = namedtuple("Transition", ["s", "o", "a", "x"])


class Hyperrectangle:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)


class UnionSetArray:
    def __init__(self, sets):
        self.sets = list(sets)


def extract(env, x):
    raise NotImplementedError  # system-specific


def sets(sys, d):
    raise NotImplementedError  # system-specific


def step(sys, s, x):
    o = sys.sensor.observe(s, x.xo)
    a = sys.agent.act(o, x.xa)
    s_next = sys.env.step(s, a, x.xs)
    return o, a, s_next


def rollout(sys, s, x_traj, d=None):
    if d is None:
        d = len(x_traj)
    tau = []
    for t in range(d):
        x = x_traj[t]
        o, a, s_next = step(sys, s, x)
        tau.append(Transition(s, o, a, x))
        s = s_next
    return tau


def r(sys, x):
    s, x = extract(sys.env, x)
    tau = rollout(sys, s, x)
    return tau[-1].s


def to_hyperrectangle(I):
    return Hyperrectangle(low=[i.lo for i in I], high=[i.hi for i in I])


def to_intervals(P):
    return [Interval(lo, hi) for lo, hi in zip(P.low, P.high)]


def cartesian_product(A, B):
    return Hyperrectangle(low=np.concatenate([A.low, B.low]),
                          high=np.concatenate([A.high, B.high]))


def _dot(a, b):
    acc = Interval(0.0)
    for ai, bi in zip(a, b):
        acc = acc + ai * bi
    return acc


def conservative_linearization(sys, P):
    I = to_intervals(P)
    c = [mid(i) for i in I]
    fc = [mid(v) for v in r(sys, [Interval(ci) for ci in c])]
    Jiv = jacobian(lambda x: r(sys, x), [Interval(ci) for ci in c])
    J = np.array([[mid(Jiv[i][j]) for j in range(len(I))] for i in range(len(fc))])
    Ic = [I[k] - c[k] for k in range(len(I))]
    alpha = []
    for i in range(len(fc)):
        H = hessian(lambda x: r(sys, x)[i], I)
        alpha.append(_dot(Ic, [_dot(H[a], Ic) for a in range(len(Ic))]))
    out = []
    for i in range(len(fc)):
        Jrow = [Interval(J[i][j]) for j in range(len(I))]
        out.append(Interval(fc[i]) + _dot(Jrow, Ic) + alpha[i])
    return to_hyperrectangle(out)


class ReachabilityAlgorithm:
    pass


class ConcreteConservativeLinearization(ReachabilityAlgorithm):
    def __init__(self, h):
        self.h = h


def reachable(alg, sys):
    S, X = sets(sys, 2)
    Rs = [S]
    for d in range(2, alg.h + 1):  # Julia 2:h
        S = conservative_linearization(sys, cartesian_product(S, X))
        Rs.append(S)
    return UnionSetArray(Rs)