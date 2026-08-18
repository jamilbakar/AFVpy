import numpy as np
from collections import namedtuple
from Interval_ad import Interval, gradient, hessian, mid

# Algorithm 9.2: nonlinear forward reachability using first/second-order Taylor
# inclusion (eq 9.17 / 9.19). Backend: interval_ad provides gradient and hessian
# evaluated OVER INTERVALS (replaces ForwardDiff.jl composed with IntervalArithmetic.jl).
# The system r must be written with interval_ad ops. intervals/extract are
# system-specific.

Transition = namedtuple("Transition", ["s", "o", "a", "x"])


class Hyperrectangle:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)


class UnionSetArray:
    def __init__(self, sets):
        self.sets = list(sets)


def intervals(sys, d):
    raise NotImplementedError  # system-specific


def extract(env, x):
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


def _dot(a, b):  # interval dot product of two interval vectors
    acc = Interval(0.0)
    for ai, bi in zip(a, b):
        acc = acc + ai * bi
    return acc


def taylor_inclusion(sys, I, order):
    c = [mid(i) for i in I]                    # mid.(I)
    fc = r(sys, [Interval(ci) for ci in c])    # r at the (degenerate-interval) midpoint
    Ic = [I[k] - c[k] for k in range(len(I))]  # I - c
    Ip = []
    for i in range(len(fc)):
        if order == 1:
            g = gradient(lambda x: r(sys, x)[i], I)            # gradient over I
            Ip.append(fc[i] + _dot(g, Ic))
        else:
            g = gradient(lambda x: r(sys, x)[i], [Interval(ci) for ci in c])  # at midpoint
            H = hessian(lambda x: r(sys, x)[i], I)             # hessian over I
            quad = _dot(Ic, [_dot(H[a], Ic) for a in range(len(Ic))])
            Ip.append(fc[i] + _dot(g, Ic) + quad)
    return Ip


class ReachabilityAlgorithm:
    pass


class TaylorInclusion(ReachabilityAlgorithm):
    def __init__(self, h, order):
        self.h = h
        self.order = order  # 1 or 2


def reachable(alg, sys):
    Is = []
    for d in range(1, alg.h + 1):
        I = intervals(sys, d)
        Is.append(taylor_inclusion(sys, I, alg.order))
    return UnionSetArray([to_hyperrectangle(Ip) for Ip in Is])