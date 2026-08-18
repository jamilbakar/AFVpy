import numpy as np
from collections import namedtuple
from Interval_ad import Interval, gradient, hessian, mid

# Algorithm 9.4: nonlinear forward reachability using Taylor inclusion, concretizing
# the reachable set at each step. Backend: interval_ad (interval-valued gradient/
# hessian). Includes taylor_inclusion (9.2). intervals/extract are system-specific.

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


def _dot(a, b):
    acc = Interval(0.0)
    for ai, bi in zip(a, b):
        acc = acc + ai * bi
    return acc


def taylor_inclusion(sys, I, order):
    c = [mid(i) for i in I]
    fc = r(sys, [Interval(ci) for ci in c])
    Ic = [I[k] - c[k] for k in range(len(I))]
    Ip = []
    for i in range(len(fc)):
        if order == 1:
            g = gradient(lambda x: r(sys, x)[i], I)
            Ip.append(fc[i] + _dot(g, Ic))
        else:
            g = gradient(lambda x: r(sys, x)[i], [Interval(ci) for ci in c])
            H = hessian(lambda x: r(sys, x)[i], I)
            quad = _dot(Ic, [_dot(H[a], Ic) for a in range(len(Ic))])
            Ip.append(fc[i] + _dot(g, Ic) + quad)
    return Ip


class ReachabilityAlgorithm:
    pass


class ConcreteTaylorInclusion(ReachabilityAlgorithm):
    def __init__(self, h, order):
        self.h = h
        self.order = order


def reachable(alg, sys):
    I = intervals(sys, 2)
    s, _ = extract(sys.env, I)
    Is = [s]
    for d in range(2, alg.h + 1):  # Julia 2:h
        Ip = taylor_inclusion(sys, I, alg.order)
        Is.append(Ip)
        # NOTE: extract the NEW state from I' for the next step (see 9.2 caption).
        s, _ = extract(sys.env, Ip)
        I = list(I)
        I[:len(s)] = s
    return UnionSetArray([to_hyperrectangle(Ip) for Ip in Is])