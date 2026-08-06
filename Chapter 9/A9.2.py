import numpy as np
from collections import namedtuple

# Algorithm 9.2: nonlinear forward reachability using first- or second-order Taylor
# inclusion functions (eq 9.17 / 9.19). WALLs: ForwardDiff.jl gradient/hessian
# evaluated OVER INTERVALS has no drop-in Python equivalent (autodiff + interval
# arithmetic); intervals/extract are system-specific; LazySets Hyperrectangle/
# UnionSetArray are minimal here.

Transition = namedtuple("Transition", ["s", "o", "a", "x"])
Interval = namedtuple("Interval", ["lo", "hi"])


class Hyperrectangle:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)


class UnionSetArray:
    def __init__(self, sets):
        self.sets = list(sets)


def extract(env, x):
    raise NotImplementedError  # system-specific


def intervals(sys, d):
    raise NotImplementedError  # system-specific


def gradient(f, x):
    raise NotImplementedError  # ForwardDiff.gradient, evaluated over intervals


def hessian(f, x):
    raise NotImplementedError  # ForwardDiff.hessian, evaluated over intervals


def mid(i):
    return (i.lo + i.hi) / 2


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


def taylor_inclusion(sys, I, order):
    c = [mid(i) for i in I]           # mid.(I)
    fc = r(sys, c)
    Ic = np.array(I) - np.array(c)    # I - c
    if order == 1:
        Ip = [fc[i] + gradient(lambda x: r(sys, x)[i], I) @ Ic
              for i in range(len(fc))]
    else:
        Ip = [fc[i] + gradient(lambda x: r(sys, x)[i], c) @ Ic
              + Ic @ hessian(lambda x: r(sys, x)[i], I) @ Ic
              for i in range(len(fc))]
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
        Ip = taylor_inclusion(sys, I, alg.order)
        Is.append(Ip)
    return UnionSetArray([to_hyperrectangle(Ip) for Ip in Is])