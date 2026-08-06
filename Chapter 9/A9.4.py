import numpy as np
from collections import namedtuple

# Algorithm 9.4: nonlinear forward reachability using Taylor inclusion, concretizing
# the reachable set at each step. Self-sufficient: includes taylor_inclusion (9.2).
# WALLs: ForwardDiff gradient/hessian over intervals, IntervalArithmetic, and LazySets
# have no drop-in Python equivalents; intervals/extract are system-specific.

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
    raise NotImplementedError  # ForwardDiff.gradient over intervals


def hessian(f, x):
    raise NotImplementedError  # ForwardDiff.hessian over intervals


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
    c = [mid(i) for i in I]
    fc = r(sys, c)
    Ic = np.array(I) - np.array(c)
    if order == 1:
        Ip = [fc[i] + gradient(lambda x: r(sys, x)[i], I) @ Ic for i in range(len(fc))]
    else:
        Ip = [fc[i] + gradient(lambda x: r(sys, x)[i], c) @ Ic
              + Ic @ hessian(lambda x: r(sys, x)[i], I) @ Ic for i in range(len(fc))]
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
        # NOTE: the source prints extract(sys.env, I); the intent (see caption) is to
        # extract the NEW state from I' for the next step, so Ip is used here.
        s, _ = extract(sys.env, Ip)
        I = list(I)
        I[:len(s)] = s  # I[1:length(s)] = s
    return UnionSetArray([to_hyperrectangle(Ip) for Ip in Is])
