import numpy as np
from collections import namedtuple

# Algorithm 9.1: nonlinear forward reachability using natural inclusion functions.
# For each depth it gets the input intervals and propagates them through the rollout
# function using interval arithmetic. WALLs: IntervalArithmetic.jl (intervals flow
# through the ordinary rollout via operator overloading) and LazySets.jl
# (Hyperrectangle/UnionSetArray) have no drop-in Python equivalent. intervals and
# extract are system-specific; the interval-aware component ops are assumed.

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
    raise NotImplementedError  # system-specific: x -> (initial_state, disturbance_traj)


def intervals(sys, d):
    raise NotImplementedError  # system-specific: input intervals for a depth-d rollout


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


class ReachabilityAlgorithm:
    pass


class NaturalInclusion(ReachabilityAlgorithm):
    def __init__(self, h):
        self.h = h


def reachable(alg, sys):
    Is = []
    for d in range(1, alg.h + 1):  # Julia 1:h
        I = intervals(sys, d)
        Is.append(r(sys, I))
    return UnionSetArray([to_hyperrectangle(Ip) for Ip in Is])