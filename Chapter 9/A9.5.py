import numpy as np
from collections import namedtuple

# Algorithm 9.5: nonlinear forward reachability using conservative linearization,
# concretizing the reachable set at each step. Self-sufficient: includes
# conservative_linearization (9.3). WALLs: IntervalArithmetic, ForwardDiff over
# intervals, and LazySets (Hyperrectangle, UnionSetArray, ⊕, ×, low/high,
# interval_hull) have no drop-in Python equivalents; sets is system-specific.

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


def sets(sys, d):
    raise NotImplementedError  # system-specific


def jacobian(f, x):
    raise NotImplementedError  # ForwardDiff.jacobian


def hessian(f, x):
    raise NotImplementedError  # ForwardDiff.hessian over intervals


def interval(lo, hi):
    return Interval(lo, hi)


def interval_hull(P):
    raise NotImplementedError  # LazySets


def low(P):
    raise NotImplementedError  # LazySets


def high(P):
    raise NotImplementedError  # LazySets


def minkowski_sum(A, B):
    return A + B


def linear_map(M, S):
    return M @ S


def cartesian_product(A, B):
    return A * B


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


def to_intervals(P):
    return [interval(lo, hi) for lo, hi in zip(low(P), high(P))]


def conservative_linearization(sys, P):
    I = to_intervals(interval_hull(P))
    c = [mid(i) for i in I]
    fc = r(sys, c)
    J = jacobian(lambda x: r(sys, x), c)
    Ic = np.array(I) - np.array(c)
    alpha = to_hyperrectangle([Ic @ hessian(lambda x: r(sys, x)[i], I) @ Ic
                               for i in range(len(fc))])
    return minkowski_sum(fc + linear_map(J, minkowski_sum(P, [-ci for ci in c])), alpha)


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