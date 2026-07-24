import math
import numpy as np
from collections import namedtuple
Disturbance = namedtuple("Disturbance", ["xa", "xs", "xo"])
Transition = namedtuple("Transition", ["s", "o", "a", "x"])


def step(sys, s, D):
    xo = D.Do(s).rvs()
    o = sys.sensor.observe(s, xo)
    xa = D.Da(o).rvs()
    a = sys.agent.act(o, xa)
    xs = D.Ds(s, a).rvs()
    s_next = sys.env.step(s, a, xs)
    return o, a, s_next, Disturbance(xa, xs, xo)


def initial_state_distribution(p):
    return p.Ps


def disturbance_distribution(p, t):
    return p.D


def depth(p):
    return p.d


def rollout(sys, p, d=None):
    if d is None:
        d = depth(p)
    s = initial_state_distribution(p).rvs()
    tau = []
    for t in range(d):
        o, a, s_next, x = step(sys, s, disturbance_distribution(p, t))
        tau.append(Transition(s, o, a, x))
        s = s_next
    return tau


def pdf(p, tau):
    logprob = initial_state_distribution(p).logpdf(tau[0].s)
    for t, st in enumerate(tau):
        s, o, a, x = st
        D = disturbance_distribution(p, t)
        logprob += D.Da(o).logpdf(x.xa) + D.Ds(s, a).logpdf(x.xs) + D.Do(s).logpdf(x.xo)
    return math.exp(logprob)


class RejectionSampling:
    def __init__(self, p_bar, q, c, k_max):
        self.p_bar = p_bar  # target density
        self.q = q          # proposal trajectory distribution
        self.c = c          # constant such that p_bar(tau) <= c * q(tau)
        self.k_max = k_max

    def sample_failures(self, sys, psi):
        taus = []
        for k in range(self.k_max):
            tau = rollout(sys, self.q)
            if np.random.rand() < self.p_bar(tau) / (self.c * pdf(self.q, tau)):
                taus.append(tau)
        return taus