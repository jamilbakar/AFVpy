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


class MCMCSampling:
    def __init__(self, p_bar, g, tau, k_max, m_burnin, m_skip):
        self.p_bar = p_bar        # target density
        self.g = g                # kernel: tau' = rollout(sys, g(tau))
        self.tau = tau            # initial trajectory
        self.k_max = k_max
        self.m_burnin = m_burnin  # samples discarded as burn-in
        self.m_skip = m_skip      # thinning stride

    def sample_failures(self, sys, psi):
        p_bar, g, tau = self.p_bar, self.g, self.tau
        taus = []
        for k in range(self.k_max):
            tau_prime = rollout(sys, g(tau))
            ratio = (p_bar(tau_prime) * pdf(g(tau_prime), tau)) / \
                    (p_bar(tau) * pdf(g(tau), tau_prime))
            if np.random.rand() < ratio:
                tau = tau_prime
            taus.append(tau)
        # Julia τs[m_burnin:m_skip:end] is 1-based inclusive with stride m_skip
        return taus[self.m_burnin - 1::self.m_skip]