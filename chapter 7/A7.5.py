import math
import numpy as np
from collections import namedtuple

Disturbance = namedtuple("Disturbance", ["xa", "xs", "xo"])
Transition = namedtuple("Transition", ["s", "o", "a", "x"])


def rollout(sys, p, d=None):
    if d is None:
        d = p.d
    s = p.Ps.rvs()
    tau = []
    for t in range(d):
        xo = p.D.Do(s).rvs(); o = sys.sensor.observe(s, xo)
        xa = p.D.Da(o).rvs(); a = sys.agent.act(o, xa)
        xs = p.D.Ds(s, a).rvs(); s_next = sys.env.step(s, a, xs)
        tau.append(Transition(s, o, a, Disturbance(xa, xs, xo)))
        s = s_next
    return tau


def pdf(p, tau):
    logprob = p.Ps.logpdf(tau[0].s)
    for st in tau:
        s, o, a, x = st
        logprob += p.D.Da(o).logpdf(x.xa) + p.D.Ds(s, a).logpdf(x.xs) + p.D.Do(s).logpdf(x.xo)
    return math.exp(logprob)


def isfailure(psi, tau):
    return not psi.evaluate(tau)


def fit(proposal_type, taus, ws):
    raise NotImplementedError  # proposal-type-specific weighted maximum-likelihood fit


class ImportanceSamplingEstimation:
    def __init__(self, p, q, m):
        self.p, self.q, self.m = p, q, m

    def estimate(self, sys, psi):
        taus = [rollout(sys, self.q) for _ in range(self.m)]
        ws = [pdf(self.p, tau) / pdf(self.q, tau) for tau in taus]
        return float(np.mean([w * isfailure(psi, tau) for w, tau in zip(ws, taus)]))


class CrossEntropyEstimation:
    def __init__(self, p, q0, f, k_max, m, m_elite):
        self.p, self.q0, self.f = p, q0, f
        self.k_max, self.m, self.m_elite = k_max, m, m_elite

    def estimate(self, sys, psi):
        p, q, f = self.p, self.q0, self.f
        for k in range(self.k_max):
            taus = [rollout(sys, q) for _ in range(self.m)]
            Y = np.array([f(tau, psi) for tau in taus])
            order = np.argsort(Y)
            gamma = max(0, Y[order[self.m_elite - 1]])  # Julia Y[order[m_elite]] (1-based)
            ps = np.array([pdf(p, tau) for tau in taus])
            qs = np.array([pdf(q, tau) for tau in taus])
            ws = ps / qs
            ws[Y > gamma] = 0
            q = fit(type(q), taus, ws=ws)
        return ImportanceSamplingEstimation(p, q, self.m).estimate(sys, psi)
