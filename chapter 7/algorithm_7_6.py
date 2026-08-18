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


def proposal(q, tau):
    raise NotImplementedError  # trajectory-dist-specific: build a proposal from a sample


def smis(p, qs, taus):
    return [pdf(p, tau) / pdf(q, tau) for q, tau in zip(qs, taus)]


def dmmis(p, qs, taus):
    return [pdf(p, tau) / np.mean([pdf(q, tau) for q in qs]) for tau in taus]


class MultipleImportanceSamplingEstimation:
    def __init__(self, p, qs, weighting):
        self.p, self.qs, self.weighting = p, qs, weighting

    def estimate(self, sys, psi):
        taus = [rollout(sys, q) for q in self.qs]
        ws = self.weighting(self.p, self.qs, taus)
        return float(np.mean([w * isfailure(psi, tau) for w, tau in zip(ws, taus)]))


class PopulationMonteCarloEstimation:
    def __init__(self, p, qs, weighting, k_max):
        self.p, self.qs, self.weighting, self.k_max = p, qs, weighting, k_max

    def estimate(self, sys, psi):
        p, qs, weighting = self.p, self.qs, self.weighting
        m = len(qs)
        for k in range(self.k_max):
            taus = [rollout(sys, q) for q in qs]
            ws = np.array([pdf(p, tau) * isfailure(psi, tau) / pdf(q, tau)
                           for q, tau in zip(qs, taus)])
            idx = np.random.choice(len(ws), size=m, p=ws / ws.sum())  # rand(Categorical, m)
            qs = [proposal(qs[i], taus[i]) for i in idx]
        mis = MultipleImportanceSamplingEstimation(p, qs, weighting)
        return mis.estimate(sys, psi)
