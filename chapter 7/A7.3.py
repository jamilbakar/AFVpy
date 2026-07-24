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


class ImportanceSamplingEstimation:
    def __init__(self, p, q, m):
        self.p = p  # nominal distribution
        self.q = q  # proposal distribution
        self.m = m  # number of samples

    def estimate(self, sys, psi):
        taus = [rollout(sys, self.q) for _ in range(self.m)]
        ps = [pdf(self.p, tau) for tau in taus]
        qs = [pdf(self.q, tau) for tau in taus]
        ws = [pi / qi for pi, qi in zip(ps, qs)]
        return float(np.mean([w * isfailure(psi, tau) for w, tau in zip(ws, taus)]))
