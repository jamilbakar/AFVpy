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


def perturb(taus, g):
    raise NotImplementedError  # system-specific: transition samples toward density g


class SequentialMonteCarloEstimation:
    def __init__(self, p, gs, perturb, m):
        self.p, self.gs, self.perturb, self.m = p, gs, perturb, m

    def estimate(self, sys, psi):
        p, gs, perturb, m = self.p, self.gs, self.perturb, self.m

        def p_bar_failure(tau):
            return isfailure(psi, tau) * pdf(p, tau)

        taus = [rollout(sys, p) for _ in range(m)]
        ws = np.array([gs[0](tau) / pdf(p, tau) for tau in taus])
        for g, g_next in zip(gs, list(gs[1:]) + [p_bar_failure]):
            taus_new = perturb(taus, g)
            ws = ws * np.array([g_next(tau) / g(tau) for tau in taus_new])
            idx = np.random.choice(len(ws), size=m, p=ws / ws.sum())
            taus = [taus_new[i] for i in idx]
            ws = np.full(m, ws.mean())
        return float(np.mean(ws))
