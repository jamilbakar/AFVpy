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
    raise NotImplementedError  # system-specific: produce samples from the next density


def bridge_sampling_estimator(g1taus, g1, g2taus, g2, gb):
    g1s = np.array([g1(t) for t in g1taus])
    g2s = np.array([g2(t) for t in g2taus])
    gb1s = np.array([gb(t) for t in g1taus])
    gb2s = np.array([gb(t) for t in g2taus])
    return np.mean(gb2s / g2s) / np.mean(gb1s / g1s)


def optimal_bridge(g1taus, g1, g2taus, g2, k_max):
    ratio = 1.0
    m1, m2 = len(g1taus), len(g2taus)

    def gb(tau):
        return (g1(tau) * g2(tau)) / (m1 * g1(tau) + m2 * ratio * g2(tau))

    for k in range(k_max):
        ratio = bridge_sampling_estimator(g1taus, g1, g2taus, g2, gb)
    return gb


class BridgeSamplingEstimation:
    def __init__(self, p, gs, perturb, m, kb):
        self.p, self.gs, self.perturb, self.m, self.kb = p, gs, perturb, m, kb

    def estimate(self, sys, psi):
        p, gs, perturb, m, kb = self.p, self.gs, self.perturb, self.m, self.kb

        def p_bar_failure(tau):
            return isfailure(psi, tau) * pdf(p, tau)

        def p_density(tau):
            return pdf(p, tau)  # p used as a density: g(tau) = pdf(p, tau)

        taus = [rollout(sys, p) for _ in range(m)]
        p_hat_fail = 1.0
        for g, g_next in zip([p_density] + list(gs), list(gs) + [p_bar_failure]):
            ws = np.array([g_next(t) / g(t) for t in taus])
            idx = np.random.choice(len(ws), size=m, p=ws / ws.sum())
            taus_new = [taus[i] for i in idx]
            taus_new = perturb(taus_new, g_next)
            gb = optimal_bridge(taus_new, g_next, taus, g, kb)
            ratio = bridge_sampling_estimator(taus_new, g_next, taus, g, gb)
            p_hat_fail *= ratio
            taus = taus_new
        return p_hat_fail
