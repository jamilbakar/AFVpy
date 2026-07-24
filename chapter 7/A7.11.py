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
    raise NotImplementedError  # system-specific: perturb samples toward density g


class AdaptiveMultilevelSplitting:
    def __init__(self, p, m, m_elite, k_max, f, perturb):
        self.p, self.m, self.m_elite = p, m, m_elite
        self.k_max, self.f, self.perturb = k_max, f, perturb

    def estimate(self, sys, psi):
        p, m, m_elite, k_max = self.p, self.m, self.m_elite, self.k_max
        f, perturb = self.f, self.perturb
        taus = [rollout(sys, p) for _ in range(m)]
        p_hat_fail = 1.0
        for i in range(k_max):
            Y = np.array([f(tau, psi) for tau in taus])
            order = np.argsort(Y)
            gamma = 0 if i == k_max - 1 else max(0, Y[order[m_elite - 1]])
            p_hat_fail *= float(np.mean(Y <= gamma))
            if gamma == 0:
                break
            elite = [taus[order[j]] for j in range(m_elite)]
            taus = [elite[i] for i in np.random.choice(len(elite), size=m)]

            def p_bar_gamma(tau, g=gamma):
                return pdf(p, tau) * (f(tau, psi) <= g)

            taus = perturb(taus, p_bar_gamma)
        return p_hat_fail
