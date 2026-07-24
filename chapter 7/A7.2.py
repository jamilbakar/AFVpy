import numpy as np
from collections import namedtuple
from scipy.stats import beta as Beta

Disturbance = namedtuple("Disturbance", ["xa", "xs", "xo"])
Transition = namedtuple("Transition", ["s", "o", "a", "x"])
DisturbanceDistribution = namedtuple("DisturbanceDistribution", ["Da", "Ds", "Do"])


class NominalTrajectoryDistribution:
    def __init__(self, sys, d):
        self.D = DisturbanceDistribution(
            Da=lambda o: sys.agent.Da(o),
            Ds=lambda s, a: sys.env.Ds(s, a),
            Do=lambda s: sys.sensor.Do(s))
        self.Ps = sys.env.initial_distribution()
        self.d = d


def rollout(sys, p=None, d=None):
    if p is None:
        p = NominalTrajectoryDistribution(sys, d)
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


def isfailure(psi, tau):
    return not psi.evaluate(tau)


class BayesianEstimation:
    def __init__(self, prior, d, m):
        self.prior = prior  # scipy Beta
        self.d = d
        self.m = m

    def estimate(self, sys, psi):
        taus = [rollout(sys, d=self.d) for _ in range(self.m)]
        n = sum(isfailure(psi, tau) for tau in taus)
        m = len(taus)
        alpha, beta = self.prior.args  # prior.α, prior.β
        return Beta(alpha + n, beta + m - n)
