import math
import numpy as np

def pdf(p, tau):
    logprob = p.Ps.logpdf(tau[0].s)
    for st in tau:
        s, o, a, x = st
        logprob += p.D.Da(o).logpdf(x.xa) + p.D.Ds(s, a).logpdf(x.xs) + p.D.Do(s).logpdf(x.xo)
    return math.exp(logprob)


def isfailure(psi, tau):
    return not psi.evaluate(tau)


class SelfImportanceSamplingEstimation:
    def __init__(self, p, q_bar, q_bar_taus):
        self.p = p                    # nominal distribution
        self.q_bar = q_bar            # unnormalized proposal density
        self.q_bar_taus = q_bar_taus  # samples from q_bar

    def estimate(self, sys, psi):
        p, q_bar, q_bar_taus = self.p, self.q_bar, self.q_bar_taus
        ws = np.array([pdf(p, tau) / q_bar(tau) for tau in q_bar_taus])
        ws = ws / ws.sum()
        # NOTE: reproduced exactly from the source (uses mean). Eq 7.33 for SNIS is a
        # weighted sum of the normalized weights; swap np.mean -> np.sum if matching it.
        return float(np.mean([w * isfailure(psi, tau) for w, tau in zip(ws, q_bar_taus)]))
