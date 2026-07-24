import math
def logpdf_disturbance(D, s, o, a, x):  # Julia Distributions.logpdf(D::DisturbanceDistribution, ...)
    logp_xa = D.Da(o).logpdf(x.xa)
    logp_xs = D.Ds(s, a).logpdf(x.xs)
    logp_xo = D.Do(s).logpdf(x.xo)
    return logp_xa + logp_xs + logp_xo


def initial_state_distribution(p):
    return p.Ps


def disturbance_distribution(p, t):
    return p.D


def pdf(p, tau):  # Julia Distributions.pdf(p::TrajectoryDistribution, τ)
    logprob = initial_state_distribution(p).logpdf(tau[0].s)
    for t, step in enumerate(tau):
        s, o, a, x = step
        logprob += logpdf_disturbance(disturbance_distribution(p, t), s, o, a, x)
    return math.exp(logprob)