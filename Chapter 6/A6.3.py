from collections import namedtuple
DisturbanceDistribution = namedtuple("DisturbanceDistribution", ["Da", "Ds", "Do"])


class NominalTrajectoryDistribution:
    def __init__(self, sys, d):
        self.D = DisturbanceDistribution(
            Da=lambda o: sys.agent.Da(o),
            Ds=lambda s, a: sys.env.Ds(s, a),
            Do=lambda s: sys.sensor.Do(s))
        self.Ps = sys.env.initial_distribution()
        self.d = d


def initial_state_distribution(p):
    return p.Ps


def disturbance_distribution(p, t):
    return p.D


class ProbabilisticProgramming:
    def __init__(self, Delta, mcmc_alg, k_max, d, epsilon):
        self.Delta = Delta        # distance function Delta(states)
        self.mcmc_alg = mcmc_alg  # e.g. a NUTS/MH sampler
        self.k_max = k_max        # number of samples
        self.d = d                # trajectory depth
        self.epsilon = epsilon    # smoothing parameter

    def sample_failures(self, sys, psi):
        from scipy.stats import norm
        Delta, d, eps = self.Delta, self.d, self.epsilon

        # The Turing @model rollout, as a log-density over latents.
        def log_density(s0, xo, xa, xs):
            p = NominalTrajectoryDistribution(sys, d)
            logp = initial_state_distribution(p).logpdf(s0)  # s ~ initial_state_distribution
            s_list = [s0] + [None] * d                        # 𝐬
            for t in range(d):
                D = disturbance_distribution(p, t)
                s = s_list[t]
                logp += D.Do(s).logpdf(xo[t])                 # xo[t] ~ D.Do(s)
                o = sys.sensor.observe(s, xo[t])
                logp += D.Da(o).logpdf(xa[t])                 # xa[t] ~ D.Da(o)
                a = sys.agent.act(o, xa[t])
                logp += D.Ds(s, a).logpdf(xs[t])              # xs[t] ~ D.Ds(s, a)
                s_list[t + 1] = sys.env.step(s, a, xs[t])
            logp += norm(0.0, eps).logpdf(Delta(s_list))      # @addlogprob!
            return logp

        return self.mcmc_alg(log_density, sys, d, self.k_max)