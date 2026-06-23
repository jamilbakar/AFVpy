# Algorithm 2.2: Bayesian parameter estimation.
# Turing.jl has no drop-in Python equivalent, so the @model is expressed as a
# log-posterior: theta ~ prior contributes prior.logpdf(theta), and each
# y[i] ~ likelihood(x[i], theta) contributes likelihood(x[i], theta).logpdf(y[i]).
# sampler(log_posterior, m) plays the role of Turing.sample(..., sampler, m)
# and returns m samples from the posterior.

class BayesianParameterEstimation:
    def __init__(self, likelihood, prior, sampler, m):
        self.likelihood = likelihood
        self.prior = prior
        self.sampler = sampler
        self.m = m

    def fit(self, data):
        x = [d[0] for d in data]
        y = [d[1] for d in data]

        def log_posterior(theta):
            lp = self.prior.logpdf(theta)
            for xi, yi in zip(x, y):
                lp += self.likelihood(xi, theta).logpdf(yi)
            return lp

        return self.sampler(log_posterior, self.m)