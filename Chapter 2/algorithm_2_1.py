# Algorithm 2.1: maximum likelihood parameter estimation.
# likelihood(x, theta) returns a distribution (with .logpdf); optimizer(f)
# minimizes f and returns the minimizing theta.

class MaximumLikelihoodParameterEstimation:
    def __init__(self, likelihood, optimizer):
        self.likelihood = likelihood
        self.optimizer = optimizer

    def fit(self, data):
        def f(theta):
            return sum(-self.likelihood(x, theta).logpdf(y) for x, y in data)
        return self.optimizer(f)