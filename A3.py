import numpy as np
from scipy.stats import multivariate_normal
class MvGaussian:
    # (env::MvGaussian)(s, a) = s
    def step(self, s, a):
        return s
 
    # Ps(env::MvGaussian) = MvNormal(zeros(2), I)
    def initial_distribution(self):
        return multivariate_normal(np.zeros(2), np.eye(2))