try:
    import numpy as np
except ImportError as e:
    raise ImportError("numpy is required to run A3.py. Install it with: pip install numpy") from e

try:
    from scipy.stats import multivariate_normal
except ImportError as e:
    raise ImportError("scipy is required to run A3.py. Install it with: pip install scipy") from e
class MvGaussian:
    # (env::MvGaussian)(s, a) = s
    def step(self, s, a):
        return s
 
    # Ps(env::MvGaussian) = MvNormal(zeros(2), I)
    def initial_distribution(self):
        return multivariate_normal(np.zeros(2), np.eye(2))