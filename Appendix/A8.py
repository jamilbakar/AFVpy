import numpy as np
from scipy.stats import norm, uniform


class Product:
    def __init__(self, marginals):
        self.marginals = marginals

    def rvs(self, random_state=None):
        return np.array([m.rvs(random_state=random_state) for m in self.marginals])


class DiscreteNonParametric:
    def __init__(self, support, probs):
        self.support = np.asarray(support, dtype=float)
        self.probs = np.asarray(probs, dtype=float)

    def rvs(self, random_state=None):
        i = np.random.choice(len(self.support), p=self.probs)
        return self.support[i]


class CollisionAvoidance:
    def __init__(self, ddh_max=1.0, A=None, Ds=None):
        self.ddh_max = ddh_max
        self.A = [-5.0, 0.0, 5.0] if A is None else A
        self.Ds = norm() if Ds is None else Ds

    def Ds_dist(self, s, a):  # Julia Ds(env, s, a); renamed to avoid clash with field Ds
        return self.Ds

    def step(self, s, a, x=None):
        if x is None:
            x = self.Ds_dist(s, a).rvs()
        a = self.A[a]  # a is a 0-based action index (Julia is 1-based)
        h, dh, a_prev, tau = s
        h = h + dh
        if a != 0.0:
            if abs(a - dh) < self.ddh_max:
                dh += a
            else:
                dh += np.sign(a - dh) * self.ddh_max
        a_prev = a
        tau = max(tau - 1.0, -1.0)
        return [h, dh + x, a_prev, tau]

    def initial_distribution(self):
        # Julia Uniform(a, b) -> scipy uniform(a, b - a)
        return Product([uniform(-100, 200),
                        uniform(-10, 20),
                        DiscreteNonParametric([0], [1.0]),
                        DiscreteNonParametric([40], [1.0])])