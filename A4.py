import numpy as np
from scipy.stats import uniform


class Product:
    def __init__(self, marginals):
        self.marginals = marginals

    def rvs(self, random_state=None):
        return np.array([m.rvs(random_state=random_state) for m in self.marginals])


class MassSpringDamper:
    def __init__(self, m=1.0, k=10.0, c=2.0, dt=0.05):
        self.m = m
        self.k = k
        self.c = c
        self.dt = dt

    def Ts(self):
        return np.array([[1.0, self.dt],
                         [-self.k * self.dt / self.m, 1 - self.c * self.dt / self.m]])

    def Ta(self):
        return np.array([[0.0],
                         [self.dt / self.m]])

    def step(self, s, a):
        return self.Ts() @ s + self.Ta() @ a

    def initial_distribution(self):
        # Julia Uniform(a, b) -> scipy uniform(a, b - a)
        return Product([uniform(-0.2, 0.4), uniform(-1e-12, 2e-12)])


class AdditiveNoiseSensor:
    def __init__(self, Do):
        self.Do = Do

    def observe(self, s, x=None):
        if x is None:
            x = self.Do_dist(s).rvs()
        return s + x

    def Do_dist(self, s):  # Julia Do(sensor, s); renamed to avoid clash with field Do
        return self.Do

    def Os(self):  # Julia I is size-agnostic; state is 2D here
        return np.eye(2)


class ProportionalController:
    def __init__(self, alpha):
        self.alpha = np.asarray(alpha)

    def act(self, o):
        return self.alpha.T @ o

    def Pi_o(self):
        return self.alpha.T