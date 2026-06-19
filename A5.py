import numpy as np
from scipy.stats import uniform


class Product:
    def __init__(self, marginals):
        self.marginals = marginals

    def rvs(self, random_state=None):
        return np.array([m.rvs(random_state=random_state) for m in self.marginals])


class InvertedPendulum:
    def __init__(self, m=1.0, l=1.0, g=10.0, dt=0.05, w_max=8.0, a_max=2.0):
        self.m = m          # mass of the pendulum
        self.l = l          # length of the pendulum
        self.g = g          # acceleration due to gravity
        self.dt = dt        # time step
        self.w_max = w_max  # maximum angular velocity
        self.a_max = a_max  # maximum torque

    def step(self, s, a):
        theta, w = s[0], s[1]  # Julia s[1], s[2] (1-based)
        dt, g, m, l = self.dt, self.g, self.m, self.l
        a = np.clip(a, -self.a_max, self.a_max)
        w = w + (3 * g / (2 * l) * np.sin(theta) + 3 * a / (m * l ** 2)) * dt
        theta = theta + w * dt
        w = np.clip(w, -self.w_max, self.w_max)
        return np.array([theta, w])

    def initial_distribution(self):
        # Julia Uniform(a, b) -> scipy uniform(a, b - a)
        return Product([uniform(-np.pi / 16, np.pi / 8), uniform(-1.0, 2.0)])