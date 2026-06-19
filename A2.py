from scipy.stats import norm

class SimpleGaussian:
    def step(self, s, a):
        return s
    def initial_distribution(self):
        return norm(0, 1)

class NoAgent:
    def act(self, s):
        return None

class IdealSensor:
    def observe(self, s):
        return s