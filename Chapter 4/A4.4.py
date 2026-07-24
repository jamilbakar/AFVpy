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


def depth(p):
    return p.d