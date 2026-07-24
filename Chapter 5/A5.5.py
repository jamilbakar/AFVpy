import numpy as np
from collections import namedtuple
Disturbance = namedtuple("Disturbance", ["xa", "xs", "xo"])
StepResult = namedtuple("StepResult", ["o", "a", "s_next", "x"])


class DisturbanceDistribution:  # DisturbanceDistribution(sys): nominal component dists
    def __init__(self, sys):
        self.Da = lambda o: sys.agent.Da(o)
        self.Ds = lambda s, a: sys.env.Ds(s, a)
        self.Do = lambda s: sys.sensor.Do(s)


def step(sys, s, D):
    xo = D.Do(s).rvs()
    o = sys.sensor.observe(s, xo)
    xa = D.Da(o).rvs()
    a = sys.agent.act(o, xa)
    xs = D.Ds(s, a).rvs()
    s_next = sys.env.step(s, a, xs)
    return StepResult(o, a, s_next, Disturbance(xa, xs, xo))


def random_goal(tree, lo, hi):
    return np.array([np.random.uniform(l, h) for l, h in zip(lo, hi)])


def distance_objectives(tree, sgoal):
    return [np.linalg.norm(sgoal - node.state) for node in tree]


def random_disturbance(sys, node):
    D = DisturbanceDistribution(sys)
    o, a, s_next, x = step(sys, node.state, D)
    return x