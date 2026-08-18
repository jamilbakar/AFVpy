import numpy as np
from collections import namedtuple
Disturbance = namedtuple("Disturbance", ["xa", "xs", "xo"])
StepResult = namedtuple("StepResult", ["o", "a", "s_next", "x"])


class DisturbanceDistribution:
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


def goal_disturbance(sys, node, m=10):
    D = DisturbanceDistribution(sys)
    steps = [step(sys, node.state, D) for _ in range(m)]
    distances = [np.linalg.norm(node.goal_state - st.s_next) for st in steps]
    return steps[int(np.argmin(distances))].x