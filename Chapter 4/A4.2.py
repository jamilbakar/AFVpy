from collections import namedtuple
Disturbance = namedtuple("Disturbance", ["xa", "xs", "xo"])            # agent/env/sensor
DisturbanceDistribution = namedtuple("DisturbanceDistribution", ["Da", "Ds", "Do"])
StepResult = namedtuple("StepResult", ["o", "a", "s_next", "x"])       # Julia (; o, a, s′, x)


def step(sys, s, D):
    xo = D.Do(s).rvs()
    o = sys.sensor.observe(s, xo)
    xa = D.Da(o).rvs()
    a = sys.agent.act(o, xa)
    xs = D.Ds(s, a).rvs()
    s_next = sys.env.step(s, a, xs)
    x = Disturbance(xa, xs, xo)
    return StepResult(o, a, s_next, x)