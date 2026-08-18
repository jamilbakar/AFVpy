import math
from collections import namedtuple
from conftest import load  # noqa: F401  (ensures Chapter 9 dir is on sys.path)
import Interval_ad as ia   # same module the A9.x files import, so classes match

Interval = ia.Interval
TRUE_HI = 0.2 + 0.1 * math.sin(0.2)


def _system():
    D = namedtuple("D", ["xo", "xa", "xs"])

    class Env:
        def step(self, s, a, xs):
            return [s[0] + 0.1 * ia.psin(s[0]) + xs[0]]

    class Sensor:
        def observe(self, s, xo):
            return s

    class Agent:
        def act(self, o, xa):
            return 0.0

    Sys = namedtuple("Sys", ["agent", "env", "sensor"])
    return Sys(Agent(), Env(), Sensor()), D


def test_ch9_natural_inclusion_sound():
    a, D = _system(); m = load("algorithm_9_1")
    m.extract = lambda env, x: ([x[0]], [D(None, None, [x[1]]), D(None, None, [x[2]])])
    m.intervals = lambda sys, d: [Interval(0.0, 0.2), Interval(0.0, 0.0), Interval(0.0, 0.0)]
    box = m.reachable(m.NaturalInclusion(1), a).sets[0]
    assert box.low[0] <= 1e-9 and box.high[0] >= TRUE_HI - 1e-9


def test_ch9_taylor_inclusion_sound():
    a, D = _system(); m = load("algorithm_9_2")
    m.extract = lambda env, x: ([x[0]], [D(None, None, [x[1]]), D(None, None, [x[2]])])
    m.intervals = lambda sys, d: [Interval(0.0, 0.2), Interval(0.0, 0.0), Interval(0.0, 0.0)]
    for order in (1, 2):
        box = m.reachable(m.TaylorInclusion(1, order), a).sets[0]
        assert box.low[0] <= 1e-9 and box.high[0] >= TRUE_HI - 1e-9


def test_ch9_conservative_linearization_sound():
    a, D = _system(); m = load("algorithm_9_3")
    m.extract = lambda env, x: ([x[0]], [D(None, None, [x[1]]), D(None, None, [x[2]])])
    Hyper = m.Hyperrectangle
    m.sets = lambda sys, d: (Hyper(low=[0.0], high=[0.2]), Hyper(low=[0.0, 0.0], high=[0.0, 0.0]))
    box = m.reachable(m.ConservativeLinearization(1), a).sets[0]
    assert box.low[0] <= 1e-9 and box.high[0] >= TRUE_HI - 1e-9


def test_ch9_concrete_variants_run():
    a, D = _system()
    m4 = load("algorithm_9_4")
    m4.extract = lambda env, x: ([x[0]], [D(None, None, [x[i]]) for i in range(1, len(x))])
    m4.intervals = lambda sys, d: [Interval(0.0, 0.2), Interval(0.0, 0.0)]
    assert len(m4.reachable(m4.ConcreteTaylorInclusion(2, 1), a).sets) == 2
    m5 = load("algorithm_9_5")
    m5.extract = lambda env, x: ([x[0]], [D(None, None, [x[1]]), D(None, None, [x[2]])])
    Hyper = m5.Hyperrectangle
    m5.sets = lambda sys, d: (Hyper(low=[0.0], high=[0.2]), Hyper(low=[0.0, 0.0], high=[0.0, 0.0]))
    assert len(m5.reachable(m5.ConcreteConservativeLinearization(2), a).sets) == 2
