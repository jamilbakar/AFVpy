from collections import namedtuple
from conftest import load


def _chain():
    class Env:
        def states(self): return [0, 1, 2]
        def S1(self): return [0]
        def initial_distribution(self):
            class Dist:
                def pdf(self, s): return 1.0 if s == 0 else 0.0
            return Dist()

    class Sys:
        env = Env()
        def successors(self, s):
            return {0: ([1, 0], [0.5, 0.5]), 1: ([2], [1.0]), 2: ([2], [1.0])}[s]
    return Sys()


def test_ch10_to_graph():
    assert load("algorithm_10_1").to_graph(_chain())._succ[0] == {1: 0.5, 0: 0.5}


def test_ch10_forward_backward():
    a2, a3 = load("algorithm_10_2"), load("algorithm_10_3")
    Spec = namedtuple("Spec", ["set"])
    assert sorted(a2.reachable(a2.DiscreteForward(5), _chain())) == [0, 1, 2]
    assert sorted(a3.backward_reachable(a3.DiscreteBackward(5), _chain(), Spec({2}))) == [0, 1, 2]


def test_ch10_occupancy():
    a4 = load("algorithm_10_4")
    assert abs(a4.reachable(a4.ProbabilisticOccupancy(3), _chain()).P[2] - 0.5) < 1e-9


def test_ch10_horizons():
    a5, a6 = load("algorithm_10_5"), load("algorithm_10_6")
    Spec = namedtuple("Spec", ["set"])
    assert abs(a5.reachable(a5.ProbabilisticFiniteHorizon(5), _chain(), Spec({2})) - 0.875) < 1e-9
    assert abs(a6.reachable(a6.ProbabilisticInfiniteHorizon(), _chain(), Spec({2})) - 1.0) < 1e-9
