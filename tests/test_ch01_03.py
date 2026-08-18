from collections import namedtuple
import numpy as np
from conftest import load


def test_ch1_step_and_rollout():
    A1_1, A1_2, A2 = load("A1.1"), load("A1.2"), load("A2")
    sys = A1_1.System(A2.NoAgent(), A2.SimpleGaussian(), A2.IdealSensor())
    o, a, s_next = A1_2.step(sys, 1.5)
    assert o == 1.5 and a is None and s_next == 1.5
    assert len(A1_2.rollout(sys, 5)) == 5


def test_ch1_specification():
    m = load("A1.3")
    T = namedtuple("T", ["s"])

    class AllPositive(m.Specification):
        pass

    @m.evaluate.register
    def _(psi: AllPositive, tau):
        return all(t.s > 0 for t in tau)

    spec = AllPositive()
    assert m.isfailure(spec, [T(1), T(-1)]) is True
    assert m.isfailure(spec, [T(1), T(2)]) is False


def test_ch2_mle():
    from scipy.stats import norm
    from scipy.optimize import minimize
    m = load("A2.1")
    np.random.seed(0)
    data = [(0.0, v) for v in np.random.normal(3.0, 1.0, 300)]
    mle = m.MaximumLikelihoodParameterEstimation(
        likelihood=lambda x, th: norm(th, 1),
        optimizer=lambda f: minimize(f, [0.0]).x[0])
    assert abs(mle.fit(data) - 3.0) < 0.3


def test_ch3_ltl_stl():
    A3_1, A3_2 = load("A3.1"), load("A3.2")
    T = namedtuple("T", ["s"])
    tau = [T(s) for s in [100, 80, 60, 40, 55, 70]]
    ltl = A3_1.LTLSpecification(lambda st: all(abs(s) > 50 for s in st))
    assert A3_1.evaluate(ltl, tau) is False and A3_1.isfailure(ltl, tau) is True
    stl = A3_2.STLSpecification(lambda st: all(abs(s) > 50 for s in st), slice(4, 6))
    assert A3_2.evaluate(stl, tau) is True
