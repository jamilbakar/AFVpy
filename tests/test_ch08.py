from collections import namedtuple
import numpy as np
import pytest
from conftest import load


def test_ch8_avoid_set_spec():
    a1 = load("A8.1")

    class Box:
        def __init__(self, lo, hi):
            self.lo, self.hi = lo, hi

        def __contains__(self, p):
            return all(self.lo <= x <= self.hi for x in p)

    T = namedtuple("T", ["s"])
    psi = a1.AvoidSetSpecification(Box(-1, 1))
    assert a1.evaluate(psi, [T([2.0]), T([3.0])]) is True
    assert a1.evaluate(psi, [T([2.0]), T([0.0])]) is False


def _box_system():
    class Box:
        def __init__(self, lo, hi):
            self.lo = np.array(lo, float); self.hi = np.array(hi, float); self.dim = len(self.lo)
            I = np.eye(self.dim)
            self.A = np.vstack([I, -I]); self.b = np.concatenate([self.hi, -self.lo])

    class Env:
        def Ts(self): return np.array([[1.0]])
        def Ta(self): return np.array([[1.0]])
        def S1(self): return Box([0.0], [0.0])
    class Agent:
        def Pi_o(self): return np.array([[0.0]])
    class Sensor:
        def Os(self): return np.array([[1.0]])
    Xset = namedtuple("Xset", ["xo", "xa", "xs"])
    Sys = namedtuple("Sys", ["agent", "env", "sensor", "disturbance_set"])
    return Sys(Agent(), Env(), Sensor(),
               lambda: Xset(xo=Box([0], [0]), xa=Box([0], [0]), xs=Box([-1], [1]))), Box


def test_ch8_support_function():
    pytest.importorskip("cvxpy")
    a6 = load("A8.6")
    sys, _ = _box_system()
    m = a6.constrained_model(sys, 3, sys.env.S1(), sys.disturbance_set())
    assert abs(a6.rho(m, np.array([1.0]), 3) - 2.0) < 1e-3


def test_ch8_satisfies_lp():
    pytest.importorskip("cvxpy")
    a8 = load("A8.8")
    sys, Box = _box_system()
    Avoid = namedtuple("Avoid", ["set"])
    alg = a8.LinearProgramming(h=3, D=None, tol=1e-4)
    assert a8.satisfies(alg, sys, Avoid(Box([10], [11]))) is True
    assert a8.satisfies(alg, sys, Avoid(Box([1.5], [2.0]))) is False


def test_ch8_pycvxset_import():
    pytest.importorskip("pycvxset")
    assert hasattr(load("A8.7"), "reachable")
