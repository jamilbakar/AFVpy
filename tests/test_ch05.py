from collections import namedtuple
import numpy as np
import pytest
from conftest import load


def test_ch5_average_dispersion():
    pts = [np.array([0.2, 0.2]), np.array([0.8, 0.8]), np.array([0.5, 0.5])]
    assert 0 < load("A5.7").average_dispersion(pts, [0, 0], [1, 1], [5, 5]) <= 1


def test_ch5_star_discrepancy():
    pts = [np.array([0.2, 0.2]), np.array([0.8, 0.8]), np.array([0.5, 0.5])]
    lb, ub = load("A5.8").star_discrepancy(pts, [0, 0], [1, 1], [4, 4])
    assert lb <= ub


def test_ch5_lcb():
    Ch = namedtuple("Ch", ["Q", "N"]); Nd = namedtuple("Nd", ["N", "children"])
    parent = Nd(N=100, children=[Ch(1.0, 10), Ch(0.1, 10), Ch(5.0, 10)])
    assert load("A5.11").lcb(parent, 1.0).Q == 0.1


def test_ch5_rrt_falsify(sys_disturbance, spec_threshold):
    a2, a4, a5 = load("A5.2"), load("A5.4"), load("A5.5")
    rrt = a4.RRT(sample_goal=lambda tree: a5.random_goal(tree, [-5], [5]),
                 compute_objectives=a5.distance_objectives,
                 select_disturbance=a5.random_disturbance, k_max=20)
    np.random.seed(1)
    assert isinstance(a2.falsify(rrt, sys_disturbance, spec_threshold), list)


def test_ch5_mcts_falsify(sys_disturbance, spec_threshold):
    a2, a10, a5 = load("A5.2"), load("A5.10"), load("A5.5")
    mcts = a10.MCTS(estimate_value=lambda sys, psi, s: float(abs(s)), c=1.0, k=2.0, alpha=0.5,
                    select_disturbance=a5.random_disturbance, k_max=20)
    np.random.seed(1)
    assert isinstance(a2.falsify(mcts, sys_disturbance, spec_threshold), list)


def test_ch5_shooting_robustness(sys_disturbance):
    pytest.importorskip("stljax")
    import stljax.formula as F
    a1 = load("A5.1")
    D = namedtuple("D", ["xo", "xa", "xs"]); Seg = namedtuple("Seg", ["s", "x"])
    a1.extract = lambda env, x: [Seg(0.0, [D(0, 0, 0.0)]), Seg(0.5, [D(0, 0, 0.0)])]
    psi = namedtuple("Spec", ["formula"])(F.Always(F.Predicate("s", lambda s: s) > -100.0))
    assert np.isfinite(a1.shooting_robustness([0.0] * 4, sys_disturbance, psi))
