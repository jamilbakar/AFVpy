from collections import namedtuple
import numpy as np
import pytest
from conftest import load, Normal


def test_ch4_disturbance_step(sys_disturbance):
    a2 = load("A4.2")
    D = a2.DisturbanceDistribution(Da=lambda o: Normal(0, 1), Ds=lambda s, a: Normal(0, 1),
                                   Do=lambda s: Normal(0, 1))
    r = a2.step(sys_disturbance, 0.0, D)
    assert r._fields == ("o", "a", "s_next", "x")


def test_ch4_nominal_and_rollout(sys_disturbance):
    a4, a5 = load("A4.4"), load("A4.5")
    p = a4.NominalTrajectoryDistribution(sys_disturbance, 5)
    assert a4.depth(p) == 5 and len(a5.rollout(sys_disturbance, p)) == 5


def test_ch4_fixed_rollout_and_pdf(sys_disturbance, traj_distribution):
    a6, a8, a2 = load("A4.6"), load("A4.8"), load("A4.2")
    xtraj = [a2.Disturbance(0.0, 0.0, 0.0) for _ in range(4)]
    tau = a6.rollout(sys_disturbance, 0.0, xtraj)
    assert len(tau) == 4 and a8.pdf(traj_distribution, tau) > 0


def test_ch4_direct_falsification(sys_disturbance, spec_threshold):
    a1 = load("A4.1")
    assert isinstance(a1.DirectFalsification(d=3, m=20).falsify(sys_disturbance, spec_threshold), list)


def test_ch4_optimization_based_falsification():
    a11 = load("A4.11")
    alg = a11.OptimizationBasedFalsification(
        objective=lambda x, s, p: (x - 3) ** 2, optimizer=lambda f, s, p: min(range(7), key=f))
    assert alg.falsify(None, None) == 3


def _stl_spec():
    import stljax.formula as F

    class Spec:
        formula = F.Always(F.Predicate("s", lambda s: s) > 0.0)

        def evaluate(self, tau):
            return all(getattr(t, "s", t) < 100 for t in tau)

    return Spec()


def test_ch4_robustness_objective(sys_disturbance):
    pytest.importorskip("stljax")
    a7 = load("A4.7")
    D = namedtuple("D", ["xo", "xa", "xs"])
    a7.extract = lambda env, x: (x[0], [D(0.0, 0.0, x[i]) for i in range(1, len(x))])
    psi = _stl_spec()
    assert a7.robustness_objective([1.0, 0.5, 0.5, 0.5], sys_disturbance, psi) > 0
    assert a7.robustness_objective([1.0, -2.0, 0.5, 0.5], sys_disturbance, psi) < 0


def test_ch4_likelihood_objectives(sys_disturbance):
    pytest.importorskip("stljax")
    D = namedtuple("D", ["xo", "xa", "xs"])
    ex = lambda env, x: (x[0], [D(0.0, 0.0, x[i]) for i in range(1, len(x))])
    a9 = load("A4.9"); a9.extract = ex
    a10 = load("A4.10"); a10.extract = ex
    assert np.isfinite(a9.likelihood_objective([1.0, 0.5, 0.5], sys_disturbance, _stl_spec()))
    assert np.isfinite(a10.weighted_likelihood_objective([1.0, 0.5, 0.5], sys_disturbance, _stl_spec()))
