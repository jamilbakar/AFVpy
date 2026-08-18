from collections import namedtuple
import numpy as np
import pytest
from conftest import load


def test_ch11_kmeans():
    a6 = load("A11.6")
    taus = [np.array([0, 0.1]), np.array([0.2, 0]), np.array([0.1, 0.2]),
            np.array([10, 10]), np.array([10.1, 9.9]), np.array([9.9, 10.1])]
    np.random.seed(0)
    km = a6.Kmeans(taus, phi=lambda t: t, d=lambda a, b: np.linalg.norm(a - b), k=2, max_iter=10)
    C, mu = km.describe(None, None)
    assert sorted(len(c) for c in C) == [3, 3]


def test_ch11_sensitivity(sys_disturbance):
    a1 = load("A11.1")
    D = namedtuple("D", ["xo", "xa", "xs"])
    a1.extract = lambda env, x: (0.0, list(x))
    a1.robustness = lambda states, f: min(states)

    def perturb(x, t):
        xp = list(x); xp[t] = D(0.0, 0.0, xp[t].xs + np.random.randn()); return xp

    x0 = [D(0, 0, 1.0), D(0, 0, 1.0), D(0, 0, 1.0)]
    np.random.seed(1)
    sens = a1.Sensitivity(x=x0, perturb=perturb, m=30).describe(
        sys_disturbance, namedtuple("P", ["formula"])(None))
    assert len(sens) == 3 and np.all(sens >= 0)


def test_ch11_shapley(sys_disturbance):
    pytest.importorskip("stljax")
    import stljax.formula as F
    a4 = load("A11.4")
    p = a4.NominalTrajectoryDistribution(sys_disturbance, 3)
    np.random.seed(0)
    tau = a4.rollout(sys_disturbance, p)
    psi = namedtuple("Spec", ["formula"])(F.Always(F.Predicate("s", lambda s: s) > -100.0))
    phis = a4.Shapley(tau, m=3).describe(sys_disturbance, psi)
    assert len(phis) == len(tau) and np.all(np.isfinite(phis))


def test_ch11_counterfactual(sys_disturbance):
    pytest.importorskip("stljax")
    import stljax.formula as F
    a5 = load("A11.5")
    D = namedtuple("D", ["xo", "xa", "xs"])
    a5.extract = lambda env, x: (x[0], [D(0.0, 0.0, x[i]) for i in range(1, len(x))])
    psi = namedtuple("Spec", ["formula"])(F.Always(F.Predicate("s", lambda s: s) > 0.0))
    assert np.isfinite(a5.counterfactual_objective([1.0, 0.5, 0.5], sys_disturbance, psi, [1.0, 0.0, 0.0]))


def test_ch11_gradient_and_integrated(sys_disturbance):
    pytest.importorskip("stljax")
    pytest.importorskip("jax")
    import jax.numpy as jnp
    import stljax.formula as F
    D = namedtuple("D", ["xo", "xa", "xs"])
    ex = lambda env, x: (x[0], [D(0.0, 0.0, x[i]) for i in range(1, len(x))])
    psi = namedtuple("Spec", ["formula"])(F.Always(F.Predicate("s", lambda s: s) > 0.0))
    a2 = load("A11.2"); a2.extract = ex
    g = a2.GradientSensitivity(x=jnp.array([1.0, 0.5, -2.0, 0.5])).describe(sys_disturbance, psi)
    assert g.shape == (4,) and np.all(np.isfinite(g))
    a3 = load("A11.3"); a3.extract = ex
    ig = a3.IntegratedGradients(x=[1.0, 0.5, -2.0, 0.5], b=[0.0, 0.0, 0.0, 0.0], m=5)
    assert np.all(np.isfinite(np.asarray(ig.describe(sys_disturbance, psi))))
