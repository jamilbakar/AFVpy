import numpy as np
from scipy.stats import beta
from conftest import load


def test_ch7_direct_estimation(sys_disturbance, spec_threshold):
    np.random.seed(0)
    assert 0.0 <= load("A7.1").DirectEstimation(3, 200).estimate(sys_disturbance, spec_threshold) <= 1.0


def test_ch7_bayesian_estimation(sys_disturbance, spec_threshold):
    np.random.seed(0)
    post = load("A7.2").BayesianEstimation(beta(1, 1), 3, 100).estimate(sys_disturbance, spec_threshold)
    assert hasattr(post, "mean")


def test_ch7_importance_sampling(sys_disturbance, traj_distribution, spec_threshold):
    np.random.seed(0)
    est = load("A7.3").ImportanceSamplingEstimation(traj_distribution, traj_distribution, 200)
    assert 0.0 <= est.estimate(sys_disturbance, spec_threshold) <= 1.0


def test_ch7_multiple_importance_sampling(sys_disturbance, traj_distribution, spec_threshold):
    a4 = load("A7.4")
    np.random.seed(0)
    mis = a4.MultipleImportanceSamplingEstimation(
        traj_distribution, [traj_distribution, traj_distribution], a4.smis)
    assert 0.0 <= mis.estimate(sys_disturbance, spec_threshold) <= 1.0


def test_ch7_bridge_functions():
    a8 = load("A7.8")
    g1 = lambda t: np.exp(-0.5 * (t - 0.0) ** 2)
    g2 = lambda t: np.exp(-0.5 * (t - 1.0) ** 2)
    np.random.seed(0)
    g1s, g2s = np.random.normal(0, 1, 200), np.random.normal(1, 1, 200)
    gb = a8.optimal_bridge(g1s, g1, g2s, g2, 10)
    assert np.isfinite(a8.bridge_sampling_estimator(g1s, g1, g2s, g2, gb))


def test_ch7_self_importance_sampling(sys_disturbance, traj_distribution, spec_threshold):
    a9, a3 = load("A7.9"), load("A7.3")
    p = traj_distribution
    samples = [a3.rollout(sys_disturbance, p) for _ in range(50)]
    est = a9.SelfImportanceSamplingEstimation(p, lambda t: a9.pdf(p, t), samples)
    assert np.isfinite(est.estimate(sys_disturbance, spec_threshold))
