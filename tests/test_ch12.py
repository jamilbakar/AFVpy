import numpy as np
from conftest import load

DATA = np.array([[0, 0.1, 0.2, 0.0, 10.0, 10.1, 9.9],
                 [0, 0.0, 0.1, 0.2, 10.0, 9.9, 10.1]])


def test_ch12_knn_monitor():
    m = load("algorithm_12_1").KNNMonitor(DATA, k=2, gamma=1.0)
    assert m.monitor([0.05, 0.05]) and not m.monitor([5.0, 5.0])


def test_ch12_hull_monitor():
    m = load("algorithm_12_2").HullMonitor(DATA, {0: [0, 1, 2, 3], 1: [4, 5, 6]})
    assert m.monitor([0.05, 0.05]) and not m.monitor([5.0, 5.0])


def test_ch12_superlevel_set_monitor():
    from scipy.stats import multivariate_normal
    m = load("algorithm_12_3").SuperlevelSetMonitor(multivariate_normal(mean=[0, 0], cov=np.eye(2)), gamma=0.05)
    assert m.monitor([0.0, 0.0]) and not m.monitor([4.0, 4.0])
