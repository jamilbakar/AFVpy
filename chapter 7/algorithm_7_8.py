import numpy as np

def bridge_sampling_estimator(g1taus, g1, g2taus, g2, gb):
    g1s = np.array([g1(t) for t in g1taus])
    g2s = np.array([g2(t) for t in g2taus])
    gb1s = np.array([gb(t) for t in g1taus])
    gb2s = np.array([gb(t) for t in g2taus])
    return np.mean(gb2s / g2s) / np.mean(gb1s / g1s)


def optimal_bridge(g1taus, g1, g2taus, g2, k_max):
    ratio = 1.0
    m1, m2 = len(g1taus), len(g2taus)

    def gb(tau):  # reads the latest `ratio` via closure (updated in the loop)
        return (g1(tau) * g2(tau)) / (m1 * g1(tau) + m2 * ratio * g2(tau))

    for k in range(k_max):  # Julia `for k in k_max` -> iterate k_max times
        ratio = bridge_sampling_estimator(g1taus, g1, g2taus, g2, gb)
    return gb
