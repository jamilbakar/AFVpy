import numpy as np
from conftest import load


def test_ch6_rejection_sampling(sys_disturbance, traj_distribution):
    a1 = load("A6.1"); q = traj_distribution
    np.random.seed(0)
    rs = a1.RejectionSampling(p_bar=lambda tau: a1.pdf(q, tau), q=q, c=1.0, k_max=30)
    assert len(rs.sample_failures(sys_disturbance, None)) == 30


def test_ch6_mcmc_sampling(sys_disturbance, traj_distribution):
    a2 = load("A6.2"); q = traj_distribution
    np.random.seed(0)
    mc = a2.MCMCSampling(p_bar=lambda tau: a2.pdf(q, tau), g=lambda tau: q,
                         tau=a2.rollout(sys_disturbance, q), k_max=20, m_burnin=5, m_skip=2)
    assert len(mc.sample_failures(sys_disturbance, None)) == 8
