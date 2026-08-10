import numpy as np
from collections import namedtuple

# Algorithm 11.4: Shapley values of the disturbances in a trajectory. For each time
# step and m sampled permutations, it measures the marginal robustness change from
# adding that step's disturbance. Self-sufficient: includes the nominal
# disturbance-sampling rollout (4.5) and the fixed-disturbance step (4.6), plus the
# nominal trajectory distribution. robustness (STL) is external.

Disturbance = namedtuple("Disturbance", ["xa", "xs", "xo"])
Transition = namedtuple("Transition", ["s", "o", "a", "x"])
DisturbanceDistribution = namedtuple("DisturbanceDistribution", ["Da", "Ds", "Do"])


def robustness(states, formula):
    raise NotImplementedError  # STL robustness


class NominalTrajectoryDistribution:
    def __init__(self, sys, d):
        self.D = DisturbanceDistribution(
            Da=lambda o: sys.agent.Da(o),
            Ds=lambda s, a: sys.env.Ds(s, a),
            Do=lambda s: sys.sensor.Do(s))
        self.Ps = sys.env.initial_distribution()
        self.d = d


def step(sys, s, x):  # fixed-disturbance step (4.6)
    o = sys.sensor.observe(s, x.xo)
    a = sys.agent.act(o, x.xa)
    s_next = sys.env.step(s, a, x.xs)
    return o, a, s_next


def rollout(sys, p, d=None):  # nominal disturbance-sampling rollout (4.5)
    if d is None:
        d = p.d
    s = p.Ps.rvs()
    tau = []
    for t in range(d):
        xo = p.D.Do(s).rvs(); o = sys.sensor.observe(s, xo)
        xa = p.D.Da(o).rvs(); a = sys.agent.act(o, xa)
        xs = p.D.Ds(s, a).rvs(); s_next = sys.env.step(s, a, xs)
        tau.append(Transition(s, o, a, Disturbance(xa, xs, xo)))
        s = s_next
    return tau


def shapley_rollout(sys, s, x, w, inds):
    tau = []
    for t in range(len(x)):
        xt = x[t] if t in inds else w[t]  # inds is a set of 0-based indices
        o, a, s_next = step(sys, s, xt)
        tau.append(Transition(s, o, a, xt))
        s = s_next
    return tau


class Shapley:
    def __init__(self, tau, m):
        self.tau = tau  # current trajectory
        self.m = m      # samples per time step

    def describe(self, sys, psi):
        tau, m = self.tau, self.m
        p = NominalTrajectoryDistribution(sys, len(tau))
        x = [step.x for step in tau]
        phis = np.zeros(len(tau))
        for t in range(len(tau)):
            for _ in range(m):
                w = [st.x for st in rollout(sys, p)]
                perm = np.random.permutation(len(tau))
                j = int(np.where(perm == t)[0][0])           # findfirst(𝒫 .== t)
                tau_plus = shapley_rollout(sys, tau[0].s, x, w, set(perm[:j + 1]))   # 𝒫[1:j]
                tau_minus = shapley_rollout(sys, tau[0].s, x, w, set(perm[:j]))      # 𝒫[1:j-1]
                phis[t] += (robustness([st.s for st in tau_plus], psi.formula)
                            - robustness([st.s for st in tau_minus], psi.formula))
            phis[t] /= m
        return phis