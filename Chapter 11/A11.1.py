import numpy as np
from collections import namedtuple

# Algorithm 11.1: sampling-based sensitivity of robustness to disturbances. For each
# time step it perturbs the disturbance m times, re-rolls out, and takes the std of the
# absolute change in robustness. Self-sufficient: includes the fixed-disturbance
# rollout (4.6). extract (system-specific) and robustness (STL) are external; perturb
# is supplied on the struct. NOTE: Julia std uses N-1 (ddof=1).

Transition = namedtuple("Transition", ["s", "o", "a", "x"])


def extract(env, x):
    raise NotImplementedError  # system-specific: x -> (initial_state, disturbance_traj)


def robustness(states, formula):
    raise NotImplementedError  # STL robustness (SignalTemporalLogic.jl)


def step(sys, s, x):
    o = sys.sensor.observe(s, x.xo)
    a = sys.agent.act(o, x.xa)
    s_next = sys.env.step(s, a, x.xs)
    return o, a, s_next


def rollout(sys, s, x_traj, d=None):
    if d is None:
        d = len(x_traj)
    tau = []
    for t in range(d):
        x = x_traj[t]
        o, a, s_next = step(sys, s, x)
        tau.append(Transition(s, o, a, x))
        s = s_next
    return tau


class Sensitivity:
    def __init__(self, x, perturb, m):
        self.x = x              # vector of trajectory inputs
        self.perturb = perturb  # x' = perturb(x, t)
        self.m = m              # samples per time step

    def describe(self, sys, psi):
        m, x, perturb = self.m, self.x, self.perturb
        s, x = extract(sys.env, x)
        tau = rollout(sys, s, x)
        rho0 = robustness([step.s for step in tau], psi.formula)
        sensitivities = np.zeros(len(tau))
        for t in range(len(tau)):  # Julia eachindex(τ) is 1-based; perturb sees 0-based t
            xs = [perturb(x, t) for _ in range(m)]
            taus = [rollout(sys, *extract(sys.env, xp)) for xp in xs]
            rhos = [robustness([st.s for st in tp], psi.formula) for tp in taus]
            sensitivities[t] = np.std(np.abs(np.array(rhos) - rho0), ddof=1)
        return sensitivities