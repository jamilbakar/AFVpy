import numpy as np
from collections import namedtuple

# Algorithm 5.1: temporal-logic robustness objective for multiple shooting.
# Rolls out each segment, concatenates, and adds a defect penalty for the gap
# between consecutive segments. Self-sufficient: includes the fixed-disturbance
# step/rollout (4.6). extract (system-specific) and smooth_robustness (STL
# library) are external and must be supplied.

Transition = namedtuple("Transition", ["s", "o", "a", "x"])


def extract(env, x):
    raise NotImplementedError  # system-specific: x -> segments, each with .s and .x


def smooth_robustness(states, formula, w=0.0):
    # stljax backend; signal shape [time, state_dim]. w=0 gives exact robustness.
    import jax.numpy as jnp
    signal = jnp.asarray(states)
    if w and w > 0:
        return float(formula.robustness(signal, approx_method="logsumexp", temperature=w))
    return float(formula.robustness(signal))


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


def defect(taui, tauj):  # gap between end of one segment and start of the next
    return np.linalg.norm(tauj[0].s - taui[-1].s)


def shooting_robustness(x, sys, psi, smoothness=0.0, lam=1.0):
    segments = extract(sys.env, x)
    n = len(segments)
    tau_segments = [rollout(sys, seg.s, seg.x) for seg in segments]
    tau = [tr for seg in tau_segments for tr in seg]  # vcat
    states = [step.s for step in tau]
    rho = smooth_robustness(states, psi.formula, w=smoothness)
    defects = [defect(tau_segments[i], tau_segments[i + 1]) for i in range(n - 1)]
    return rho + lam * sum(defects)