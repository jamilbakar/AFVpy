from collections import namedtuple

# Algorithm 4.7: temporal-logic robustness objective. Extracts an initial state and
# disturbance trajectory from the real-valued vector x, rolls out, and returns the
# smoothed robustness of the resulting state trajectory (robustness if smoothness=0).
# Self-sufficient: includes the fixed-disturbance step/rollout (algorithm 4.6).
# extract (system-specific) and robustness (from an STL library) are external and
# must be provided; the stubs below raise until you supply real ones.

Transition = namedtuple("Transition", ["s", "o", "a", "x"])


def extract(env, x):
    # System-specific: split the real vector x into (initial_state, disturbance_traj).
    raise NotImplementedError


def robustness(states, formula, w=0.0):
    # stljax backend. formula is an stljax formula; signal shape [time, state_dim].
    # w > 0 selects the smoothed robustness (for gradient-based optimization).
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


def robustness_objective(x, sys, psi, smoothness=0.0):
    s, x_traj = extract(sys.env, x)
    tau = rollout(sys, s, x_traj)
    states = [step.s for step in tau]
    return robustness(states, psi.formula, w=smoothness)