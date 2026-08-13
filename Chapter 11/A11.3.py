import numpy as np
from collections import namedtuple

# Algorithm 11.3: integrated gradients. Averages the robustness gradient along the path
# from a baseline b to the input x. Self-sufficient: includes the fixed-disturbance
# rollout (4.6). WALL: ForwardDiff.gradient is stubbed; extract (system-specific) and
# robustness (STL) are external.

Transition = namedtuple("Transition", ["s", "o", "a", "x"])


def extract(env, x):
    raise NotImplementedError  # system-specific


def robustness(states, formula):
    # stljax backend; signal shape [time, state_dim]. Use jnp so jax.grad can trace it.
    import jax.numpy as jnp
    return formula.robustness(jnp.asarray(states))


def gradient(f, x):
    # ForwardDiff.gradient -> jax.grad. The system (rollout/extract) must be written
    # with jax.numpy so the computation is differentiable.
    import jax
    import jax.numpy as jnp
    return np.asarray(jax.grad(f)(jnp.asarray(x, dtype=float)))


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


class IntegratedGradients:
    def __init__(self, x, b, m):
        self.x = x  # vector of trajectory inputs
        self.b = b  # vector of baseline inputs
        self.m = m  # steps for numerical integration

    def describe(self, sys, psi):
        def current_robustness(x):
            s, x = extract(sys.env, x)
            tau = rollout(sys, s, x)
            return robustness([step.s for step in tau], psi.formula)

        alphas = np.linspace(0, 1, self.m)
        xs = [(1 - a) * np.asarray(self.b) + a * np.asarray(self.x) for a in alphas]
        grads = [gradient(current_robustness, x) for x in xs]
        return np.mean(np.column_stack(grads), axis=1)  # mean(hcat(grads...), dims=2)