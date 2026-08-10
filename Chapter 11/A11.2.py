import numpy as np
from collections import namedtuple

# Algorithm 11.2: gradient-based sensitivity. Returns the gradient of the robustness
# with respect to the input features. Self-sufficient: includes the fixed-disturbance
# rollout (4.6). WALL: ForwardDiff.gradient (autodiff) has no drop-in Python equivalent
# and is stubbed; extract (system-specific) and robustness (STL) are external.

Transition = namedtuple("Transition", ["s", "o", "a", "x"])


def extract(env, x):
    raise NotImplementedError  # system-specific


def robustness(states, formula):
    raise NotImplementedError  # STL robustness


def gradient(f, x):
    raise NotImplementedError  # ForwardDiff.gradient (use autograd/JAX or finite diff)


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


class GradientSensitivity:
    def __init__(self, x):
        self.x = x  # vector of trajectory inputs

    def describe(self, sys, psi):
        def current_robustness(x):
            s, x = extract(sys.env, x)
            tau = rollout(sys, s, x)
            return robustness([step.s for step in tau], psi.formula)

        return gradient(current_robustness, self.x)