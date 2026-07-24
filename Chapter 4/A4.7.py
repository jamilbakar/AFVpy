from collections import namedtuple
Transition = namedtuple("Transition", ["s", "o", "a", "x"])


def extract(env, x):
    # System-specific: split the real vector x into (initial_state, disturbance_traj).
    raise NotImplementedError


def robustness(states, formula, w=0.0):
    # STL robustness (SignalTemporalLogic.jl); no Python built-in equivalent.
    raise NotImplementedError


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