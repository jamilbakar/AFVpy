from collections import namedtuple
Transition = namedtuple("Transition", ["s", "o", "a", "x"])


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
        x = x_traj[t]  # Julia x = 𝐱[t]
        o, a, s_next = step(sys, s, x)
        tau.append(Transition(s, o, a, x))
        s = s_next
    return tau