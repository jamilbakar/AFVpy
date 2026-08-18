from collections import namedtuple

Step = namedtuple("Step", ["o", "a", "s_next"])         # Julia (; o, a, s′)
Transition = namedtuple("Transition", ["s", "o", "a"])  # Julia (; s, o, a)


def step(sys, s):
    o = sys.sensor.observe(s)
    a = sys.agent.act(o)
    s_next = sys.env.step(s, a)
    return Step(o, a, s_next)


def rollout(sys, d):
    s = sys.env.initial_distribution().rvs()
    tau = []
    for t in range(d):  # Julia 1:d
        o, a, s_next = step(sys, s)
        tau.append(Transition(s, o, a))
        s = s_next
    return tau