from collections import namedtuple

Transition = namedtuple("Transition", ["s", "o", "a"])


def step(sys, s):
    o = sys.sensor.observe(s)
    a = sys.agent.act(o)
    s_next = sys.env.step(s, a)
    return o, a, s_next


def rollout(sys, d):
    s = sys.env.initial_distribution().rvs()
    tau = []
    for t in range(d):
        o, a, s_next = step(sys, s)
        tau.append(Transition(s, o, a))
        s = s_next
    return tau


def isfailure(psi, tau):
    return not psi.evaluate(tau)


class DirectFalsification:
    def __init__(self, d, m):
        self.d = d  # depth
        self.m = m  # number of samples

    def falsify(self, sys, psi):
        taus = [rollout(sys, self.d) for _ in range(self.m)]
        return [tau for tau in taus if isfailure(psi, tau)]