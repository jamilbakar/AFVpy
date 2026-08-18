from collections import namedtuple
Disturbance = namedtuple("Disturbance", ["xa", "xs", "xo"])
Transition = namedtuple("Transition", ["s", "o", "a", "x"])  # Julia (; s, o, a, x)


def step(sys, s, D):
    xo = D.Do(s).rvs()
    o = sys.sensor.observe(s, xo)
    xa = D.Da(o).rvs()
    a = sys.agent.act(o, xa)
    xs = D.Ds(s, a).rvs()
    s_next = sys.env.step(s, a, xs)
    return o, a, s_next, Disturbance(xa, xs, xo)


def initial_state_distribution(p):
    return p.Ps


def disturbance_distribution(p, t):
    return p.D


def depth(p):
    return p.d


def rollout(sys, p, d=None):
    if d is None:
        d = depth(p)
    s = initial_state_distribution(p).rvs()
    tau = []
    for t in range(d):
        o, a, s_next, x = step(sys, s, disturbance_distribution(p, t))
        tau.append(Transition(s, o, a, x))
        s = s_next
    return tau