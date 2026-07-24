import math
from collections import namedtuple
Disturbance = namedtuple("Disturbance", ["xa", "xs", "xo"])
Transition = namedtuple("Transition", ["s", "o", "a", "x"])
DisturbanceDistribution = namedtuple("DisturbanceDistribution", ["Da", "Ds", "Do"])


def extract(env, x):
    raise NotImplementedError  # system-specific: x -> (initial_state, disturbance_traj)


def robustness(states, formula, w=0.0):
    raise NotImplementedError  # STL robustness (SignalTemporalLogic.jl)


def isfailure(psi, tau):
    return not psi.evaluate(tau)


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


class NominalTrajectoryDistribution:
    def __init__(self, sys, d):
        self.D = DisturbanceDistribution(
            Da=lambda o: sys.agent.Da(o),
            Ds=lambda s, a: sys.env.Ds(s, a),
            Do=lambda s: sys.sensor.Do(s))
        self.Ps = sys.env.initial_distribution()
        self.d = d


def pdf(p, tau):
    logprob = p.Ps.logpdf(tau[0].s)
    for t, step in enumerate(tau):
        s, o, a, x = step
        logprob += (p.D.Da(o).logpdf(x.xa)
                    + p.D.Ds(s, a).logpdf(x.xs)
                    + p.D.Do(s).logpdf(x.xo))
    return math.exp(logprob)


def likelihood_objective(x, sys, psi, smoothness=0.0):
    s, x_traj = extract(sys.env, x)
    tau = rollout(sys, s, x_traj)
    if isfailure(psi, tau):
        p = NominalTrajectoryDistribution(sys, len(x_traj))
        return -pdf(p, tau)
    states = [step.s for step in tau]
    return robustness(states, psi.formula, w=smoothness)