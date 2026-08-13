import math
import numpy as np
from collections import namedtuple

# Algorithm 11.5: counterfactual objective (eqs 11.5, 11.6, 11.8). Weighted sum of the
# outcome robustness, closeness to the original input (negative L1 distance), and
# plausibility (log-likelihood under the nominal trajectory distribution).
# Self-sufficient: includes the fixed-disturbance rollout (4.6) and logpdf (4.8).
# extract (system-specific) and robustness (STL) are external.

Disturbance = namedtuple("Disturbance", ["xa", "xs", "xo"])
Transition = namedtuple("Transition", ["s", "o", "a", "x"])
DisturbanceDistribution = namedtuple("DisturbanceDistribution", ["Da", "Ds", "Do"])


def extract(env, x):
    raise NotImplementedError  # system-specific


def robustness(states, formula):
    # stljax backend; signal shape [time, state_dim].
    import jax.numpy as jnp
    return float(formula.robustness(jnp.asarray(states)))


class NominalTrajectoryDistribution:
    def __init__(self, sys, d):
        self.D = DisturbanceDistribution(
            Da=lambda o: sys.agent.Da(o),
            Ds=lambda s, a: sys.env.Ds(s, a),
            Do=lambda s: sys.sensor.Do(s))
        self.Ps = sys.env.initial_distribution()
        self.d = d


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


def logpdf(p, tau):  # log of the trajectory-distribution pdf (4.8)
    logprob = p.Ps.logpdf(tau[0].s)
    for st in tau:
        s, o, a, x = st
        logprob += p.D.Da(o).logpdf(x.xa) + p.D.Ds(s, a).logpdf(x.xs) + p.D.Do(s).logpdf(x.xo)
    return logprob


def counterfactual_objective(x, sys, psi, x0, ws=None):
    if ws is None:
        ws = np.ones(3)
    s, x = extract(sys.env, x)
    tau = rollout(sys, s, x)
    foutcome = robustness([step.s for step in tau], psi.formula)
    fclose = -np.linalg.norm(np.asarray(x) - np.asarray(x0), 1)  # -norm(x - x0, 1)
    fplaus = logpdf(NominalTrajectoryDistribution(sys, len(x)), tau)
    return np.asarray(ws) @ np.array([foutcome, fclose, fplaus])