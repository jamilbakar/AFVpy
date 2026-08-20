# Chapter 4 — Falsification Through Optimization

Finding a failure by treating the search over disturbances as an optimization
problem. The chapter first builds the machinery (disturbances, trajectory
distributions, rollouts, densities) and then the objective functions and the
falsification loop.

### algorithm_4_1.py — Direct Falsification
The baseline: draw `m` nominal rollouts to depth `d` and return the ones that
violate the specification. No cleverness — just sample and filter.

### algorithm_4_2.py — Disturbance and disturbance step
Defines a `Disturbance` (the agent/environment/sensor noise for one step) and a
`DisturbanceDistribution` (the distributions those noises come from). Its `step`
samples a disturbance and applies it through the sensor, agent, and environment,
returning the next state *and* the disturbance that produced it.

### algorithm_4_3.py — Trajectory distribution (abstract)
The interface for a distribution over whole trajectories: `initial_state_distribution`,
`disturbance_distribution(t)`, and `depth`. Concrete versions implement these.

### algorithm_4_4.py — Nominal trajectory distribution
Builds the "natural" trajectory distribution for a system from its component noise
distributions (`Da`/`Ds`/`Do`) and initial-state distribution. Because it's
stationary, the disturbance distribution is the same at every time step.

### algorithm_4_5.py — Rollout under a trajectory distribution
Rolls the system out by sampling the initial state and each step's disturbance from
a given trajectory distribution, returning the trajectory with the disturbances
recorded.

### algorithm_4_6.py — Rollout under a fixed disturbance trajectory
The deterministic counterpart: given a *specific* sequence of disturbances, replay
it through the system. This is what optimizers drive — they propose a disturbance
vector and this reconstructs the resulting trajectory.

### algorithm_4_7.py — Robustness objective
Turns a specification into a smooth objective. It extracts the initial state and
disturbances from the optimizer's vector, rolls out, and returns the temporal-logic
*robustness* of the trajectory (how strongly the spec is satisfied/violated).
Minimizing it drives the system toward a failure.

### algorithm_4_8.py — Trajectory density (pdf / logpdf)
Computes the likelihood of a trajectory under a trajectory distribution, in log
space for numerical stability: the log-likelihood of the initial state plus the
log-likelihood of every disturbance along the way.

### algorithm_4_9.py — Likelihood objective (most-likely failure)
An objective for finding the *most probable* failure: if the rolled-out trajectory
is a failure it returns the negative likelihood under the nominal distribution
(so the optimizer seeks likely failures); otherwise it returns the robustness (to
push it toward failing in the first place).

### algorithm_4_10.py — Weighted likelihood objective
Combines the two goals — trade off robustness against likelihood with a weight `λ`,
so you can look for failures that are both severe and plausible.

### algorithm_4_11.py — Optimization-based falsification
The driver: build a system-specific objective from a generic one and run an
optimizer on it, returning whatever the optimizer finds.

## Walls / problems

- **`extract` is system-specific** (splits the optimizer's flat vector into initial state + disturbances) — left as a stub you supply. Needed by 4.7, 4.9, 4.10.
- **`robustness` → stljax** (`formula.robustness(signal)`, smoothed via `logsumexp`+temperature when `smoothness>0`).
- Callable-struct/named-method mapping and named tuples as in Chapter 1.

---

*Python code created by Jamil Bakar.*
