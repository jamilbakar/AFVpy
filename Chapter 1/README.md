# Chapter 1 — Systems, Rollouts, Specifications

The foundation the whole book builds on: how a system is represented, how it is
simulated, and how we decide whether a run counts as a failure.

### algorithm_1_1.py — System definition
Defines the core abstraction. A system has three parts — an `Agent` (chooses
actions), an `Environment` (evolves the state), and a `Sensor` (produces
observations) — held together by the `System` container. The `Agent`,
`Environment`, and `Sensor` base classes are just roles; concrete systems (see the
Appendix) fill them in.

### algorithm_1_2.py — step and rollout
`step` advances the system one tick: the sensor turns the true state into an
observation, the agent turns that into an action, and the environment produces the
next state, returning `(o, a, s_next)`. `rollout` samples an initial state from the
environment and calls `step` repeatedly to depth `d`, collecting the sequence of
`(state, observation, action)` tuples — the trajectory that every later algorithm
analyzes.

### algorithm_1_3.py — Specification
Defines what "failure" means. A `Specification` is evaluated on a trajectory:
`evaluate` returns True if the trajectory satisfies the property, and `isfailure`
is simply its negation. Concrete specifications (Chapter 3) register their own
`evaluate` for their type.

## Walls / problems

- **Callable structs → named methods.** Julia calls objects directly (`sys.sensor(s)`); Python maps these to `.observe` / `.act` / `.step`, and `rand(Ps(env))` → `env.initial_distribution().rvs()`.
- **Named tuples** `(; o, a, s′)` → Python `namedtuple` (`s_next`, since `s′` isn't a legal name).
- **Dispatch for `evaluate`.** Julia dispatches `evaluate` on the specification type; Python uses `functools.singledispatch` so each concrete spec registers its own.
- **1-based loop** `1:d` → `range(d)`.
