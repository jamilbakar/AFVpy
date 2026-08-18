# Chapter 1 — Systems, Rollouts, Specifications

- **algorithm_1_1.py** — abstract `Agent`/`Environment`/`Sensor` types and the `System` container.
- **algorithm_1_2.py** — `step` (sensor → agent → environment) and `rollout` (repeat step to depth d, collecting a trajectory).
- **algorithm_1_3.py** — `Specification` base, `evaluate`, and `isfailure = not evaluate`.

## Walls / problems

- **Callable structs → named methods.** Julia calls objects directly (`sys.sensor(s)`); Python maps these to `.observe` / `.act` / `.step`. `rand(Ps(env))` → `env.initial_distribution().rvs()`.
- **Named tuples** `(; o, a, s′)` → Python `namedtuple` (with `s_next` since `s′` isn't a legal name).
- **Dispatch for `evaluate`.** Julia dispatches `evaluate` on the specification type. Python uses `functools.singledispatch` so each concrete spec registers its own `evaluate`.
- **1-based loop** `1:d` → `range(d)`.
