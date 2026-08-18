# Chapter 4 — Falsification Through Optimization

- **algorithm_4_1.py** — `DirectFalsification`: sample m rollouts, keep the failures.
- **algorithm_4_2.py** — `Disturbance`, `DisturbanceDistribution`, and a disturbance-sampling `step`.
- **algorithm_4_3.py** — abstract `TrajectoryDistribution` + accessors.
- **algorithm_4_4.py** — `NominalTrajectoryDistribution` built from the component distributions.
- **algorithm_4_5.py** — `rollout` over a trajectory distribution.
- **algorithm_4_6.py** — deterministic `step`/`rollout` that replay a fixed disturbance trajectory.
- **algorithm_4_7.py** — `robustness_objective` (temporal-logic robustness of a rollout).
- **algorithm_4_8.py** — `logpdf`/`pdf` of a trajectory distribution (log space).
- **algorithm_4_9.py** — `likelihood_objective` (most-likely-failure).
- **algorithm_4_10.py** — `weighted_likelihood_objective`.
- **algorithm_4_11.py** — `OptimizationBasedFalsification`.

## Walls / problems

- **`extract` is system-specific** (splits the optimizer's flat vector into initial state + disturbances) — left as a stub you supply. Needed by 4.7, 4.9, 4.10.
- **`robustness` → stljax.** The STL robustness is computed with `formula.robustness(signal)` (smoothed via `logsumexp`+temperature when `smoothness>0`).
- Callable-struct / named-method mapping and named tuples as in Chapter 1.
