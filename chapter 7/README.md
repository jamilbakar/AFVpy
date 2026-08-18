# Chapter 7 — Failure Probability Estimation

- **algorithm_7_1.py** — `DirectEstimation` (mean failure over m rollouts).
- **algorithm_7_2.py** — `BayesianEstimation` (Beta posterior).
- **algorithm_7_3.py** — `ImportanceSamplingEstimation`.
- **algorithm_7_4.py** — `MultipleImportanceSamplingEstimation` + `smis`/`dmmis` weighting.
- **algorithm_7_5.py** — `CrossEntropyEstimation`.
- **algorithm_7_6.py** — `PopulationMonteCarloEstimation`.
- **algorithm_7_7.py** — `SequentialMonteCarloEstimation`.
- **algorithm_7_8.py** — `bridge_sampling_estimator` + `optimal_bridge`.
- **algorithm_7_9.py** — `SelfImportanceSamplingEstimation`.
- **algorithm_7_10.py** — `BridgeSamplingEstimation`.
- **algorithm_7_11.py** — `AdaptiveMultilevelSplitting`.

## Walls / problems

- **System-specific stubs:** `fit` (7.5), `proposal` (7.6), `perturb` (7.7, 7.10, 7.11) — supply these to run those algorithms.
- **`Beta` → `scipy.stats.beta`** (prior params read via `.args`).
- **Source oddity in 7.9:** the code uses `mean(...)` on already-normalized weights; self-normalized IS (eq 7.33) is usually a weighted *sum* — reproduced as `np.mean` with a comment (swap to `np.sum` to match the equation).
- **`for k in k_max` in 7.8** (iterating an integer, not valid Julia as printed) translated as `range(k_max)`; `gb` closure captures the loop-updated `ratio` by late binding.
- `rand(Categorical(w/sum w), m)` → `np.random.choice(..., size=m, p=...)`.
