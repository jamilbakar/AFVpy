# Chapter 11 — Failure Analysis

- **algorithm_11_1.py** — `Sensitivity` (sampling-based: std of robustness change per step).
- **algorithm_11_2.py** — `GradientSensitivity` (gradient of robustness).
- **algorithm_11_3.py** — `IntegratedGradients` (averaged gradient along a baseline→input path).
- **algorithm_11_4.py** — `Shapley` (Shapley values of the disturbances).
- **algorithm_11_5.py** — `counterfactual_objective` (weighted outcome / closeness / plausibility).
- **algorithm_11_6.py** — `Kmeans` (cluster failure trajectories).

## Walls / problems

- **ForwardDiff.jl → `jax.grad`** (11.2, 11.3). Because stljax is jax-based, `jax.grad` differentiates through the robustness — this removes the autodiff wall. The system's `step`/`extract` must be written with `jax.numpy` so the computation is traceable (same requirement Julia's ForwardDiff has).
- **`robustness` → stljax** (11.1, 11.4, 11.5).
- **System-specific stubs:** `extract` (11.1, 11.2, 11.3, 11.5) and `perturb` (11.1).
- **k-means (11.6) is fully runnable** with no extra dependencies (you supply the feature extractor `phi` and distance `d`).
- Faithful details: `std` → `np.std(ddof=1)`; permutation slices `𝒫[1:j]`/`𝒫[1:j-1]` → `perm[:j+1]`/`perm[:j]`; `mean(hcat(grads...), dims=2)` → `np.mean(np.column_stack(grads), axis=1)`.
