# Chapter 11 — Failure Analysis

Once a failure is found, these algorithms explain *why* it happens — which
disturbances matter, how to change the input to avoid it, and how failures group
together.

### algorithm_11_1.py — Sensitivity (sampling-based)
Estimates how sensitive the trajectory's robustness is to each time step's
disturbance: perturb the disturbance at a step `m` times, re-roll out, and take the
standard deviation of the resulting change in robustness. Big values flag the steps
that matter most.

### algorithm_11_2.py — Gradient sensitivity
The same idea via calculus: returns the gradient of the robustness with respect to
the input disturbances, computed by automatic differentiation — one exact
sensitivity vector instead of a sampled estimate.

### algorithm_11_3.py — Integrated gradients
A cleaner attribution: average the robustness gradient along the straight path from a
baseline input to the actual input. This distributes "credit" for the outcome across
the inputs more faithfully than a single gradient at one point.

### algorithm_11_4.py — Shapley values
Attributes the failure to each disturbance fairly using Shapley values from
cooperative game theory: over many random orderings, measure how much adding each
step's disturbance changes the robustness, and average. Returns one importance value
per time step.

### algorithm_11_5.py — Counterfactual objective
Scores a *counterfactual* input — one that would have avoided failure — by a weighted
sum of three terms: the outcome robustness, closeness to the original input (small
L1 change), and plausibility (log-likelihood under the nominal distribution). An
optimizer over this finds the smallest, most plausible change that flips the outcome.

### algorithm_11_6.py — K-means clustering
Groups a set of failure trajectories: extract a feature vector from each, then run
k-means (assign to nearest centroid, update centroids) to discover distinct
*failure modes*. Returns the clusters and their centroids.

## Walls / problems

- **ForwardDiff.jl → `jax.grad`** (11.2, 11.3). Because stljax is jax-based, `jax.grad` differentiates through the robustness — removing the autodiff wall. The system's `step`/`extract` must use `jax.numpy` so the computation is traceable (same requirement Julia's ForwardDiff has).
- **`robustness` → stljax** (11.1, 11.4, 11.5).
- **System-specific stubs:** `extract` (11.1, 11.2, 11.3, 11.5) and `perturb` (11.1).
- **k-means (11.6) is fully runnable** with no extra dependencies (you supply the feature extractor `phi` and distance `d`).
- Faithful details: `std` → `np.std(ddof=1)`; permutation slices `𝒫[1:j]`/`𝒫[1:j-1]` → `perm[:j+1]`/`perm[:j]`; `mean(hcat(grads...), dims=2)` → `np.mean(np.column_stack(grads), axis=1)`.
