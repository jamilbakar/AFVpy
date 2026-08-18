# Chapter 5 — Falsification Through Planning

- **algorithm_5_1.py** — `shooting_robustness` (multiple-shooting objective with a defect penalty).
- **algorithm_5_2.py** — generic `TreeSearch` + `falsify` (with `trajectory`/`failures` bundled in).
- **algorithm_5_3.py** — `trajectory` + `failures` (extract failure paths from a tree).
- **algorithm_5_4.py** — `RRT` + `RRTNode`.
- **algorithm_5_5.py** — RRT helpers: `random_goal`, `distance_objectives`, `random_disturbance`.
- **algorithm_5_6.py** — `goal_disturbance`.
- **algorithm_5_7.py** — `average_dispersion`.
- **algorithm_5_8.py** — `star_discrepancy`.
- **algorithm_5_9.py** — `distance_c`, `distance_h`, `cost_objectives`.
- **algorithm_5_10.py** — `MCTS` + `MCTSNode`.
- **algorithm_5_11.py** — `lcb` (lower confidence bound).

## Walls / problems

- **LazySets.jl in 5.8** (Hyperrectangle volume/membership) reimplemented inline with numpy — no polytope library needed.
- **`extract`/`smooth_robustness` (5.1)** and **`perturb` (5.6)** are system-specific stubs; `smooth_robustness` uses stljax.
- **Source inconsistency in 5.9:** the book calls `h(sgoal, node)` although `distance_h` is `(node, sgoal)`. Reproduced as-is with a comment — swap the arg order if using the default `distance_h`.
- `argmin`/`norm`/`sortperm` → `np.argmin`/`np.linalg.norm`/`np.argsort`; `pushfirst!` → `list.insert(0, ...)`.
