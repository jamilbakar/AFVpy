# Chapter 5 — Falsification Through Planning

Finding failures with tree search and planning instead of black-box optimization,
plus the coverage metrics used to judge how well the search explores.

### algorithm_5_1.py — Shooting robustness objective
A "multiple shooting" objective: the trajectory is split into segments, each rolled
out independently, and the objective is the overall robustness *plus* a penalty for
the "defect" (the gap between where one segment ends and the next begins). This lets
an optimizer work on segments in parallel while keeping them stitched together.

### algorithm_5_2.py — Tree search (generic)
The generic falsification loop shared by RRT and MCTS: initialize a tree, then
repeatedly `select` a node and `extend!` it for `k_max` iterations, and finally
return the failure trajectories in the tree. `trajectory`/`failures` are bundled in.

### algorithm_5_3.py — Extract failure trajectories
`trajectory` walks from a leaf back up to the root to reconstruct a full path;
`failures` finds all leaves, builds their trajectories, and keeps the failing ones.

### algorithm_5_4.py — RRT (Rapidly-exploring Random Trees)
A tree-search strategy. `select` samples a goal state, scores every node by a
distance-to-goal objective, and picks the closest; `extend!` applies a disturbance
from that node and adds the resulting state as a new child, growing the tree toward
unexplored regions.

### algorithm_5_5.py — RRT helpers
`random_goal` samples a goal uniformly from the state space, `distance_objectives`
scores tree nodes by Euclidean distance to the goal, and `random_disturbance` draws
one disturbance from the nominal distribution — the plug-ins RRT uses.

### algorithm_5_6.py — Goal-directed disturbance
A smarter disturbance selector: sample several candidate disturbances, simulate each
one step, and keep the one whose next state lands closest to the node's goal.

### algorithm_5_7.py — Average dispersion
A coverage metric: normalize the sampled points to the unit cube, lay down a grid,
and average how far each grid cell is from its nearest sample (capped at the grid
spacing). Lower means the samples cover the space more evenly.

### algorithm_5_8.py — Star discrepancy
Another coverage metric: bounds how far the fraction of points inside sub-boxes
deviates from those boxes' volumes, returning lower and upper bounds on the star
discrepancy. Lower discrepancy means more uniform coverage.

### algorithm_5_9.py — Cost objectives (for RRT*-style search)
Traverses the tree accumulating each node's path cost, then adds a heuristic
(distance to goal) to get an objective per node — encouraging the search toward
shorter paths.

### algorithm_5_10.py — MCTS (Monte Carlo Tree Search)
A tree search with progressive widening: `select` descends using the lower
confidence bound until a node is eligible to expand; `extend!` adds a child,
estimates its value, and backs that value up the path, updating visit counts and
value estimates.

### algorithm_5_11.py — Lower confidence bound
The selection rule used by MCTS: compute each child's value minus an
exploration bonus and return the child with the lowest LCB (favoring low-value,
i.e. failure-prone, branches while still exploring).

## Walls / problems

- **LazySets.jl in 5.8** (Hyperrectangle volume/membership) reimplemented inline with numpy — no polytope library needed.
- **`extract`/`smooth_robustness` (5.1)** and **`perturb` (5.6)** are system-specific stubs; `smooth_robustness` uses stljax.
- **Source inconsistency in 5.9:** the book calls `h(sgoal, node)` although `distance_h` is `(node, sgoal)`. Reproduced as-is with a comment — swap the arg order if using the default `distance_h`.
- `argmin`/`norm`/`sortperm` → numpy; `pushfirst!` → `list.insert(0, ...)`.
