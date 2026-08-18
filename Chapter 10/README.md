# Chapter 10 — Reachability for Discrete Systems

- **algorithm_10_1.py** — `to_graph` (build the weighted graph from the system).
- **algorithm_10_2.py** — `DiscreteForward` (forward reachable set).
- **algorithm_10_3.py** — `DiscreteBackward` (backward reachable set).
- **algorithm_10_4.py** — `ProbabilisticOccupancy` (distribution over states through time).
- **algorithm_10_5.py** — `ProbabilisticFiniteHorizon` (P(reach target within h)).
- **algorithm_10_6.py** — `ProbabilisticInfiniteHorizon` (P(eventually reach target)).

## Walls / problems

- **Graphs.jl reimplemented inline.** `WeightedGraph` / `add_edge!` / `outneighbors` / `inneighbors` / `to_matrix` are a small dict-based adjacency structure — no external graph library, so this whole chapter runs as-is.
- `Set(reduce(vcat, ...))` → set comprehension; the `R == (R ∪ S) && break` convergence check is preserved.
- `(I - TR) \ R1` → `np.linalg.solve`; target rows are zeroed (`TR[STi,:] = 0`) to make targets absorbing before the solve.
- Indices are 0-based throughout; graph *positions* are just node labels.
