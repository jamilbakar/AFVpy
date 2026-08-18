# Chapter 10 — Reachability for Discrete Systems

When states, actions, and disturbances are all discrete, the system is a directed
weighted graph and reachability becomes exact graph search / linear algebra — no
approximation needed.

### algorithm_10_1.py — Convert system to graph
`to_graph` turns a discrete system into a weighted directed graph: one node per
state, and an edge for every possible transition with its probability as the weight.
Everything else in the chapter operates on this graph.

### algorithm_10_2.py — Discrete forward reachability
Starting from the initial states, repeatedly add all out-neighbors until the
reachable set stops growing — the set of states the system can ever reach.

### algorithm_10_3.py — Discrete backward reachability
The mirror image: starting from a target set, repeatedly add all *in*-neighbors to
find every state from which the target can be reached.

### algorithm_10_4.py — Probabilistic occupancy
Starts from the initial-state distribution and pushes it forward through the graph
step by step, giving the probability of occupying each state at each time — the
distribution over where the system is over time.

### algorithm_10_5.py — Finite-horizon probability
Computes the probability of reaching a target set within `h` steps by
back-propagating reach probabilities from the target through the graph, then
weighting by the initial distribution.

### algorithm_10_6.py — Infinite-horizon probability
Computes the probability of *eventually* reaching the target: it makes the target
states absorbing and solves the linear system `(I − TR) R∞ = R1` for the reach
probability from every state, then weights by the initial distribution.

## Walls / problems

- **Graphs.jl reimplemented inline.** `WeightedGraph` / `add_edge!` / `outneighbors` / `inneighbors` / `to_matrix` are a small dict-based adjacency structure — no external graph library, so this whole chapter runs as-is.
- `Set(reduce(vcat, ...))` → set comprehension; the `R == (R ∪ S) && break` convergence check is preserved.
- `(I − TR) \ R1` → `np.linalg.solve`; target rows are zeroed (`TR[STi,:] = 0`) to make targets absorbing before the solve.
- Indices are 0-based throughout.
