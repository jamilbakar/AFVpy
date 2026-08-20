# Chapter 8 — Reachability for Linear Systems

A guarantee-oriented alternative to sampling: instead of looking for one failure,
compute the *entire set* of states a linear system can reach and check whether it
ever touches an unsafe region. Two representations are used — explicit sets (set
propagation) and support functions via linear programming.

### algorithm_8_1.py — Avoid-set specification
The property for this chapter: a trajectory is safe as long as none of its states
land in a given avoid set. `evaluate` checks that every state stays out of the set.

### algorithm_8_2.py — One-step reachable set
Computes the set of states reachable in a single step of a linear system, using a
linear map of the current set plus a Minkowski sum of the disturbance sets — the
building block for the propagation methods.

### algorithm_8_3.py — Set-propagation reachability
Repeatedly applies the one-step operator and unions the results to build the
reachable set over the whole horizon.

### algorithm_8_4.py — Satisfies (set propagation)
Uses set propagation to compute the reachable set, then reports failure iff that set
intersects the avoid set (i.e. some reachable state is unsafe).

### algorithm_8_5.py — Overapproximate set propagation
Same as 8.3 but periodically replaces the growing set with a simpler
over-approximation, trading a little conservatism for much cheaper computation over
long horizons.

### algorithm_8_6.py — Support function via LP
Represents reachable sets implicitly. `constrained_model` builds an optimization
model of all trajectories consistent with the dynamics and disturbance bounds, and
`ρ` (the support function) maximizes the state in a given direction — one linear
program gives the set's extent along that direction.

### algorithm_8_7.py — Linear-programming reachability
Builds the reachable set as a polytope by evaluating the support function (8.6) in
many directions and intersecting the resulting half-spaces, unioned over the horizon.

### algorithm_8_8.py — Satisfies (convex programming)
Checks reachability of a convex avoid set directly: for each depth it minimizes the
squared distance between the reachable set and the avoid set; a zero distance (within
tolerance) means the system can reach the unsafe set, so the spec is not satisfied.

## Walls / problems

- **JuMP.jl → cvxpy.** `@variable`/`@constraint`/`@objective`/`optimize!` map to `cp.Variable` / a constraints list / `cp.Problem(...).solve()`. Used by 8.6 and 8.8 (fully runnable).
- **LazySets.jl → pycvxset** (8.2–8.5, 8.7). `Ab(P)` → `(P.A, P.b)`; `HPolytope`/`HalfSpace` → `Polytope(A=, b=)`; `∩`→`.intersection`, `isempty`→`.is_empty`, `⊕`→`+`, linear map→`@`; `overapproximate` via support functions; `union` → a `UnionSetArray` list (a union of convex sets isn't convex, so the pieces are kept, as LazySets does). Install: `brew install gmp cddlib`, then pip pycddlib + pycvxset.
- **Source detail in 8.6:** the book's `ρ(model, d, d)` reuses `d` for the direction and the depth (bold vs plain); split here into `rho(model, direction, depth)`.

---

*Python code created by Jamil Bakar.*
