# Chapter 8 — Reachability for Linear Systems

- **algorithm_8_1.py** — `AvoidSetSpecification` (trajectory avoids a set).
- **algorithm_8_2.py** — one-step reachable set (linear map + Minkowski sum).
- **algorithm_8_3.py** — `SetPropagation` forward reachability.
- **algorithm_8_4.py** — `satisfies` via set propagation (intersect reachable set with avoid set).
- **algorithm_8_5.py** — `OverapproximateSetPropagation`.
- **algorithm_8_6.py** — support function of a reachable set (`constrained_model` + `ρ`).
- **algorithm_8_7.py** — `LinearProgramming` forward reachability.
- **algorithm_8_8.py** — `satisfies` via convex programming.

## Walls / problems

- **JuMP.jl → cvxpy.** `@variable`/`@constraint`/`@objective`/`optimize!` map to `cp.Variable` / a constraints list / `cp.Problem(...).solve()`. Used by 8.6 and 8.8 (fully runnable).
- **LazySets.jl → pycvxset** (8.2–8.5, 8.7). `Ab(P)` → `(P.A, P.b)`; `HPolytope`/`HalfSpace` → `Polytope(A=, b=)`; `∩`→`.intersection`, `isempty`→`.is_empty`, `⊕`→`+`, linear map→`@`; `overapproximate` via support functions; `union` → a `UnionSetArray` list (a union of convex sets isn't convex, so the pieces are kept, as LazySets does). Install: `brew install gmp cddlib` then pip pycddlib + pycvxset.
- **Source detail in 8.6:** the book's `ρ(model, d, d)` reuses `d` for the direction and the depth (bold vs plain); split here into `rho(model, direction, depth)`.
