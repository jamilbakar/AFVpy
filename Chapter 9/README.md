# Chapter 9 — Reachability for Nonlinear Systems

- **algorithm_9_1.py** — `NaturalInclusion` (propagate intervals through the rollout).
- **algorithm_9_2.py** — `TaylorInclusion` (first/second-order Taylor inclusion).
- **algorithm_9_3.py** — `ConservativeLinearization`.
- **algorithm_9_4.py** — `ConcreteTaylorInclusion` (concretize each step).
- **algorithm_9_5.py** — `ConcreteConservativeLinearization`.
- **Interval_ad.py** — the backend: `Interval` arithmetic (with `sin`/`cos`), forward-mode `gradient`/`jacobian`, and second-order `hessian`, all evaluated **over intervals**, plus polymorphic `psin`/`pcos`.

## Walls / problems

- **No Python library does interval arithmetic *and* interval-valued autodiff**, which this chapter needs. So `Interval_ad.py` was written from scratch (replaces IntervalArithmetic.jl + ForwardDiff.jl). It's tested: `sin([0,π])=[0,1]`, `hessian(sin)` over `[0,π/2]` = `[-1,0]`, and 9.1/9.2/9.3 produce sound (overapproximating) reachable sets.
- **LazySets.jl avoided.** Sets are represented as hyperrectangles, and `⊕`, `×`, `interval_hull`, and linear maps are done with interval arithmetic — no polytope library required.
- **The system must be interval/AD-compatible.** `step` must use `Interval_ad.psin`/`pcos` (not `np.sin`) so intervals and dual numbers flow through it — the same requirement Julia has for ForwardDiff.
- **Source detail in 9.4:** the printed loop extracts from `I`, but (per the caption) it should extract the *new* state from `I'`; done that way here with a comment.
