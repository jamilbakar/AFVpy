# Chapter 9 — Reachability for Nonlinear Systems

Reachability when the dynamics are nonlinear, so the exact reachable set can't be
computed. Each method returns a sound *over-approximation* (it always contains the
true reachable set) using interval arithmetic and Taylor expansions.

### algorithm_9_1.py — Natural inclusion
The simplest over-approximation: run the ordinary rollout but with *intervals*
instead of numbers, so interval arithmetic propagates a box of possible states
forward. Cheap, but the box can grow loose ("dependency" over-conservatism).

### algorithm_9_2.py — Taylor inclusion
Tighter bounds via a first- or second-order Taylor expansion of the dynamics around
the interval's midpoint, with the remainder bounded over the interval using the
gradient (order 1) or the gradient plus Hessian (order 2).

### algorithm_9_3.py — Conservative linearization
Linearizes the dynamics at the midpoint (using the Jacobian) and captures the
nonlinear remainder as an interval "error box" from the Hessian, then propagates the
linear map plus that error box — often tighter than natural inclusion for mildly
nonlinear systems.

### algorithm_9_4.py — Concrete Taylor inclusion
Applies Taylor inclusion step by step, "concretizing" (collapsing to a fresh
hyperrectangle) after each step so error doesn't compound symbolically across the
whole horizon.

### algorithm_9_5.py — Concrete conservative linearization
The same concretize-each-step strategy applied to conservative linearization.

### Interval_ad.py — the backend
The engine the chapter needed but no Python library provides: `Interval` arithmetic
(with correct `sin`/`cos` that find extrema inside an interval), forward-mode
`gradient`/`jacobian`, and second-order `hessian` — all evaluated **over intervals**
— plus polymorphic `psin`/`pcos` that work on plain numbers, intervals, or dual
numbers. It replaces both IntervalArithmetic.jl and ForwardDiff.jl.

## Walls / problems

- **No Python library does interval arithmetic *and* interval-valued autodiff.** So `Interval_ad.py` was written from scratch and tested: `sin([0,π])=[0,1]`, `hessian(sin)` over `[0,π/2]` = `[-1,0]`, and 9.1/9.2/9.3 produce sound (overapproximating) reachable sets.
- **LazySets.jl avoided.** Sets are hyperrectangles, and `⊕`, `×`, `interval_hull`, and linear maps are done with interval arithmetic — no polytope library required.
- **The system must be interval/AD-compatible:** `step` must use `Interval_ad.psin`/`pcos` (not `np.sin`) so intervals and dual numbers flow through — the same requirement Julia has for ForwardDiff.
- **Source detail in 9.4:** the printed loop extracts from `I`, but (per the caption) it should extract the *new* state from `I'`; done that way here with a comment.
