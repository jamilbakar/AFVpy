# Chapter 3 — Property Specification

How safety properties are written in temporal logic and checked against a
trajectory. Both files subclass the Chapter 1 `Specification` interface.

### algorithm_3_1.py — LTL Specification
A Linear Temporal Logic specification. It holds a `formula` (a callable that takes
the sequence of states and returns True/False, e.g. "the state is always above a
threshold"); `evaluate` pulls the states out of the trajectory and applies the
formula over the whole run.

### algorithm_3_2.py — STL Specification
A Signal Temporal Logic specification, which adds a *time interval*. It works like
the LTL version but only applies the formula to the states within a specified time
window `I` (e.g. "between steps 40 and 41 the altitude must exceed 50 m"), which is
what the aircraft example in later chapters needs.

## Walls / problems

- **SignalTemporalLogic.jl formulas** have no drop-in Python equivalent; `formula` is a plug-in callable (later, a real robustness value via stljax).
- **Interval indexing.** Julia's `ψ.I` is a 1-based inclusive range (e.g. `3:10`); in Python it's a `slice`, so `3:10` → `slice(2, 10)`.
- Each spec is self-sufficient: it defines its own `Specification`/`evaluate`/`isfailure` rather than importing Chapter 1.

---

*Python code created by Jamil Bakar.*
