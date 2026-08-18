# Chapter 3 — Property Specification

- **algorithm_3_1.py** — `LTLSpecification`: `evaluate` applies the formula to the full sequence of trajectory states.
- **algorithm_3_2.py** — `STLSpecification`: same, but only over a time interval `I`.

## Walls / problems

- **SignalTemporalLogic.jl formulas** have no drop-in Python equivalent; `formula` is a plug-in callable that takes the list of states and returns True/False (or, later, a robustness value via stljax).
- **Interval indexing.** Julia's `ψ.I` is a 1-based inclusive range (e.g. `3:10`); in Python it's a `slice`, so `3:10` → `slice(2, 10)`.
- Each spec is self-sufficient: it defines its own `Specification`/`evaluate`/`isfailure` rather than importing Chapter 1.
