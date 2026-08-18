# Chapter 6 — Falsification Through Sampling

- **algorithm_6_1.py** — `RejectionSampling`: accept a rollout with probability `p_bar(τ) / (c·pdf(q,τ))`.
- **algorithm_6_2.py** — `MCMCSampling`: Metropolis-Hastings with burn-in and thinning.
- **algorithm_6_3.py** — `ProbabilisticProgramming`: the Turing.jl failure-sampling model.

## Walls / problems

- **1-based inclusive slice.** Julia's `τs[m_burnin:m_skip:end]` → Python `taus[m_burnin-1::m_skip]`.
- **Turing.jl (6.3) has no drop-in Python equivalent.** The `@model` with latent `~` disturbances and `@addlogprob!` is rewritten as a `log_density(s0, xo, xa, xs)` (each `~` adds a `logpdf`; `@addlogprob!` becomes the smoothed indicator term), and `Turing.sample(...)` becomes a pluggable `sampler(log_density, sys, d, k_max)`.
- 6.1 and 6.2 are direct translations (rollout + pdf carry over exactly).
