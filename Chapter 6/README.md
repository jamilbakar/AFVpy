# Chapter 6 — Falsification Through Sampling

Finding failures by sampling trajectories from cleverly chosen distributions rather
than optimizing or planning.

### algorithm_6_1.py — Rejection sampling
Draw a trajectory from a proposal distribution `q`, then accept it with probability
`p_bar(τ) / (c·pdf(q,τ))`, where `p_bar` is the target (failure) density and `c` is
a constant bounding it. Accepted samples are distributed according to the target.
Simple, but wasteful if the acceptance rate is low.

### algorithm_6_2.py — MCMC sampling (Metropolis-Hastings)
Build a chain of trajectories: at each step propose a new one via a kernel `g`,
accept or reject it by the Metropolis-Hastings ratio, and record the current
sample. Finally discard `m_burnin` warm-up samples and keep every `m_skip`-th of the
rest (thinning). Handles low-probability failure regions far better than rejection
sampling.

### algorithm_6_3.py — Probabilistic programming
Expresses the failure-sampling problem as a probabilistic program: the rollout is a
model where each disturbance is a latent variable, plus an added log-probability
term (a smoothed indicator) that biases sampling toward failure trajectories. An
MCMC algorithm then draws samples from it.

## Walls / problems

- **1-based inclusive slice.** Julia's `τs[m_burnin:m_skip:end]` → Python `taus[m_burnin-1::m_skip]`.
- **Turing.jl (6.3) has no drop-in Python equivalent.** The `@model` with latent `~` disturbances and `@addlogprob!` is rewritten as a `log_density(s0, xo, xa, xs)` (each `~` adds a `logpdf`; `@addlogprob!` becomes the smoothed indicator term), and `Turing.sample(...)` becomes a pluggable `sampler(log_density, sys, d, k_max)`.
- 6.1 and 6.2 are direct translations (rollout + pdf carry over exactly).
