# Chapter 2 — Parameter Estimation

- **algorithm_2_1.py** — `MaximumLikelihoodParameterEstimation`: builds a cost function (sum of negative log-likelihoods) and hands it to a pluggable optimizer.
- **algorithm_2_2.py** — `BayesianParameterEstimation`: samples the posterior over parameters given a prior, likelihood, and a sampler.

## Walls / problems

- **`logpdf(dist, y)` → `dist.logpdf(y)`** (free function in Julia, method in Python).
- **Turing.jl has no drop-in Python equivalent.** The `@model` / `~` probabilistic program is rewritten by hand as a log-posterior (each `~` becomes an added `logpdf` term), and `Turing.sample(...)` becomes a pluggable `sampler(log_post, m)` you supply (e.g. a small Metropolis-Hastings, or PyMC/NumPyro). Faithful in meaning, not a literal transcription.
