# Chapter 7 — Failure Probability Estimation

Moving from finding *a* failure to estimating *how likely* failure is. These are
Monte-Carlo estimators that trade off accuracy against the number of rollouts,
especially for rare failures.

### algorithm_7_1.py — Direct estimation
The simplest estimate: roll out `m` nominal trajectories and report the fraction
that fail. Unbiased, but needs enormous `m` when failures are rare.

### algorithm_7_2.py — Bayesian estimation
Treats the failure probability as unknown with a Beta prior; after observing `n`
failures in `m` rollouts it returns the Beta posterior, giving both an estimate and
its uncertainty.

### algorithm_7_3.py — Importance sampling
Rolls out from a proposal distribution `q` that makes failures more common, then
reweights each sample by `pdf(p)/pdf(q)` so the estimate is still unbiased for the
true nominal probability `p`. Much more efficient than direct sampling for rare
events — if `q` is good.

### algorithm_7_4.py — Multiple importance sampling
Uses several proposal distributions at once and combines them with a weighting
scheme (`smis` for standard MIS, `dmmis` for the deterministic-mixture variant),
which is more robust than relying on a single proposal.

### algorithm_7_5.py — Cross-entropy method
Iteratively *learns* a good proposal: sample, keep the `m_elite` most-failing
samples, refit the proposal to them, and repeat; then do a final importance-sampling
estimate with the learned proposal.

### algorithm_7_6.py — Population Monte Carlo
Maintains a *population* of proposals, reweights and resamples them each iteration to
concentrate on failure regions, and finishes with a multiple-importance-sampling
estimate over the final population.

### algorithm_7_7.py — Sequential Monte Carlo
Walks through a sequence of intermediate distributions bridging the nominal
distribution to the failure distribution, reweighting and resampling at each stage
so samples migrate gradually into the rare failure region.

### algorithm_7_8.py — Optimal bridge / bridge-sampling estimator
Helper routines for bridge sampling: `bridge_sampling_estimator` computes the ratio
of two distributions' normalizing constants from samples of each, and
`optimal_bridge` iterates that to build the variance-optimal bridge density between
them.

### algorithm_7_9.py — Self-normalized importance sampling
Importance sampling when the proposal density is only known up to a constant: it
normalizes the importance weights so they sum to one, then estimates the failure
probability from the normalized weights.

### algorithm_7_10.py — Bridge-sampling estimation
Chains bridge sampling across a sequence of intermediate densities — resampling,
perturbing, fitting the optimal bridge, and multiplying the per-stage
normalizing-constant ratios — to estimate the (tiny) failure probability.

### algorithm_7_11.py — Adaptive multilevel splitting
Estimates a rare-event probability by progressively lowering a threshold: each
iteration keeps the elite samples, estimates the conditional probability of crossing
the next threshold, and resamples/perturbs toward it; the product of the conditional
probabilities is the failure probability.

## Walls / problems

- **System-specific stubs:** `fit` (7.5), `proposal` (7.6), `perturb` (7.7, 7.10, 7.11) — supply these to run those algorithms.
- **`Beta` → `scipy.stats.beta`** (prior params read via `.args`).
- **Source oddity in 7.9:** the code uses `mean(...)` on already-normalized weights; self-normalized IS (eq 7.33) is usually a weighted *sum* — reproduced as `np.mean` with a comment (swap to `np.sum` to match the equation).
- **`for k in k_max` in 7.8** (iterating an integer, not valid Julia as printed) → `range(k_max)`; the `gb` closure captures the loop-updated `ratio` by late binding.
- `rand(Categorical(w/sum w), m)` → `np.random.choice(..., size=m, p=...)`.

---

*Python code created by Jamil Bakar.*
