# Tests

Run from the repo root with `pytest`. Each test loads the algorithm files by path
(via the `load("algorithm_4_1")` helper in `conftest.py`, since the filenames
contain dots and can't be imported directly). Tests that need an optional library
(stljax, jax, cvxpy, pycvxset) are **skipped** rather than failed if it isn't
installed.

## conftest.py — shared fixtures
- **`load(stem)`** — locate and import an algorithm file by name from anywhere under the repo.
- **`Normal`** — a lightweight normal distribution (`.rvs`, `.logpdf`) used to avoid slow scipy objects in tight loops.
- **`sys_disturbance`** — a minimal system whose components expose the disturbance API (`observe`/`act`/`step`, `Da`/`Ds`/`Do`, `initial_distribution`).
- **`traj_distribution`** — a nominal trajectory distribution object (`.Ps`/`.D`/`.d`).
- **`spec_threshold`** — a specification whose `evaluate` fails if any state exceeds 5.

## test_interval_ad.py — the Chapter 9 backend
- **test_interval_arithmetic** — interval add/subtract/multiply give correct bounds.
- **test_interval_sin_finds_extrema** — `sin([0, π]) = [0, 1]` (captures the max at π/2 inside the interval).
- **test_gradient_over_intervals** — forward-mode gradient of `x0² + sin(x1)` at (3, 0) is (6, 1).
- **test_hessian_over_intervals** — Hessian of `x0² + x0·x1` is `[[2, 1], [1, 0]]`.
- **test_hessian_sin_over_interval** — second derivative of `sin` over `[0, π/2]` is `[-1, 0]`.

## test_appendix_systems.py — the example systems
- **test_simple_gaussian** — state stays fixed, standard-normal initial distribution, no-op agent, perfect sensor.
- **test_mv_gaussian** — 2-D initial distribution has mean 0 and identity covariance.
- **test_mass_spring_damper** — the `Ts` matrix and a one-step update match hand-computed values.
- **test_inverted_pendulum** — an out-of-range torque is clamped (angular velocity stays within its bound).
- **test_grid_world** — moving right advances the cell; a terminal cell stays put.
- **test_continuum_world_normalize** — `normalize([3, 4]) = [0.6, 0.8]`.
- **test_collision_avoidance** — a "climb" advisory produces the expected next state.

## test_ch01_03.py — framework, estimation, specs
- **test_ch1_step_and_rollout** — one `step` passes the state through correctly; a depth-5 rollout yields 5 transitions.
- **test_ch1_specification** — a registered `evaluate` makes `isfailure` True on a violating trajectory, False otherwise.
- **test_ch2_mle** — MLE recovers the mean (≈3) of 300 `Normal(3,1)` samples.
- **test_ch3_ltl_stl** — LTL "always |s|>50" flags failure (a state is 40); the STL version restricted to a window passes.

## test_ch04.py — falsification through optimization
- **test_ch4_disturbance_step** — the disturbance-sampling step returns `(o, a, s_next, x)`.
- **test_ch4_nominal_and_rollout** — nominal trajectory distribution has the right depth; rollout length matches.
- **test_ch4_fixed_rollout_and_pdf** — fixed-disturbance rollout length is correct and `pdf > 0`.
- **test_ch4_direct_falsification** — `falsify` returns a list of failing trajectories.
- **test_ch4_optimization_based_falsification** — the optimizer minimizes a toy objective (argmin at 3).
- **test_ch4_robustness_objective** *(stljax)* — robustness is positive when the trajectory stays safe, negative when it dips.
- **test_ch4_likelihood_objectives** *(stljax)* — the likelihood and weighted-likelihood objectives return finite values.

## test_ch05.py — falsification through planning
- **test_ch5_average_dispersion / test_ch5_star_discrepancy** — coverage metrics return sane ranges (`0<δ≤1`, `lb≤ub`).
- **test_ch5_lcb** — LCB selects the lowest-value child.
- **test_ch5_rrt_falsify / test_ch5_mcts_falsify** — RRT and MCTS run end-to-end through `falsify` and return a list.
- **test_ch5_shooting_robustness** *(stljax)* — the multiple-shooting objective returns a finite value.

## test_ch06.py — falsification through sampling
- **test_ch6_rejection_sampling** — with target = proposal and c = 1, all 30 samples are accepted.
- **test_ch6_mcmc_sampling** — burn-in + thinning of 20 samples leaves the expected 8.

## test_ch07.py — failure-probability estimation
- **test_ch7_direct_estimation / importance_sampling / multiple_importance_sampling** — each returns a probability in [0, 1].
- **test_ch7_bayesian_estimation** — returns a Beta posterior object.
- **test_ch7_bridge_functions** — the optimal-bridge + estimator ratio is finite.
- **test_ch7_self_importance_sampling** — the self-normalized estimate is finite.

## test_ch08.py — reachability (linear)
- **test_ch8_avoid_set_spec** — a trajectory staying outside the avoid set passes; one entering it fails.
- **test_ch8_support_function** *(cvxpy)* — the support function at depth 3 equals the true reachable extent (≈2).
- **test_ch8_satisfies_lp** *(cvxpy)* — a far avoid set is safe; a reachable one is flagged as a failure.
- **test_ch8_pycvxset_import** *(pycvxset)* — the pycvxset-backed module loads and exposes `reachable`.

## test_ch09.py — reachability (nonlinear)
- **test_ch9_natural_inclusion_sound** — the natural-inclusion box contains the true one-step reachable range.
- **test_ch9_taylor_inclusion_sound** — order-1 and order-2 Taylor boxes are also sound overapproximations.
- **test_ch9_conservative_linearization_sound** — the conservative-linearization box is sound.
- **test_ch9_concrete_variants_run** — the concrete Taylor and conservative variants run and return the right number of sets.

## test_ch10.py — reachability (discrete)
Uses a 3-state Markov chain (0→{0,1}, 1→2, 2 absorbing target).
- **test_ch10_to_graph** — the graph has the right edge weights out of state 0.
- **test_ch10_forward_backward** — forward and backward reachability both reach all three states.
- **test_ch10_occupancy** — occupancy probability of the absorbing state at t=3 is 0.5.
- **test_ch10_horizons** — finite-horizon P(reach)=0.875, infinite-horizon P(reach)=1.0.

## test_ch11.py — failure analysis
- **test_ch11_kmeans** — clusters six points into the two expected groups of three.
- **test_ch11_sensitivity** — sensitivities are the right length and non-negative.
- **test_ch11_shapley** *(stljax)* — Shapley values are finite, one per time step.
- **test_ch11_counterfactual** *(stljax)* — the counterfactual objective returns a finite value.
- **test_ch11_gradient_and_integrated** *(stljax + jax)* — gradient sensitivity and integrated gradients return finite vectors via `jax.grad`.

## test_ch12.py — ODD monitoring
- **test_ch12_knn_monitor** — a near-cluster point is in-distribution; a far one is not.
- **test_ch12_hull_monitor** — a point inside a cluster's convex hull is in-distribution; one between clusters is not.
- **test_ch12_superlevel_set_monitor** — a high-density point is in-distribution; a tail point is not.

---

*Python code created by Jamil Bakar.*
