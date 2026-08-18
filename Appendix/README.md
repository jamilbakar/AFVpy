# Appendix — Example Systems

The systems used throughout the book, each an environment (+ agent/sensor where
relevant). Files are named by system.

- **appendix_simple_gaussian.py** (A.2) — `SimpleGaussian`, `NoAgent`, `IdealSensor`. State sampled from a standard normal and held fixed; no agent; perfect sensor.
- **appendix_mv_gaussian.py** (A.3) — `MvGaussian`. 2-D state from a multivariate normal (mean 0, identity covariance).
- **appendix_mass_spring_damper.py** (A.4) — `MassSpringDamper` (linear `Ts`/`Ta`), `AdditiveNoiseSensor`, `ProportionalController`.
- **appendix_inverted_pendulum.py** (A.5) — `InvertedPendulum`; nonlinear dynamics with torque/angular-velocity clamping.
- **appendix_grid_world.py** (A.6) — `GridWorld` (discrete, slip probability), `DiscreteAgent`.
- **appendix_continuum_world.py** (A.7) — `ContinuumWorld` (continuous grid with disturbance push + renormalize), `InterpAgent`.
- **appendix_collision_avoidance.py** (A.8) — `CollisionAvoidance`; aircraft climb/descend advisories.

## Walls / problems

- **Distributions are objects, not samples.** Julia `Ps(env) = Normal()` returns a distribution; the naive Python `np.random.normal(0,1)` returns one sample. Fixed by returning `scipy.stats.norm(...)`.
- **`Uniform(a,b)` reparameterization.** Julia's `Uniform(a,b)` → scipy `uniform(loc=a, scale=b-a)`.
- **1-based → 0-based indexing.** Grid/continuum action and direction indices shift by one; grid *positions* stay 1-based (they're coordinates, not list indices).
- **Name clashes.** Julia lets a struct field and a function share a name (`Do`, `Ds`); Python can't, so the accessor was renamed (`Do_dist`, `Ds_dist`).
- **Size-agnostic `I`.** Julia's identity `I` → explicit `np.eye(2)`.
- **GridInterpolations.jl / LazySets.jl** (continuum world) have no drop-in Python equivalent; interpolation is done with `scipy.interpolate` and membership via a small LP.
