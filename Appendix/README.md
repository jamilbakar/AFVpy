# Appendix — Example Systems

The concrete systems the chapters plug into. Each is an *environment* (dynamics +
initial-state distribution), sometimes bundled with an *agent* and *sensor*. Every
system exposes the same interface — `step`, `initial_distribution`, and (for the
disturbance-based chapters) the noise distributions `Da`/`Ds`/`Do` — so any
validation algorithm can run against any system unchanged.

### appendix_simple_gaussian.py — Simple Gaussian (A.2)
The minimal system: the state is a single number drawn from a standard normal and
then held fixed for all time. `SimpleGaussian` is the environment, `NoAgent` takes
no action, and `IdealSensor` returns the true state with no noise. Used as the
simplest possible thing to validate.

### appendix_mv_gaussian.py — Multivariate Gaussian (A.3)
Same idea in two dimensions: `MvGaussian` draws its state from a 2-D normal with
mean at the origin and identity covariance. Useful for testing algorithms on a
continuous, multi-dimensional state.

### appendix_mass_spring_damper.py — Mass-Spring-Damper (A.4)
A linear dynamical system: the state is position and velocity, and the dynamics are
the matrices `Ts` (state transition) and `Ta` (action effect). It also defines an
`AdditiveNoiseSensor` (adds measurement noise) and a `ProportionalController`
(pushes the mass back toward rest). This is the linear system used in the
reachability chapters.

### appendix_inverted_pendulum.py — Inverted Pendulum (A.5)
A classic nonlinear system: balance a pole by applying torque at its base. `step`
integrates the nonlinear dynamics (gravity + torque) one time step, clamping both
the applied torque and the angular velocity to their limits.

### appendix_grid_world.py — Grid World (A.6)
A discrete system: an agent moves on a grid, usually going the intended direction
but occasionally "slipping" to a random one, and staying put if it would leave the
grid. `DiscreteAgent` is a lookup-table policy; `GridWorld` is the environment.

### appendix_continuum_world.py — Continuum World (A.7)
The continuous version of grid world: the agent lives at any real point, its
intended direction is nudged by a random push and re-normalized to unit length, and
special circular regions are the goal/obstacle. `InterpAgent` chooses actions by
interpolating a value table over the state.

### appendix_collision_avoidance.py — Aircraft Collision Avoidance (A.8)
Issue climb/descend/no-op advisories to avoid an intruder aircraft. The 4-D state is
relative altitude, relative vertical rate, previous advisory, and time to collision;
`step` advances the kinematics with the commanded rate change (bounded by an
acceleration limit) plus noise.

## Walls / problems

- **Distributions are objects, not samples.** Julia `Ps(env) = Normal()` returns a *distribution*; the naive `np.random.normal(0,1)` returns one sample. Fixed by returning `scipy.stats.norm(...)`.
- **`Uniform(a,b)` reparameterization.** Julia's `Uniform(a,b)` → scipy `uniform(loc=a, scale=b-a)`.
- **1-based → 0-based indexing.** Grid/continuum action indices shift by one; grid *positions* stay 1-based (they're coordinates, not list indices).
- **Name clashes.** Julia lets a struct field and a function share a name (`Do`, `Ds`); Python can't, so the accessors were renamed (`Do_dist`, `Ds_dist`).
- **Size-agnostic `I`** → explicit `np.eye(2)`; **GridInterpolations.jl / LazySets.jl** (continuum world) replaced by `scipy.interpolate` + a small containment LP.
