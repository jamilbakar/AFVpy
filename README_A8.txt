README - collision_avoidance.py
HOW TO RUN (Terminal and VS Code)
----------
1. Make sure Python is installed on your computer.

2. Install the required libraries by typing this in your terminal:
   pip install scipy numpy

3. Run it by typing:
   python collision_avoidance.py


WHAT THIS MODELS
------------------------
An aircraft collision-avoidance system. Our aircraft decides whether to climb,
descend, or do nothing to avoid an intruder coming head-on. The state has four
numbers:
   h      = our height relative to the intruder
   dh     = our vertical speed relative to the intruder
   a_prev = the previous advisory we gave
   tau    = seconds left until the possible collision
The action is one of three advisories, numbered from 0 (Python counts from 0):
   0 = descend 5 m/s
   1 = no advisory
   2 = climb 5 m/s


WHAT EACH FUNCTION DOES
------------------------
Product.rvs()
   Holds several independent random ranges and draws one number from each,
   returning them together. Builds the random starting state.

DiscreteNonParametric.rvs()
   A distribution over a fixed set of values with given probabilities. Used here
   to always return one fixed value (starting previous action 0, start time 40).

CollisionAvoidance.Ds_dist(s, a)
   Returns the noise distribution added to the vertical speed each step, standing
   in for unpredictable intruder behavior.

CollisionAvoidance.step(s, a, x=None)
   Takes the state s, the chosen advisory a, and optionally the noise x (drawn
   randomly if left out). Returns the next state one second later: height moves by
   the current vertical speed; if an advisory was given, the vertical speed changes
   toward the commanded rate but no faster than the acceleration limit; the previous
   action is recorded; and the countdown ticks down (never below -1).

CollisionAvoidance.initial_distribution()
   Returns the random starting state: height in [-100, 100], vertical speed in
   [-10, 10], previous action fixed at 0, time-to-collision fixed at 40.


NOTES ON THE JULIA TRANSLATION
------------------------
- Action indices are 0-based here (0/1/2), versus 1-based in Julia (1/2/3).
- Product and DiscreteNonParametric stand in for Julia's product_distribution
  and DiscreteNonParametric, which scipy has no direct equivalent for.
- Julia Uniform(a, b) -> scipy uniform(a, b - a).

hello there