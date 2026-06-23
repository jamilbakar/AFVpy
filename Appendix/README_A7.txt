README - continuum_world.py
HOW TO RUN (Terminal and VS Code)
----------
1. Make sure Python is installed on your computer.

2. Install the required libraries by typing this in your terminal:
   pip install scipy numpy

3. Run it by typing:
   python continuum_world.py


WHAT EACH FUNCTION DOES
------------------------

This file models a continuum world — like the grid world, but the agent can
stand at any point, not just whole-number squares. The action is still one of
four directions, but instead of cleanly slipping into another direction, the
agent's intended direction gets nudged by a random push and then re-pointed to
a step of length 1. There are two special circles on the map: an obstacle and
a goal. Landing inside either circle ends the episode.

NOTE ON COUNTING: Python counts directions from 0, so up/down/left/right are
numbered 0, 1, 2, 3.

norm(v)
   Returns the length of a vector (how far a point is from the origin).
   Used to measure distance from the obstacle/goal centers.

normalize(v)
   Returns the vector pointing the same way but with length exactly 1.
   Used to turn the nudged direction back into a single step.

SetCategorical.rvs()
   Picks one value at random from a fixed list of options.
   Used here to choose the starting point.

RectangleGrid / interpolate(grid, values, point)
   A stand-in for the Julia grid-interpolation library. The grid is a set of
   reference points; interpolate estimates a value at any in-between point by
   blending the nearby reference values. Points outside the grid are pulled
   back to the edge.

ContinuumWorld.Ds(s, a)
   Returns the random "push" distribution (a 2D bell curve) that gets added to
   the intended direction, causing the agent to drift.

ContinuumWorld.step(s, a, x=None)
   Takes the position s, the intended action a, and optionally the random push
   x. If x is left out it is drawn randomly. Returns the next position. If the
   agent is inside the obstacle or goal circle it stays put; otherwise it moves
   one step in the (nudged, re-pointed) direction, kept inside the map edges.

ContinuumWorld.initial_distribution()
   Returns the starting point, which is always [0.5, 0.5].

InterpAgent.act(s)
   Takes the current position s. For each of the four actions it estimates the
   expected future reward by interpolating that action's value table at s, then
   returns the action with the highest estimate.