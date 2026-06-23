README - grid_world.py
HOW TO RUN (Terminal and VS Code)
----------
1. Make sure Python is installed on your computer.

2. Install the required library by typing this in your terminal:
   pip install numpy

3. Run it by typing:
   python grid_world.py


WHAT EACH FUNCTION DOES
------------------------

This file models a grid world — an agent moving around squares on a grid.
The state is a position [x, y]. The action is a direction to move
(up, down, left, or right). Most of the time the agent moves the way it
intended, but sometimes it "slips" and goes a random direction instead.
There are special terminal squares: one is the goal, one is an obstacle.

NOTE ON COUNTING: Python counts from 0, so the four directions are numbered
0, 1, 2, 3 (up, down, left, right). The grid positions themselves count from
1 (the corners are [1,1] and [10,10]).

Categorical.rvs()
   Picks one direction number at random, using a list of probabilities.
   Used to decide whether the agent moves as intended or slips.

SetCategorical.rvs()
   Picks one value at random from a fixed list of options.
   Used here to choose the starting square.

GridWorld.Ds(s, a)
   Takes the current position s and the chosen action a.
   Returns the probabilities of actually moving each direction: a high chance
   (0.7) of going the way you intended, and a small chance of slipping into
   each of the other directions.

GridWorld.step(s, a, x=None)
   Takes the position s, the intended action a, and optionally the actual
   direction x. If x is left out, it is drawn randomly (this is where slipping
   happens). Returns the next position. If the agent is on a terminal square it
   stays there; if a move would leave the grid, it stays in place instead.

GridWorld.initial_distribution()
   Returns the starting state, which is always the corner square [1, 1].

DiscreteAgent.act(o)
   Takes the current position o.
   Looks it up in the policy table and returns the action to take there.
   The policy is just a lookup table telling the agent what to do in each square.