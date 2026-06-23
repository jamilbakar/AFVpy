README - mv_gaussian.py
HOW TO RUN (Terminal and VS Code)
----------
1. Make sure Python is installed on your computer.
 
2. Install the required libraries by typing this in your terminal:
   pip install scipy numpy
 
3. Run it by typing:
   python mv_gaussian.py
 
 
WHAT EACH FUNCTION DOES
------------------------
 
MvGaussian.step(s, a)
   Takes a state s and an action a.
   Returns s unchanged — the world never changes no matter what action you take.
   Same idea as the simple Gaussian, but here the state has two numbers
   instead of one (a 2D point).
 
MvGaussian.initial_distribution()
   Returns a 2D bell curve centered at the origin (the point [0, 0]),
   with the same spread in both directions and no tilt.
   Used to randomly pick the starting state, which will be a pair of numbers.
 