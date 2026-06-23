README - inverted_pendulum.py
HOW TO RUN (Terminal and VS Code)
----------
1. Make sure Python is installed on your computer.

2. Install the required libraries by typing this in your terminal:
   pip install scipy numpy

3. Run it by typing:
   python inverted_pendulum.py


WHAT EACH FUNCTION DOES
------------------------

This file models an inverted pendulum — a pole you try to balance upright
by applying a twisting force (torque) at its base.
The state s is two numbers: the angle theta and the angular velocity w,
written as [theta, w]. The action a is the torque applied.

Product.rvs()
   A helper that holds two independent random ranges and draws one number
   from each, returning them together as a pair.
   Used to build the random starting state.

InvertedPendulum.step(s, a)
   Takes the current state s = [angle, angular velocity] and an action a
   (the torque). Returns the next state one time step later, using the
   pendulum's physics (gravity pulling it down, plus your torque).
   The torque is limited to a maximum, and the angular velocity is also
   capped so the pole cannot spin arbitrarily fast.

InvertedPendulum.initial_distribution()
   Returns the random starting state: the pole begins at a small angle near
   upright (between about -11 and +11 degrees) with a small random spin.