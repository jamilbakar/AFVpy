README - mass_spring_damper.py
HOW TO RUN (Terminal and VS Code)
----------
1. Make sure Python is installed on your computer.

2. Install the required libraries by typing this in your terminal:
   pip install scipy numpy

3. Run it by typing:
   python mass_spring_damper.py


WHAT EACH FUNCTION DOES
------------------------

This file models a mass on a spring with a damper (a shock absorber).
The state s is two numbers: position p and velocity v, written as [p, v].
The action a is a pushing force applied to the mass.

Product.rvs()
   A helper that holds two independent random ranges and draws one number
   from each, returning them together as a pair.
   Used to build the random starting state.

MassSpringDamper.step(s, a)
   Takes the current state s = [position, velocity] and an action a (a force).
   Returns the next state one time step later, using the physics of the
   spring and damper (the matrices Ts and Ta below do the actual math).

MassSpringDamper.Ts()
   Returns the 2x2 matrix that describes how the mass moves on its own
   (position and velocity feeding into each other) when no force is applied.

MassSpringDamper.Ta()
   Returns the column that describes how an applied force changes the state.

MassSpringDamper.initial_distribution()
   Returns the random starting state: the position starts somewhere between
   -0.2 and 0.2, and the velocity starts essentially at 0.

AdditiveNoiseSensor.observe(s, x=None)
   Takes the true state s and adds measurement noise to it.
   If you do not pass in noise x, it draws random noise automatically.
   Returns the noisy reading — a realistic, imperfect measurement.

AdditiveNoiseSensor.Do_dist(s)
   Returns the noise distribution the sensor uses (the random "wobble"
   added to each reading).

AdditiveNoiseSensor.Os()
   Returns the identity matrix, meaning the sensor reads the state directly
   (before noise is added) without mixing the numbers up.

ProportionalController.act(o)
   Takes an observation o (a state reading).
   Returns the control force to apply, calculated as the gain times the
   observation. This is what tries to push the mass back toward rest.

ProportionalController.Pi_o()
   Returns the gain row used in the calculation above.