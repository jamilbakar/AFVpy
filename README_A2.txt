README - simple_gaussian.py
HOW TO RUN (Terminal and VS Code)
----------
1. Make sure Python is installed on your computer.

2. Install the required library by typing this in your terminal:
   pip install scipy


4. Run it by typing:
   python simple_gaussian.py


WHAT EACH FUNCTION DOES
------------------------

SimpleGaussian.step(s, a)
   Takes a state s and an action a.
   Returns s unchanged — the world never changes no matter what action you take.

SimpleGaussian.initial_distribution()
   Returns a bell curve centered at 0.
   Used to randomly pick the starting state.

NoAgent.act(s)
   Takes the current state s.
   Returns None — no action is taken, nobody is controlling anything.

IdealSensor.observe(s)
   Takes the true state s.
   Returns it exactly as it is — the sensor is perfect, no noise.