import numpy as np
from conftest import load


def test_simple_gaussian():          # A.2
    m = load("A2")
    env = m.SimpleGaussian()
    assert env.step(3.0, 1.0) == 3.0
    assert env.initial_distribution().mean() == 0
    assert m.NoAgent().act(5.0) is None
    assert m.IdealSensor().observe(7.0) == 7.0


def test_mv_gaussian():              # A.3
    d = load("A3").MvGaussian().initial_distribution()
    assert np.allclose(d.mean, [0, 0]) and np.allclose(d.cov, np.eye(2))


def test_mass_spring_damper():       # A.4
    e = load("A4").MassSpringDamper()
    assert np.allclose(e.Ts(), [[1, 0.05], [-0.5, 0.9]])
    assert np.allclose(e.step(np.array([0.5, -0.3]), np.array([0.3])), [0.485, -0.505])


def test_inverted_pendulum():        # A.5
    e = load("A5").InvertedPendulum()
    assert e.step(np.array([0.0, 0.0]), 100.0)[1] <= e.w_max


def test_grid_world():               # A.6
    e = load("A6").GridWorld()
    assert list(e.step([1, 1], 3, x=3)) == [2, 1]
    assert e.step([5, 5], 3, x=3) == [5, 5]


def test_continuum_world_normalize():  # A.7
    assert np.allclose(load("A7").normalize([3, 4]), [0.6, 0.8])


def test_collision_avoidance():      # A.8
    e = load("A8").CollisionAvoidance()
    assert e.step([0.0, -4.0, 0.0, 40.0], 2, x=0.0) == [-4.0, -3.0, 5.0, 39.0]
