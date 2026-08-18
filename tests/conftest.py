"""Test fixtures + a loader for the dotted algorithm filenames (e.g. A4.1.py).

Python can't `import A4.1` (the dot is illegal in a module name), so we load each
algorithm file by path with importlib. `load("A4.1")` returns the module object for
the file A4.1.py found anywhere under the repo root.
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Chapter 9 files do `from Interval_ad import ...`; put that file's dir on sys.path.
for _p in list(ROOT.rglob("Interval_ad.py")) + list(ROOT.rglob("interval_ad.py")):
    sys.path.insert(0, str(_p.parent))


def load(stem):
    """Load the algorithm file named <stem>.py (e.g. 'A4.1', 'A2') by path."""
    matches = [m for m in ROOT.rglob(stem + ".py") if m.is_file()]
    if not matches:
        pytest.skip(f"{stem}.py not found under {ROOT}")
    name = "alg_" + stem.replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, matches[0])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- lightweight distributions (avoid slow scipy frozen dists in tight loops) ----
class Normal:
    def __init__(self, mu=0.0, sd=1.0):
        self.mu, self.sd = float(mu), float(sd)

    def rvs(self, random_state=None):
        return float(np.random.normal(self.mu, self.sd))

    def logpdf(self, y):
        return -0.5 * ((y - self.mu) / self.sd) ** 2 - np.log(self.sd * np.sqrt(2 * np.pi))


class _Env:
    def initial_distribution(self):
        return Normal(0.0, 1.0)

    def step(self, s, a, xs=0.0):
        return s + a + xs

    def Ds(self, s, a):
        return Normal(0.0, 1.0)


class _Sensor:
    def observe(self, s, xo=0.0):
        return s + xo

    def Do(self, s):
        return Normal(0.0, 1.0)


class _Agent:
    def act(self, o, xa=0.0):
        return xa

    def Da(self, o):
        return Normal(0.0, 1.0)


class System:
    def __init__(self, agent, env, sensor):
        self.agent, self.env, self.sensor = agent, env, sensor


@pytest.fixture
def sys_disturbance():
    return System(_Agent(), _Env(), _Sensor())


@pytest.fixture
def traj_distribution():
    from collections import namedtuple
    Dd = namedtuple("Dd", ["Da", "Ds", "Do"])
    P = namedtuple("P", ["Ps", "D", "d"])
    D = Dd(Da=lambda o: Normal(0, 1), Ds=lambda s, a: Normal(0, 1), Do=lambda s: Normal(0, 1))
    return P(Ps=Normal(0, 1), D=D, d=3)


@pytest.fixture
def spec_threshold():
    class Spec:
        set = None
        formula = None

        def evaluate(self, tau):
            return all(getattr(tr, "s", tr) < 5 for tr in tau)

    return Spec()
