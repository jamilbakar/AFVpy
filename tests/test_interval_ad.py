import math
from conftest import load  # noqa: F401  (ensures the Interval_ad dir is on sys.path)
import Interval_ad as ia

Interval = ia.Interval


def test_interval_arithmetic():
    a, b = Interval(1, 2), Interval(-1, 3)
    assert (a + b).lo == 0 and (a + b).hi == 5
    assert (a * b).lo == -2 and (a * b).hi == 6
    assert (a - b).lo == -2 and (a - b).hi == 3


def test_interval_sin_finds_extrema():
    s = ia.sin(Interval(0, math.pi))
    assert abs(s.lo) < 1e-9 and abs(s.hi - 1.0) < 1e-9


def test_gradient_over_intervals():
    g = ia.gradient(lambda x: x[0] * x[0] + ia.dsin(x[1]), [Interval(3, 3), Interval(0, 0)])
    assert abs(g[0].lo - 6.0) < 1e-9 and abs(g[1].lo - 1.0) < 1e-9


def test_hessian_over_intervals():
    H = ia.hessian(lambda x: x[0] * x[0] + x[0] * x[1], [Interval(3, 3), Interval(5, 5)])
    assert H[0][0].lo == 2 and H[0][1].lo == 1 and H[1][1].lo == 0


def test_hessian_sin_over_interval():
    H = ia.hessian(lambda x: ia.hdsin(x[0]), [Interval(0, math.pi / 2)])
    assert abs(H[0][0].lo + 1.0) < 1e-9 and abs(H[0][0].hi) < 1e-9
