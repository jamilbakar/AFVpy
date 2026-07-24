import numpy as np
from itertools import product
def _volume(low, high):
    return float(np.prod(np.asarray(high) - np.asarray(low)))


def _in_rect(v, low, high):
    return bool(np.all(v >= low) and np.all(v <= high))


def star_discrepancy(points, lo, hi, lengths):
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    n = len(points)
    dim = len(lo)
    V = [(np.asarray(p, dtype=float) - lo) / (hi - lo) for p in points]
    ranges = [np.linspace(0, 1, L)[:-1] for L in lengths]         # drop last cutpoint
    steps = np.array([1.0 / (L - 1) for L in lengths])            # r.step
    zeros = np.zeros(dim)
    volB = _volume(zeros, np.ones(dim))
    lbs, ubs = [], []
    for grid_point in product(*ranges):
        gp = np.array(grid_point)
        hminus_high = gp
        hplus_high = gp + steps
        Vhm = sum(1 for v in V if _in_rect(v, zeros, hminus_high))
        Vhp = sum(1 for v in V if _in_rect(v, zeros, hplus_high))
        volhm = _volume(zeros, hminus_high)
        volhp = _volume(zeros, hplus_high)
        lbs.append(max(abs(Vhm / n - volhm / volB),
                       abs(Vhp / n - volhp / volB)))
        ubs.append(max(Vhp / n - volhm / volB,
                       volhp / volB - Vhm / n))
    return max(lbs), max(ubs)