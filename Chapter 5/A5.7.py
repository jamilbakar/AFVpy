import numpy as np
from itertools import product
def average_dispersion(points, lo, hi, lengths):
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    points_norm = [(np.asarray(p, dtype=float) - lo) / (hi - lo) for p in points]
    ranges = [np.linspace(0, 1, L) for L in lengths]
    delta = min(1.0 / (L - 1) for L in lengths)  # r.step of linspace(0,1,L)
    grid_dispersions = []
    for grid_point in product(*ranges):
        gp = np.array(grid_point)
        dmin = min(np.linalg.norm(gp - p) for p in points_norm)
        grid_dispersions.append(min(dmin, delta) / delta)
    return float(np.mean(grid_dispersions))