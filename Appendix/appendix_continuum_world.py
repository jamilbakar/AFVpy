import numpy as np
from scipy.stats import multivariate_normal
from scipy.interpolate import RegularGridInterpolator


def norm(v):
    return np.linalg.norm(v)


def normalize(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


class SetCategorical:
    def __init__(self, values, probs=None):
        self.values = values
        self.probs = None if probs is None else np.asarray(probs, dtype=float)

    def rvs(self):
        idx = np.random.choice(len(self.values), p=self.probs)
        return self.values[idx]


class RectangleGrid:
    # Stand-in for GridInterpolations.jl RectangleGrid: cutpoints per dimension.
    def __init__(self, *cutpoints):
        self.cutpoints = [np.asarray(c, dtype=float) for c in cutpoints]

    @property
    def shape(self):
        return tuple(len(c) for c in self.cutpoints)


def interpolate(grid, values, point):
    # Multilinear interpolation. values is flat in Julia (column-major) order.
    V = np.asarray(values, dtype=float).reshape(grid.shape, order='F')
    f = RegularGridInterpolator(grid.cutpoints, V, bounds_error=False, fill_value=None)
    # GridInterpolations.jl clamps out-of-grid points to the boundary.
    pt = [np.clip(p, c[0], c[-1]) for p, c in zip(point, grid.cutpoints)]
    return float(f(pt))


class ContinuumWorld:
    def __init__(self, size=None, terminal_centers=None, terminal_radii=None,
                 directions=None, Sigma=None):
        self.size = size or [10, 10]                                # dimensions
        # obstacle and goal centers
        self.terminal_centers = terminal_centers or [[4.5, 4.5], [6.5, 7.5]]
        self.terminal_radii = terminal_radii or [0.5, 0.5]          # radii
        # up, down, left, right
        self.directions = directions or [[0, 1], [0, -1], [-1, 0], [1, 0]]
        self.Sigma = 0.5 * np.eye(2) if Sigma is None else np.asarray(Sigma)

    def Ds(self, s, a):
        return multivariate_normal(np.zeros(2), self.Sigma)

    def step(self, s, a, x=None):
        if x is None:
            x = self.Ds(s, a).rvs()
        s = np.asarray(s, dtype=float)
        is_terminal = [norm(s - np.asarray(c)) <= r
                       for c, r in zip(self.terminal_centers, self.terminal_radii)]
        if any(is_terminal):
            return s
        dir = normalize(np.asarray(self.directions[a], dtype=float) + x)  # a is 0-based
        return np.clip(s + dir, [0, 0], self.size)

    def initial_distribution(self):
        return SetCategorical([[0.5, 0.5]])


class InterpAgent:
    def __init__(self, grid, Q):
        self.grid = grid  # grid of discrete states
        self.Q = Q        # corresponding state-action values (one array per action)

    def act(self, s):
        return int(np.argmax([interpolate(self.grid, q, s) for q in self.Q]))