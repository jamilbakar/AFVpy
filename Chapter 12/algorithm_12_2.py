import numpy as np
from scipy.optimize import linprog

# Algorithm 12.2: ODD monitoring via the convex hull of clustered data. For each
# cluster it checks whether the input lies inside the convex hull of that cluster's
# points. WALL: Julia's LazySets convex_hull + VPolytope membership -> membership is
# tested here with a linear program (input is a convex combination of the cluster
# points), which is equivalent and needs no polytope library. C maps clusters to
# column-index lists (a dict, or any iterable of (key, indices) pairs).


def _in_convex_hull(points, x):
    # points: (n, d) rows. Feasible convex combination points.T @ lam = x, sum lam = 1,
    # lam >= 0  <=>  x is inside the convex hull.
    n = points.shape[0]
    A_eq = np.vstack([points.T, np.ones(n)])
    b_eq = np.concatenate([np.asarray(x, dtype=float), [1.0]])
    res = linprog(c=np.zeros(n), A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * n)
    return bool(res.success)


class HullMonitor:
    def __init__(self, data, C):
        self.data = np.asarray(data, dtype=float)  # each column is a datapoint
        self.C = C  # collection of cluster column-index vectors

    def monitor(self, input):
        items = self.C.items() if hasattr(self.C, "items") else enumerate(self.C)
        for k, v in items:
            points = self.data[:, list(v)].T  # cluster points as rows
            if _in_convex_hull(points, input):  # input ∈ VPolytope(convex_hull(...))
                return True
        return False