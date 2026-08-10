import numpy as np
from scipy.spatial import cKDTree

# Algorithm 12.1: ODD monitoring via k-nearest neighbors. Builds a k-d tree from the
# ODD data and flags an input as in-distribution if all k neighbors are within gamma.
# WALL: Julia's NearestNeighbors.jl -> scipy.spatial.cKDTree. Julia's data has one
# datapoint PER COLUMN; scipy wants one per row, so the matrix is transposed. scipy's
# query returns (distances, indices) — the opposite order of Julia's knn.


class KNNMonitor:
    def __init__(self, data, k, gamma):
        self.data = np.asarray(data, dtype=float)  # each column is a datapoint
        self.k = k                                  # number of neighbors
        self.gamma = gamma                          # threshold

    def monitor(self, input):
        kdtree = cKDTree(self.data.T)               # columns -> rows
        distances, neighbors = kdtree.query(np.asarray(input, dtype=float), k=self.k)
        distances = np.atleast_1d(distances)
        return bool(np.all(distances < self.gamma))