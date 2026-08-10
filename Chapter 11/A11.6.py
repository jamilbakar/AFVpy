import numpy as np

# Algorithm 11.6: k-means clustering of failure trajectories. Extracts features, seeds
# centroids at random trajectories, then alternates assignment (nearest centroid) and
# centroid update (mean of assigned features). Fully runnable: phi (feature extractor)
# and d (distance metric) are supplied on the struct. Returns (clusters, centroids),
# where clusters are lists of 0-based trajectory indices.


class Kmeans:
    def __init__(self, taus, phi, d, k, max_iter):
        self.taus = taus        # trajectories to cluster
        self.phi = phi          # feature extraction: x = phi(tau)
        self.d = d              # distance metric: d(x[i], mu_j)
        self.k = k              # number of clusters
        self.max_iter = max_iter

    def describe(self, sys, psi):
        x = [self.phi(tau) for tau in self.taus]
        perm = np.random.permutation(len(x))
        mu = [x[i] for i in perm[:self.k]]        # x[randperm(...)[1:k]]
        C = [[] for _ in range(self.k)]
        for _ in range(self.max_iter):
            C = [[] for _ in range(self.k)]
            for i in range(len(x)):
                j = int(np.argmin([self.d(x[i], muj) for muj in mu]))
                C[j].append(i)
            for j in range(self.k):
                if C[j]:  # !isempty(C[j])
                    mu[j] = np.mean([x[i] for i in C[j]], axis=0)
        return C, mu