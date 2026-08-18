# Chapter 12 — Operational Design Domain Monitoring

- **algorithm_12_1.py** — `KNNMonitor`: in-distribution if all k nearest neighbors are within a threshold distance.
- **algorithm_12_2.py** — `HullMonitor`: in-distribution if the input lies in the convex hull of any cluster.
- **algorithm_12_3.py** — `SuperlevelSetMonitor`: in-distribution if the density at the input exceeds a threshold.

## Walls / problems

- **NearestNeighbors.jl → `scipy.spatial.cKDTree`** (12.1). Two details: Julia stores one datapoint per *column* so the matrix is transposed, and scipy's `query` returns `(distances, indices)` — the reverse of Julia's `knn`.
- **LazySets `convex_hull` + `VPolytope` membership → a linear program** (12.2). Rather than a polytope library, membership is tested as "is the input a convex combination of the cluster points" via `scipy.optimize.linprog` — equivalent, works in any dimension.
- 12.3 is a direct one-liner (`dist.pdf(input) > γ`).
- All three run with only numpy/scipy.
