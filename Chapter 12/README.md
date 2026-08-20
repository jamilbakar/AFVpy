# Chapter 12 — Operational Design Domain Monitoring

Runtime monitors that decide, for a *live* input, whether it falls inside the
operational design domain (ODD) — the region of inputs the system was validated on.
If an input drifts outside it, the monitor flags that the validation guarantees no
longer apply. Each monitor defines the ODD differently.

### algorithm_12_1.py — KNN monitor
Defines the ODD by the training/ODD data itself. It builds a k-d tree of the data
and, for a new input, flags it as in-distribution only if all `k` nearest neighbors
are within a threshold distance `γ` — i.e. the input is near enough to known data.

### algorithm_12_2.py — Hull monitor
Defines the ODD as the union of the convex hulls of clustered data. An input is
in-distribution if it lies inside any cluster's convex hull. Membership is tested by
checking whether the input can be written as a convex combination of that cluster's
points.

### algorithm_12_3.py — Superlevel-set monitor
Defines the ODD via a fitted probability distribution: an input is in-distribution if
its probability density exceeds a threshold `γ` (it lies in a high-density
"superlevel set"). The most compact definition when you have a good density model.

## Walls / problems

- **NearestNeighbors.jl → `scipy.spatial.cKDTree`** (12.1). Two details: Julia stores one datapoint per *column* so the matrix is transposed, and scipy's `query` returns `(distances, indices)` — the reverse of Julia's `knn`.
- **LazySets `convex_hull` + `VPolytope` membership → a linear program** (12.2). Rather than a polytope library, membership is tested as "is the input a convex combination of the cluster points" via `scipy.optimize.linprog` — equivalent, works in any dimension.
- 12.3 is a direct one-liner (`dist.pdf(input) > γ`).
- All three run with only numpy/scipy.

---

*Python code created by Jamil Bakar.*
