# Algorithms for Validation — Python

[![Tests](https://github.com/jamilbakar/Validation-translation/actions/workflows/tests.yml/badge.svg)](https://github.com/jamilbakar/Validation-translation/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/jamilbakar/Validation-translation/branch/main/graph/badge.svg)](https://codecov.io/gh/jamilbakar/Validation-translation)

Python translations of the algorithms from *Algorithms for Validation*
(Kochenderfer, Katz, Corso, Moss), ported from the book's Julia. Every algorithm
from the Appendix systems and Chapters 1–12 is included, with a unit-test suite,
continuous integration, and code coverage.

## Layout

```
Appendix/      appendix_<system>.py         (the example systems)
Chapter 1/     algorithm_1_1.py …           (algorithms, named algorithm_<chapter>_<n>.py)
…
Chapter 9/     algorithm_9_1.py …, Interval_ad.py   (interval arithmetic + AD backend)
Chapter 12/    algorithm_12_1.py …
tests/         one test file per chapter + conftest.py
.github/workflows/tests.yml       CI: run tests + coverage on every push/PR
pyproject.toml requirements.txt
```

Files are named `algorithm_<chapter>_<n>.py` so each is identifiable by chapter,
and the Appendix systems by name (`appendix_simple_gaussian.py`, etc.).

## Running the tests

```bash
pip install -r requirements.txt
pytest
```

`pytest` runs the whole suite and prints a coverage summary. The tests locate
each module across the chapter folders via a small `load("algorithm_4_1")` helper
in `tests/conftest.py`. Tests needing an uninstalled optional library are skipped,
not failed.

## Dependencies

| Package | Used by | Install |
|---|---|---|
| numpy, scipy | most chapters | `pip install numpy scipy` |
| cvxpy | Ch.8 (A8.6, A8.8) | `pip install cvxpy` |
| stljax (+ jax) | STL robustness: A4.7/A4.9/A4.10, A5.1, A11.1–A11.5 | `pip install stljax` |
| pycvxset | Ch.8 set ops (A8.2–A8.5, A8.7) | see below |
| pytest, pytest-cov | tests | `pip install pytest pytest-cov` |

`Interval_ad.py` (Chapter 9) is pure Python — no extra install.

pycvxset needs the gmp/cddlib system libraries first:

```bash
# macOS
brew install gmp cddlib
export CFLAGS="-I$(brew --prefix)/include -L$(brew --prefix)/lib"
pip install "pycddlib>=3.0.0" "git+https://github.com/merlresearch/pycvxset.git"
```

## Library backends (what replaced the Julia packages)

| Julia | Python |
|---|---|
| SignalTemporalLogic.jl | [stljax](https://github.com/UW-CTRL/stljax) |
| ForwardDiff.jl | `jax.grad` (Ch.11) / `Interval_ad` forward-mode AD (Ch.9) |
| LazySets.jl | [pycvxset](https://github.com/merlresearch/pycvxset) (Ch.8) / hyperrectangles via `Interval_ad` (Ch.9) |
| IntervalArithmetic.jl | `Interval_ad.Interval` |
| JuMP.jl | cvxpy |
| Graphs.jl | inline `WeightedGraph` (Ch.10) |

## System-specific hooks

A few functions are intentionally system-specific in the book (they depend on
the problem, not on a library): `extract` (A4.7/A4.9/A4.10, A5.1, A11.1–A11.3,
A11.5), `perturb` (A5.6, A7.7, A7.10, A7.11, A11.1), `proposal` (A7.6), and
`fit` (A7.5). The test suite shows minimal mock implementations.

## Continuous integration & badges

`.github/workflows/tests.yml` installs the dependencies (including pycvxset),
runs `pytest` with coverage on every push/PR, and uploads to Codecov.

- The **Tests** badge works as soon as you push (built into GitHub Actions).
- For the **coverage** badge, sign in at [codecov.io](https://codecov.io) with
  GitHub and add this repository. Public repos need no token; private repos
  need the `CODECOV_TOKEN` repo secret.
