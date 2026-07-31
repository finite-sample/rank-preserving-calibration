# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2026-07-31

An audit against an independent convex solver. The core mathematics held; four
things around it did not, and the practical ceiling that stopped the package being
usable at scale is gone.

### Added
- **`calibrate()`** — the recommended entry point, dispatching on `method`
  (`"auto"`, `"qp"`, `"dykstra"`, `"admm"`). Existing functions are unchanged.
- **`calibrate_qp()`** — solves the projection exactly as a sparse quadratic program
  via Clarabel. Faster than alternating projections at **every** size measured: 4x at
  `N=10`, 13x at `N=25`, 2230x at `N=50`, and it reaches sizes the projection method
  cannot (`N=1600` in 1.4 s; `N=400` previously did not converge in 60,000 iterations).

  The lesson is about the algorithm class, not the problem. Dykstra and OSQP are both
  first-order and these constraint sets meet at a shallow angle; OSQP is *worse* than
  Dykstra here, returning 592 rank violations at `N=400` while reporting that it ran.
  An interior-point method is indifferent to that geometry.

  Its solver tolerance defaults to `1e-10`, tighter than Clarabel's own: at the looser
  default the result carries small isotonic violations — 204 at `N=400`, 1655 at
  `N=1600`, each between 1e-9 and 1e-7. Rank preservation is this package's guarantee,
  and one that holds only to 1e-7 is a different guarantee. Tightening costs 0.24 s
  against 0.19 s.
- **`is_feasible()`** — an exported feasibility predicate with *absolute* tolerances.

### Fixed
- **Every feasibility check silently ignored its stated tolerance.**
  `np.allclose(x, target, atol=...)` keeps `rtol=1e-5` unless told otherwise, so a
  stated `atol=1e-12` on row sums behaved as `1e-5`, and the column check — measured
  against `M_j`, which grows with `N` — became *looser* as the problem got larger.
  `converged=True` was returned with `max_row_error=1.0e-05`.
- **The ADMM `Q`-update was misderived.** It pulled every *element* toward the row and
  column targets, when those target the row and column *sums*. Each row was driven
  toward `J*(1 + M_j)/2` instead of 1, for any `rho`, so `calibrate_admm` failed on
  problems as small as `N=25`. Replaced with the exact minimiser, which closes in the
  row sums, column sums and grand total — no linear system, still `O(NJ)`, verified
  against a direct solve to 4.5e-14.
- **`calibrate_admm` could return a rank-violating matrix silently.** When its final
  projection failed it caught the error, warned only at `verbose=True` (default
  `False`), and returned its raw iterate — 27 rank violations on a 25-row problem, with
  an objective *below* the true optimum because the point was infeasible. It now
  raises, and the projection gets a budget proportional to `N` rather than a flat 1500.
- **The documented behaviour on infeasible targets was wrong.** The README promised
  convergence to "the closest point satisfying both sets of constraints"; no such point
  exists when the sets are disjoint, and the code raises. Feasibility requires
  `sum(M) == N` exactly — the warning fired only above 10% of `N`, so a 2% mismatch
  warned about nothing and then failed hard. It now fires at floating-point slack and
  says how to fix it.

### Changed
- **`calibrate_ovr_isotonic` is documented as the baseline it is.** Its row
  normalisation inverts **57%** of adjacent within-class pairs on columns that are
  perfectly isotonic beforehand — it does not preserve rank, despite living in a
  package named for it. Tied scores are now pooled before interpolation, so the result
  no longer depends on row order.
- Convergence tolerances are absolute and achievable (`1e-8`). Alternating projections
  end each sweep on a column projection, which perturbs the row sums, so no finite
  iterate satisfies both constraint sets exactly.
- `README.md` and `docs/source/theory.md` record the measured iteration scaling, the
  interior-point finding, and the exact-feasibility rule — including that when
  `sum(M) == N` the intersection is *never* empty, since `Q[i,j] = M[j]/N` always
  satisfies it, so failure is conditioning rather than infeasibility.

### Dependencies
- Added `scipy` and `clarabel` (Apache-2.0, ~2.5 MB). The `dykstra` path remains
  numpy-only.

## [0.7.1] - 2024-12-21

### Changed
- **Removed tqdm dependency**
  - Eliminated optional import patterns for cleaner dependency management
  - Removed `progress_bar` parameter from `calibrate_dykstra` and `calibrate_admm` functions
  - Updated examples and documentation to remove progress bar references

- **Enhanced docstrings**
  - Converted all docstrings to Google style format for consistency
  - Added comprehensive examples, args, and returns documentation
  - Improved API documentation across all public functions

- **Python modernization**
  - Refactored input validation to use Python 3.11+ match statements
  - Updated validation patterns for better readability and maintainability

### Removed
- **Breaking Change**: `progress_bar` parameter no longer supported
- Removed tqdm from both performance and dev dependency groups

## [0.7.0] - 2024-12-13

### Added
- **Performance optimizations with Numba JIT compilation**
  - JIT-compiled versions of performance-critical functions (`project_row_simplex`, `isotonic_regression`)
  - 1.5-3x speedup on moderate-sized problems when Numba is available
  - Graceful fallback to pure Python when Numba is not installed
  - New `use_jit` parameter for `calibrate_dykstra` and `calibrate_admm` (defaults to True)

- **Progress bar support for long-running calibrations**
  - Optional tqdm progress bars via `progress_bar` parameter
  - Works with both Dykstra and ADMM algorithms
  - Graceful fallback when tqdm is not installed

- **Modern Python 3.11+ features**
  - Added `slots=True` to dataclasses for better memory efficiency
  - Use pathlib instead of os.path for file operations
  - Updated to use importlib.metadata for version handling

### Changed
- **Dependency management improvements**
  - Added optional performance dependencies: `numba>=0.56`, `tqdm>=4.65`
  - Configured deptry for proper dependency tracking
  - Updated dev dependency groups to include performance deps

- **Code quality enhancements**
  - Fixed all linting issues for consistent code formatting
  - Improved numerical tolerance handling in tests
  - Enhanced error messages and validation

### Fixed
- Improved numerical stability in edge cases
- Better handling of optional dependencies with proper fallbacks
- Fixed test tolerances to account for expected JIT floating-point differences

## [0.6.0] - Previous Release
- Basic rank-preserving calibration functionality
- Dykstra's alternating projections algorithm
- ADMM optimization algorithm
- Nearly isotonic constraints (epsilon-slack and lambda-penalty)
- Comprehensive test suite and documentation
