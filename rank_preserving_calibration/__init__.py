"""
Rank-preserving calibration of multiclass probabilities.

This package provides robust implementations of rank-preserving calibration
algorithms including Dykstra's alternating projections (exact intersection)
and ADMM (penalty-based with final snap to the exact projection).

Quick start
-----------
>>> import numpy as np
>>> from rank_preserving_calibration import calibrate_dykstra, feasibility_metrics, isotonic_metrics
>>> # Toy data
>>> rng = np.random.default_rng(42)
>>> P = rng.dirichlet(np.ones(4), size=100)       # N x J predicted probs (rows sum to 1)
>>> M = (P.sum(axis=0) + rng.normal(0, 0.05, 4))  # target column marginals (sum ≈ N)
>>> M = np.maximum(M, 1e-3)
>>> # Calibrate
>>> res = calibrate_dykstra(P, M, detect_cycles=False, rtol=0.0)
>>> print(res.converged, res.iterations)
>>> # Check invariants
>>> print(feasibility_metrics(res.Q, M))
>>> print(isotonic_metrics(res.Q, P)["max_rank_violation"])
"""

# Version info - imported dynamically from pyproject.toml
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rank_preserving_calibration")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__author__ = "Gaurav Sood"
__email__ = "gsood07@gmail.com"

# Public API: solvers and results
from .calibration import (
    ADMMResult,
    CalibrationError,
    CalibrationResult,
    calibrate_admm,
    calibrate_dykstra,
)

# Public API: metrics (feasibility, isotonicity, distances, scoring, sharpness, AUC deltas)
from .metrics import (
    auc_deltas,
    brier,
    classwise_ece,
    distance_metrics,
    feasibility_metrics,
    isotonic_metrics,
    nll,
    sharpness_metrics,
    tie_group_variance,
    top_label_ece,
)

# Public API: nearly-isotonic utilities
from .nearly import (
    project_near_isotonic_euclidean,
    prox_near_isotonic,
    prox_near_isotonic_with_sum,
)
from .ovr_isotonic import calibrate_ovr_isotonic

# What gets imported with: from rank_preserving_calibration import *
__all__ = [
    "ADMMResult",
    # Solvers & results
    "CalibrationError",
    "CalibrationResult",
    "auc_deltas",
    "brier",
    "calibrate_admm",
    "calibrate_dykstra",
    "calibrate_ovr_isotonic",
    "classwise_ece",
    "distance_metrics",
    # Metrics
    "feasibility_metrics",
    "isotonic_metrics",
    "nll",
    # Nearly-isotonic utilities
    "project_near_isotonic_euclidean",
    "prox_near_isotonic",
    "prox_near_isotonic_with_sum",
    "sharpness_metrics",
    "tie_group_variance",
    "top_label_ece",
]
