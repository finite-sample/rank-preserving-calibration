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
# Public API: analysis utilities for flatness and shift diagnostics
from .analysis import (
    compare_calibration_methods,
    flatness_bound,
    flatness_metrics,
    marginal_shift_metrics,
)
from .calibration import (
    ADMMResult,
    CalibrationError,
    CalibrationResult,
    calibrate_admm,
    calibrate_dykstra,
)

# Public API: KL divergence calibration
from .kl_calibration import (
    KLCalibrationResult,
    KLParetoResult,
    calibrate_kl,
    calibrate_kl_pareto,
    calibrate_kl_soft,
)

# Public API: KL nearly-isotonic utilities
from .kl_nearly import (
    project_near_kl_isotonic,
    prox_kl_near_isotonic,
)

# Public API: metrics (feasibility, isotonicity, distances, scoring, sharpness, AUC deltas)
# Public API: KL divergence metrics
from .metrics import (
    auc_deltas,
    brier,
    classwise_ece,
    column_variance,
    distance_metrics,
    feasibility_metrics,
    informativeness_ratio,
    isotonic_metrics,
    kl_divergence,
    nll,
    reverse_kl_divergence,
    sharpness_metrics,
    symmetrized_kl,
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

# Public API: soft calibration with tunable trade-offs
from .soft_calibration import (
    SoftCalibrationResult,
    calibrate_soft,
    calibrate_soft_admm,
)

# Public API: two-stage IPF-based calibration
from .two_stage import (
    IPFResult,
    TwoStageResult,
    calibrate_ipf,
    calibrate_two_stage,
)

# What gets imported with: from rank_preserving_calibration import *
__all__ = [
    "ADMMResult",
    "CalibrationError",
    "CalibrationResult",
    "IPFResult",
    "KLCalibrationResult",
    "KLParetoResult",
    "SoftCalibrationResult",
    "TwoStageResult",
    "auc_deltas",
    "brier",
    "calibrate_admm",
    "calibrate_dykstra",
    "calibrate_ipf",
    "calibrate_kl",
    "calibrate_kl_pareto",
    "calibrate_kl_soft",
    "calibrate_ovr_isotonic",
    "calibrate_soft",
    "calibrate_soft_admm",
    "calibrate_two_stage",
    "classwise_ece",
    "column_variance",
    "compare_calibration_methods",
    "distance_metrics",
    "feasibility_metrics",
    "flatness_bound",
    "flatness_metrics",
    "informativeness_ratio",
    "isotonic_metrics",
    "kl_divergence",
    "marginal_shift_metrics",
    "nll",
    "project_near_isotonic_euclidean",
    "project_near_kl_isotonic",
    "prox_kl_near_isotonic",
    "prox_near_isotonic",
    "prox_near_isotonic_with_sum",
    "reverse_kl_divergence",
    "sharpness_metrics",
    "symmetrized_kl",
    "tie_group_variance",
    "top_label_ece",
]
