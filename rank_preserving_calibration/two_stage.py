# rank_preserving_calibration/two_stage.py
"""
Two-stage calibration approach using Iterative Proportional Fitting (IPF).

This module provides an alternative calibration strategy that may produce
less "flat" solutions when distribution shifts are large:

1. IPF (raking) to match target marginals while preserving relative structure
2. Optional isotonic projection to restore rank ordering

IPF tends to preserve more of the original probability structure compared
to direct Dykstra projection, making it useful when approximate marginal
matching is acceptable.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .calibration import (
    CalibrationError,
    _compute_rank_violation,
    _configure_logging,
    _isotonic_regression,
    _project_row_simplex,
)


@dataclass(slots=True)
class TwoStageResult:
    """Result from two-stage IPF-based calibration.

    Attributes:
        Q: Calibrated probability matrix.
        converged: Whether the algorithm converged.
        ipf_iterations: Number of IPF iterations performed.
        projection_iterations: Number of isotonic projection iterations.
        max_row_error: Maximum row sum error.
        max_col_error: Maximum column sum error.
        max_rank_violation: Maximum rank violation.
        ipf_result: The intermediate IPF result before isotonic projection.
    """

    Q: np.ndarray
    converged: bool
    ipf_iterations: int
    projection_iterations: int
    max_row_error: float
    max_col_error: float
    max_rank_violation: float
    ipf_result: np.ndarray


@dataclass(slots=True)
class IPFResult:
    """Result from Iterative Proportional Fitting.

    Attributes:
        Q: Probability matrix after IPF.
        converged: Whether IPF converged.
        iterations: Number of iterations.
        max_row_error: Maximum row sum error.
        max_col_error: Maximum column sum error.
        final_change: Final relative change between iterations.
    """

    Q: np.ndarray
    converged: bool
    iterations: int
    max_row_error: float
    max_col_error: float
    final_change: float


def calibrate_ipf(
    P: np.ndarray,
    M: np.ndarray,
    max_iters: int = 100,
    tol: float = 1e-8,
    verbose: bool = False,
) -> IPFResult:
    """Iterative Proportional Fitting (raking) to match target marginals.

    IPF alternately scales rows and columns to match target constraints:
    - Row scaling: Ensure rows sum to 1 (probability simplex)
    - Column scaling: Scale columns to match target marginals M

    Unlike Dykstra/ADMM, IPF preserves the relative structure of probabilities
    within rows (ratios between classes). This can produce less "flat" solutions
    but may not perfectly satisfy all constraints simultaneously.

    Args:
        P: Input probability matrix of shape (N, J).
        M: Target column sums of shape (J,). Should sum to N.
        max_iters: Maximum number of row/column scaling iterations.
        tol: Convergence tolerance for relative change.
        verbose: If True, enables debug logging.

    Returns:
        IPFResult with the scaled probability matrix.

    Raises:
        CalibrationError: If inputs are invalid.

    Examples:
        >>> import numpy as np
        >>> from rank_preserving_calibration import calibrate_ipf
        >>> P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> M = np.array([0.8, 0.8, 0.4])
        >>> result = calibrate_ipf(P, M)
        >>> print(f"Column sums: {result.Q.sum(axis=0)}")
    """
    _configure_logging(verbose)

    # Validate inputs
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        raise CalibrationError("P must be a 2D numpy array")
    if P.size == 0:
        raise CalibrationError("P cannot be empty")
    if not np.isfinite(P).all():
        raise CalibrationError("P must not contain NaN or infinite values")
    if np.any(P < 0):
        raise CalibrationError("P must contain non-negative values")

    N, J = P.shape
    if J < 2:
        raise CalibrationError("P must have at least 2 columns (classes)")

    if not isinstance(M, np.ndarray) or M.ndim != 1 or M.size != J:
        raise CalibrationError(f"M must be a 1D array of length {J}")
    if not np.isfinite(M).all():
        raise CalibrationError("M must not contain NaN or infinite values")
    if np.any(M < 0):
        raise CalibrationError("M must contain non-negative values")

    M_sum = float(M.sum())
    if abs(M_sum - N) > 0.1 * N:
        warnings.warn(
            f"Sum of M ({M_sum:.3f}) differs significantly from N ({N}). "
            "IPF may not converge well.",
            UserWarning,
            stacklevel=2,
        )

    P = np.asarray(P, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    Q = P.copy()
    # Ensure non-zero entries for multiplicative updates
    Q = np.maximum(Q, 1e-15)

    converged = False
    final_change = float("inf")
    iteration = 0

    for iteration in range(1, max_iters + 1):
        Q_prev = Q.copy()

        # Row scaling: normalize rows to sum to 1
        row_sums = Q.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-15)
        Q = Q / row_sums

        # Column scaling: scale columns to match target marginals
        col_sums = Q.sum(axis=0)
        col_sums = np.maximum(col_sums, 1e-15)
        scale_factors = M / col_sums
        Q = Q * scale_factors

        # Check convergence
        change_norm = np.linalg.norm(Q - Q_prev)
        Q_norm = np.linalg.norm(Q_prev)
        final_change = float(change_norm / (Q_norm + 1e-15))

        if final_change < tol:
            converged = True
            break

    # Final row normalization
    row_sums = Q.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-15)
    Q = Q / row_sums

    # Compute diagnostics
    row_sums_final = Q.sum(axis=1)
    col_sums_final = Q.sum(axis=0)
    max_row_error = float(np.max(np.abs(row_sums_final - 1.0)))
    max_col_error = float(np.max(np.abs(col_sums_final - M)))

    return IPFResult(
        Q=Q,
        converged=converged,
        iterations=iteration,
        max_row_error=max_row_error,
        max_col_error=max_col_error,
        final_change=final_change,
    )


def calibrate_two_stage(
    P: np.ndarray,
    M: np.ndarray,
    ipf_max_iters: int = 100,
    ipf_tol: float = 1e-8,
    proj_max_iters: int = 100,
    proj_tol: float = 1e-8,
    preserve_marginals: bool = False,
    verbose: bool = False,
) -> TwoStageResult:
    """Two-stage calibration: IPF followed by isotonic projection.

    This approach first uses IPF (raking) to approximately match target marginals
    while preserving the relative probability structure, then applies isotonic
    projection to restore rank ordering.

    The two-stage approach can produce less "flat" solutions compared to
    direct Dykstra projection when distribution shifts are large, because
    IPF preserves probability ratios rather than doing Euclidean projection.

    Args:
        P: Input probability matrix of shape (N, J).
        M: Target column sums of shape (J,). Should sum to N.
        ipf_max_iters: Maximum IPF iterations in stage 1.
        ipf_tol: Convergence tolerance for IPF.
        proj_max_iters: Maximum iterations for isotonic projection in stage 2.
        proj_tol: Convergence tolerance for projection.
        preserve_marginals: If True, re-apply column scaling after isotonic
            projection to maintain marginals (may re-introduce rank violations).
        verbose: If True, enables debug logging.

    Returns:
        TwoStageResult with calibrated matrix and diagnostics.

    Raises:
        CalibrationError: If inputs are invalid.

    Examples:
        >>> import numpy as np
        >>> from rank_preserving_calibration import calibrate_two_stage
        >>> P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
        >>> M = np.array([1.0, 1.2, 0.8])
        >>> result = calibrate_two_stage(P, M)
        >>> print(f"Converged: {result.converged}")
    """
    _configure_logging(verbose)

    # Validate inputs
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        raise CalibrationError("P must be a 2D numpy array")
    if P.size == 0:
        raise CalibrationError("P cannot be empty")

    _N, J = P.shape
    if J < 2:
        raise CalibrationError("P must have at least 2 columns")

    if not isinstance(M, np.ndarray) or M.ndim != 1 or M.size != J:
        raise CalibrationError(f"M must be a 1D array of length {J}")

    P = np.asarray(P, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    # Stage 1: IPF to match marginals
    ipf_result = calibrate_ipf(
        P, M, max_iters=ipf_max_iters, tol=ipf_tol, verbose=verbose
    )
    Q = ipf_result.Q.copy()
    ipf_Q = Q.copy()

    # Precompute column orders from original P
    column_orders = [np.argsort(P[:, j], kind="mergesort") for j in range(J)]

    # Stage 2: Isotonic projection to restore rank ordering
    converged = ipf_result.converged
    proj_iterations = 0

    for proj_iterations in range(1, proj_max_iters + 1):
        Q_prev = Q.copy()

        # Apply isotonic regression per column (maintaining original rank order)
        for j in range(J):
            idx = column_orders[j]
            v_sorted = Q[idx, j]
            iso = _isotonic_regression(v_sorted, rtol=0.0, ties="stable")
            Q[idx, j] = iso

        # Project onto row simplex
        Q = _project_row_simplex(Q)

        # Optionally re-scale columns to preserve marginals
        if preserve_marginals:
            col_sums = Q.sum(axis=0)
            col_sums = np.maximum(col_sums, 1e-15)
            scale_factors = M / col_sums
            Q = Q * scale_factors
            # Re-normalize rows
            row_sums = Q.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-15)
            Q = Q / row_sums

        # Check convergence
        change_norm = np.linalg.norm(Q - Q_prev)
        Q_norm = np.linalg.norm(Q_prev)
        final_change = float(change_norm / (Q_norm + 1e-15))

        if final_change < proj_tol:
            break

    # Compute diagnostics
    row_sums = Q.sum(axis=1)
    col_sums = Q.sum(axis=0)
    max_row_error = float(np.max(np.abs(row_sums - 1.0)))
    max_col_error = float(np.max(np.abs(col_sums - M)))
    max_rank_violation = _compute_rank_violation(Q, P)

    return TwoStageResult(
        Q=Q,
        converged=converged,
        ipf_iterations=ipf_result.iterations,
        projection_iterations=proj_iterations,
        max_row_error=max_row_error,
        max_col_error=max_col_error,
        max_rank_violation=max_rank_violation,
        ipf_result=ipf_Q,
    )
