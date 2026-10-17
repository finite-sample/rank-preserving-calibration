# rank_preserving_calibration/two_stage.py
"""Two-stage calibration approach using Iterative Proportional Fitting (IPF).

This module provides an alternative calibration strategy that may produce
less "flat" solutions when distribution shifts are large:

1. IPF (raking) to match target marginals while preserving relative structure
2. Optional isotonic projection to restore rank ordering

IPF tends to preserve more of the original probability structure compared
to direct Dykstra projection, making it useful when approximate marginal
matching is acceptable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._isotonic import _isotonic_regression
from .calibration import (
    _compute_rank_violation,
    _configure_logging,
    _project_row_simplex,
    _validate_arrays,
    _validate_inputs,
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
        converged: Whether both the relative change and the returned
            matrix's largest column error fell below tol.
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

    Unlike Dykstra, IPF preserves the relative structure of probabilities
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

    Examples:
        >>> import numpy as np
        >>> from rank_preserving_calibration import calibrate_ipf
        >>> P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> M = np.array([0.8, 0.8, 0.4])
        >>> result = calibrate_ipf(P, M)
        >>> result.Q.sum(axis=0).round(6).tolist()
        [0.8, 0.8, 0.4]
    """
    _configure_logging(verbose)

    _validate_inputs(P, M, max_iters, tol, feasibility_tol=0.0)

    P = np.asarray(P, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    Q = P.copy()
    # Ensure non-zero entries for multiplicative updates
    Q = np.maximum(Q, 1e-15)

    converged = False
    final_change = float("inf")
    iteration = 0

    for iteration in range(1, max_iters + 1):  # noqa: B007  # `iteration` is the reported iteration count
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

        # Test the matrix that is returned: the row-normalised iterate. A
        # small change alone would accept the fixed point IPF settles on when
        # sum(M) != N, and a small per-row gap still lets the column error
        # after normalisation build up across N rows.
        Q_out = Q / np.maximum(Q.sum(axis=1, keepdims=True), 1e-15)
        col_gap = float(np.max(np.abs(Q_out.sum(axis=0) - M)))
        if final_change < tol and col_gap < tol:
            converged = True
            break

    Q = Q / np.maximum(Q.sum(axis=1, keepdims=True), 1e-15)

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

    Examples:
        >>> import numpy as np
        >>> from rank_preserving_calibration import calibrate_two_stage
        >>> P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
        >>> M = np.array([1.0, 1.2, 0.8])
        >>> result = calibrate_two_stage(P, M)
        >>> print(f"Converged: {result.converged}")
        Converged: True
    """
    _configure_logging(verbose)

    _N, J = _validate_arrays(P, M)

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

    for proj_iterations in range(1, proj_max_iters + 1):  # noqa: B007  # counter
        Q_prev = Q.copy()

        # Apply isotonic regression per column (maintaining original rank order)
        for j in range(J):
            idx = column_orders[j]
            v_sorted = Q[idx, j]
            iso = _isotonic_regression(v_sorted)
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
