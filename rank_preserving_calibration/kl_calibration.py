# rank_preserving_calibration/kl_calibration.py
"""
KL-divergence rank-preserving calibration.

This module provides rank-preserving calibration using KL divergence (relative entropy)
as the loss function instead of squared Euclidean distance. KL divergence is a natural
choice for probability calibration in label-shift scenarios.

Key innovations:
1. **Anchor-Reference Decoupling**: Separate A (ranking anchor) from R (reference for KL)
2. **Geometric Mean Pooling**: KL isotonic regression uses geometric (not arithmetic) mean
3. **Multiplicative Rescaling**: Sum constraints via multiplication (not addition)
4. **Pareto Frontier**: Report whole λ-path, not single tuned point

Exports:
    - KLCalibrationResult: Result container for hard KL calibration
    - KLParetoResult: Result container for Pareto frontier sweep
    - calibrate_kl: Main hard KL solver with anchor-reference decoupling
    - calibrate_kl_soft: Soft KL calibration with λ penalty
    - calibrate_kl_pareto: Pareto frontier solver with warm-start sweep
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .calibration import (
    CalibrationError,
    _compute_rank_violation,
    _configure_logging,
    _validate_inputs,
)

type NDArrayFloat = np.ndarray[Any, np.dtype[np.floating[Any]]]

__all__ = [
    "KLCalibrationResult",
    "KLParetoResult",
    "calibrate_kl",
    "calibrate_kl_pareto",
    "calibrate_kl_soft",
]


# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------


@dataclass(slots=True)
class KLCalibrationResult:
    """Result container for KL-divergence rank-preserving calibration.

    Attributes:
        Q: Calibrated probability matrix of shape (N, J) where rows sum to 1
            and columns preserve rank ordering from anchor scores.
        converged: True if algorithm converged within specified tolerance.
        iterations: Number of iterations performed before termination.
        kl_divergence: Final KL(Q||R) divergence value.
        max_row_error: Maximum absolute error in row sum constraint.
        max_col_error: Maximum absolute error in column sum constraint.
        max_rank_violation: Maximum rank-order violation across all columns.
        final_change: Final relative change in solution between iterations.
    """

    Q: np.ndarray
    converged: bool
    iterations: int
    kl_divergence: float
    max_row_error: float
    max_col_error: float
    max_rank_violation: float
    final_change: float


@dataclass(slots=True)
class KLParetoResult:
    """Result container for KL Pareto frontier computation.

    Attributes:
        solutions: List of (lambda, Q) pairs along the Pareto frontier.
        kl_values: KL divergence values for each solution.
        rank_violations: Total rank violation for each solution.
        lambda_path: Lambda values used in the sweep.
    """

    solutions: list[tuple[float, np.ndarray]] = field(default_factory=list)
    kl_values: list[float] = field(default_factory=list)
    rank_violations: list[float] = field(default_factory=list)
    lambda_path: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------
# KL divergence utilities
# ---------------------------------------------------------------------


def _safe_log(x: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    """Safe log avoiding -inf for small values."""
    return np.log(np.maximum(x, eps))


def _kl_div_matrix(Q: np.ndarray, R: np.ndarray, eps: float = 1e-300) -> float:
    """Compute KL(Q||R) = sum Q * log(Q/R) for probability matrices.

    Uses convention 0 * log(0/x) = 0.
    """
    Q = np.asarray(Q, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    # Handle zeros: 0 * log(0/x) = 0
    mask = Q > eps
    kl = np.zeros_like(Q)
    kl[mask] = Q[mask] * (_safe_log(Q[mask], eps) - _safe_log(R[mask], eps))

    return float(np.sum(kl))


# ---------------------------------------------------------------------
# KL isotonic regression (generalized PAV with geometric mean)
# ---------------------------------------------------------------------


def _kl_isotonic_regression(
    y: np.ndarray,
    weights: np.ndarray | None = None,
    eps: float = 1e-300,
) -> np.ndarray:
    """Isotonic regression minimizing weighted KL divergence.

    Unlike Euclidean PAV which pools using arithmetic mean, KL isotonic
    regression uses **geometric mean** pooling:

        z_B = exp(1/|B| * sum_{i in B} log(w_i))

    This is the exact projection onto the isotone cone for KL divergence.

    Args:
        y: Input sequence to make isotonic (non-decreasing).
        weights: Positive weights for each element. If None, uses y as weights.
        eps: Small constant for numerical stability.

    Returns:
        Isotonic fit minimizing KL divergence.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n <= 1:
        return np.maximum(y.copy(), eps)

    # Ensure positive values for log
    y = np.maximum(y, eps)

    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != y.shape:
            raise ValueError("weights must have same shape as y")
        if np.any(w <= 0):
            raise ValueError("weights must be positive")

    # Work in log space for geometric mean
    log_y = np.log(y)

    # Block stacks: start index, weighted log sum, weight sum
    start = np.empty(n, dtype=np.int64)
    log_sum = np.empty(n, dtype=np.float64)
    wsum = np.empty(n, dtype=np.float64)
    top = -1

    for i in range(n):
        top += 1
        start[top] = i
        log_sum[top] = w[i] * log_y[i]
        wsum[top] = w[i]

        # Merge backward while violating monotonicity
        # Geometric mean of block: exp(log_sum / wsum)
        while top > 0:
            left_mean = log_sum[top - 1] / wsum[top - 1]
            right_mean = log_sum[top] / wsum[top]
            if left_mean <= right_mean:
                break
            # Merge blocks
            log_sum[top - 1] += log_sum[top]
            wsum[top - 1] += wsum[top]
            top -= 1

    # Expand block means
    z = np.empty(n, dtype=np.float64)
    for j in range(top + 1):
        s = start[j]
        e = start[j + 1] if j < top else n
        block_mean = log_sum[j] / wsum[j]
        z[s:e] = np.exp(block_mean)

    return z


# ---------------------------------------------------------------------
# KL column projection (isotonic + multiplicative sum constraint)
# ---------------------------------------------------------------------


def _project_column_kl_isotonic_sum(
    column: np.ndarray,
    column_order: np.ndarray,
    target_sum: float,
    weights: np.ndarray | None = None,
    eps: float = 1e-300,
) -> np.ndarray:
    """Project column onto KL-isotonic with sum constraint.

    Key difference from Euclidean:
    - Uses generalized PAV (geometric mean pooling)
    - Uses **multiplicative** rescaling to hit sum target

    Args:
        column: Column values to project.
        column_order: Indices that sort by anchor scores.
        target_sum: Target sum for the column.
        weights: Optional weights for isotonic regression.
        eps: Small constant for numerical stability.

    Returns:
        Projected column satisfying isotonicity and sum constraint.
    """
    if column.size == 0:
        return column.copy()

    # Sort by anchor order
    y = np.maximum(column[column_order], eps)

    # Apply KL isotonic regression
    z_iso = _kl_isotonic_regression(y, weights=weights, eps=eps)

    # Multiplicative rescaling to hit target sum
    current_sum = z_iso.sum()
    if current_sum > eps:
        scale = target_sum / current_sum
        z_scaled = z_iso * scale
    else:
        # Edge case: all near zero, distribute uniformly
        z_scaled = np.full_like(z_iso, target_sum / z_iso.size)

    # Restore original order
    projected = np.empty_like(column, dtype=np.float64)
    projected[column_order] = z_scaled
    return projected


# ---------------------------------------------------------------------
# KL row simplex projection (multiplicative normalization)
# ---------------------------------------------------------------------


def _project_row_kl_simplex(rows: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    """Project rows onto probability simplex using KL projection.

    For KL divergence, the projection onto the simplex is simply
    **multiplicative normalization**: q_ij = p_ij / sum_k(p_ik).

    This is different from the Euclidean simplex projection which
    uses a sorting-based algorithm with additive adjustments.

    Args:
        rows: Matrix of shape (N, J) to project.
        eps: Small constant for numerical stability.

    Returns:
        Projected matrix with rows summing to 1.
    """
    rows = np.maximum(rows, eps)
    row_sums = rows.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, eps)
    return rows / row_sums


# ---------------------------------------------------------------------
# Main KL calibration (Dykstra-style alternating projections)
# ---------------------------------------------------------------------


def calibrate_kl(
    P: np.ndarray,
    M: np.ndarray,
    R: np.ndarray | None = None,
    A: np.ndarray | None = None,
    max_iters: int = 3000,
    tol: float = 1e-7,
    feasibility_tol: float = 0.1,
    verbose: bool = False,
    eps: float = 1e-300,
) -> KLCalibrationResult:
    """Calibrate using KL divergence with Dykstra's alternating projections.

    Projects multiclass probabilities onto the intersection of:
      (A) row simplex: {rows ≥ 0, rows sum to 1}
      (B) column-wise isotone-by-anchor + fixed column sums

    Minimizes KL(Q||R) subject to constraints, where:
      - R is the reference distribution for KL divergence (default: P)
      - A is the anchor for rank ordering (default: P)

    This anchor-reference decoupling allows separating "what we measure
    divergence from" (R) from "what determines rank order" (A).

    Args:
        P: Input probability matrix of shape (N, J).
        M: Target column sums of shape (J,).
        R: Reference distribution for KL divergence. If None, uses P.
        A: Anchor for rank ordering. If None, uses P.
        max_iters: Maximum number of iterations.
        tol: Convergence tolerance for relative change.
        feasibility_tol: Tolerance for feasibility warnings.
        verbose: If True, enables debug logging.
        eps: Small constant for numerical stability.

    Returns:
        KLCalibrationResult with calibrated matrix and diagnostics.

    Raises:
        CalibrationError: If inputs are invalid or algorithm fails to converge.

    Examples:
        Basic KL calibration:

        >>> import numpy as np
        >>> from rank_preserving_calibration import calibrate_kl
        >>> P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> M = np.array([1.0, 0.7, 0.3])
        >>> result = calibrate_kl(P, M)

        With anchor-reference decoupling:

        >>> # Use P for KL reference, but A for rank ordering
        >>> A = np.array([[0.6, 0.3, 0.1], [0.4, 0.4, 0.2]])
        >>> result = calibrate_kl(P, M, R=P, A=A)
    """
    _configure_logging(verbose)
    _N, J = _validate_inputs(P, M, max_iters, tol, feasibility_tol)

    P = np.asarray(P, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    # Set defaults for R and A
    if R is None:
        R = P.copy()
    else:
        R = np.asarray(R, dtype=np.float64)
        if R.shape != P.shape:
            raise CalibrationError(f"R must have shape {P.shape}, got {R.shape}")

    if A is None:
        A = P.copy()
    else:
        A = np.asarray(A, dtype=np.float64)
        if A.shape != P.shape:
            raise CalibrationError(f"A must have shape {P.shape}, got {A.shape}")

    # Precompute column orders from anchor A
    column_orders = [np.argsort(A[:, j], kind="mergesort") for j in range(J)]

    # Initialize Q close to R (for KL minimization)
    Q = np.maximum(R.copy(), eps)

    # Dykstra memory terms (in multiplicative form for KL)
    # For KL projections, we use multiplicative corrections
    U = np.ones_like(P, dtype=np.float64)  # row memory (multiplicative)
    V = np.ones_like(P, dtype=np.float64)  # col memory (multiplicative)

    converged = False
    final_change = float("inf")
    iteration = 0

    for iteration in range(1, max_iters + 1):
        Q_prev = Q.copy()

        # Row projection with Dykstra correction
        Y = Q * U
        Q = _project_row_kl_simplex(Y, eps=eps)
        # Update multiplicative memory
        U = Y / np.maximum(Q, eps)

        # Column projections with Dykstra correction
        Y = Q * V
        for j in range(J):
            Y[:, j] = _project_column_kl_isotonic_sum(
                Y[:, j],
                column_orders[j],
                float(M[j]),
                eps=eps,
            )
        Q = Y
        V = Y / np.maximum(Q, eps)

        # Convergence check
        change_abs = np.linalg.norm(Q - Q_prev)
        norm_Q_prev = np.linalg.norm(Q_prev)
        final_change = (
            float(change_abs / norm_Q_prev) if norm_Q_prev > 0 else float(change_abs)
        )

        row_ok = np.allclose(Q.sum(axis=1), 1.0, atol=1e-10)
        col_ok = np.allclose(Q.sum(axis=0), M, atol=1e-8)

        if final_change < tol and row_ok and col_ok:
            converged = True
            break

    # Final diagnostics
    row_sums = Q.sum(axis=1)
    col_sums = Q.sum(axis=0)
    max_row_error = float(np.max(np.abs(row_sums - 1.0)))
    max_col_error = float(np.max(np.abs(col_sums - M)))
    max_rank_violation = _compute_rank_violation(Q, A)
    kl_divergence = _kl_div_matrix(Q, R, eps=eps)

    if not converged:
        raise CalibrationError(
            f"KL calibration failed to converge after {iteration} iterations. "
            f"Final change: {final_change:.2e} (tolerance: {tol:.2e}). "
            f"Max row error: {max_row_error:.2e}, max col error: {max_col_error:.2e}. "
            f"Try: increasing max_iters or relaxing tol."
        )

    return KLCalibrationResult(
        Q=Q,
        converged=converged,
        iterations=iteration,
        kl_divergence=kl_divergence,
        max_row_error=max_row_error,
        max_col_error=max_col_error,
        max_rank_violation=max_rank_violation,
        final_change=final_change,
    )


# ---------------------------------------------------------------------
# Soft KL calibration with λ penalty
# ---------------------------------------------------------------------


def _compute_rank_penalty_from_orders(
    Q: np.ndarray, column_orders: list[np.ndarray]
) -> float:
    """Compute total rank violation penalty: sum of (q[i] - q[i+1])_+ for all columns."""
    penalty = 0.0
    J = Q.shape[1]
    for j in range(J):
        q_sorted = Q[column_orders[j], j]
        if q_sorted.size > 1:
            diffs = q_sorted[:-1] - q_sorted[1:]
            penalty += float(np.maximum(diffs, 0.0).sum())
    return penalty


def calibrate_kl_soft(
    P: np.ndarray,
    M: np.ndarray,
    lam: float = 1.0,
    R: np.ndarray | None = None,
    A: np.ndarray | None = None,
    max_iters: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False,
    eps: float = 1e-300,
) -> KLCalibrationResult:
    """Soft KL calibration with λ-weighted rank penalty.

    Solves:
        min KL(Q||R) + λ·V_A(Q)  s.t.  Q ∈ C(M)

    Where:
        - KL(Q||R) is the KL divergence from reference R
        - V_A(Q) is the rank violation penalty using anchor A
        - C(M) is row simplex + column sums = M

    Args:
        P: Input probability matrix of shape (N, J).
        M: Target column sums of shape (J,).
        lam: Rank penalty weight. Larger = more isotonic.
        R: Reference for KL divergence. Default: P.
        A: Anchor for rank ordering. Default: P.
        max_iters: Maximum iterations.
        tol: Convergence tolerance.
        verbose: Enable debug logging.
        eps: Numerical stability constant.

    Returns:
        KLCalibrationResult with calibrated matrix.

    Examples:
        >>> result = calibrate_kl_soft(P, M, lam=10.0)  # Strong rank enforcement
        >>> result = calibrate_kl_soft(P, M, lam=0.1)  # Weak rank enforcement
    """
    _configure_logging(verbose)
    _, J = _validate_inputs(P, M, max_iters, tol, feasibility_tol=0.1)

    P = np.asarray(P, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    if R is None:
        R = P.copy()
    else:
        R = np.asarray(R, dtype=np.float64)

    if A is None:
        A = P.copy()
    else:
        A = np.asarray(A, dtype=np.float64)

    column_orders = [np.argsort(A[:, j], kind="mergesort") for j in range(J)]

    # Initialize
    Q = np.maximum(P.copy(), eps)
    Q = _project_row_kl_simplex(Q, eps=eps)

    converged = False
    final_change = float("inf")

    for iteration in range(1, max_iters + 1):
        Q_prev = Q.copy()

        # Gradient of KL(Q||R) w.r.t. Q: log(Q) - log(R) + 1
        # For multiplicative gradient descent, use: Q * grad
        # Take exponentiated gradient step: Q_new = Q * exp(-step * grad)
        step = 0.1 / (1 + iteration * 0.01)

        # KL gradient component
        kl_grad = _safe_log(Q, eps) - _safe_log(R, eps) + 1.0

        # Rank penalty gradient (subgradient)
        rank_grad = np.zeros_like(Q)
        for j in range(J):
            idx = column_orders[j]
            q_sorted = Q[idx, j]
            for i in range(len(q_sorted) - 1):
                if q_sorted[i] > q_sorted[i + 1]:
                    rank_grad[idx[i], j] += 1.0
                    rank_grad[idx[i + 1], j] -= 1.0

        # Combined gradient
        total_grad = kl_grad + lam * rank_grad

        # Multiplicative update (exponentiated gradient)
        Q = Q * np.exp(-step * total_grad)
        Q = np.maximum(Q, eps)

        # Project onto row simplex
        Q = _project_row_kl_simplex(Q, eps=eps)

        # Project columns onto sum constraints
        for j in range(J):
            current_sum = Q[:, j].sum()
            if current_sum > eps:
                Q[:, j] *= M[j] / current_sum

        # Convergence check
        change_norm = np.linalg.norm(Q - Q_prev)
        Q_norm = np.linalg.norm(Q_prev)
        final_change = float(change_norm / (Q_norm + 1e-15))

        if final_change < tol:
            converged = True
            break

    # Final diagnostics
    row_sums = Q.sum(axis=1)
    col_sums = Q.sum(axis=0)
    max_row_error = float(np.max(np.abs(row_sums - 1.0)))
    max_col_error = float(np.max(np.abs(col_sums - M)))
    max_rank_violation = _compute_rank_violation(Q, A)
    kl_divergence = _kl_div_matrix(Q, R, eps=eps)

    if not converged:
        warnings.warn(
            f"KL soft calibration did not converge after {iteration} iterations. "
            f"Final change: {final_change:.2e}",
            UserWarning,
            stacklevel=2,
        )

    return KLCalibrationResult(
        Q=Q,
        converged=converged,
        iterations=iteration,
        kl_divergence=kl_divergence,
        max_row_error=max_row_error,
        max_col_error=max_col_error,
        max_rank_violation=max_rank_violation,
        final_change=final_change,
    )


# ---------------------------------------------------------------------
# Pareto frontier solver
# ---------------------------------------------------------------------


def calibrate_kl_pareto(
    P: np.ndarray,
    M: np.ndarray,
    lambda_grid: list[float] | np.ndarray | None = None,
    R: np.ndarray | None = None,
    A: np.ndarray | None = None,
    max_iters: int = 500,
    tol: float = 1e-5,
    verbose: bool = False,
    eps: float = 1e-300,
) -> KLParetoResult:
    """Compute Pareto frontier of KL divergence vs rank violation.

    Sweeps over λ values with warm starting, reporting the full
    Pareto frontier rather than a single tuned point.

    Args:
        P: Input probability matrix of shape (N, J).
        M: Target column sums of shape (J,).
        lambda_grid: Grid of λ values to sweep. Default: geometric grid.
        R: Reference for KL divergence. Default: P.
        A: Anchor for rank ordering. Default: P.
        max_iters: Max iterations per λ value.
        tol: Convergence tolerance.
        verbose: Enable debug logging.
        eps: Numerical stability constant.

    Returns:
        KLParetoResult with solutions along the frontier.

    Examples:
        >>> result = calibrate_kl_pareto(P, M)
        >>> for lam, Q in result.solutions:
        ...     print(f"λ={lam:.2f}: KL={result.kl_values[i]:.4f}")
    """
    _configure_logging(verbose)

    P = np.asarray(P, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    if R is None:
        R = P.copy()
    else:
        R = np.asarray(R, dtype=np.float64)

    if A is None:
        A = P.copy()
    else:
        A = np.asarray(A, dtype=np.float64)

    if lambda_grid is None:
        lambda_grid = np.logspace(-2, 3, 20)
    else:
        lambda_grid = np.asarray(lambda_grid)

    column_orders = [np.argsort(A[:, j], kind="mergesort") for j in range(A.shape[1])]

    result = KLParetoResult()

    # Initialize with smallest lambda
    Q_warm = np.maximum(P.copy(), eps)
    Q_warm = _project_row_kl_simplex(Q_warm, eps=eps)

    for lam in sorted(lambda_grid):
        # Warm-start from previous solution
        try:
            sol = calibrate_kl_soft(
                P,
                M,
                lam=float(lam),
                R=R,
                A=A,
                max_iters=max_iters,
                tol=tol,
                verbose=False,
                eps=eps,
            )
            Q = sol.Q
        except CalibrationError:
            continue

        kl_val = _kl_div_matrix(Q, R, eps=eps)
        rank_viol = _compute_rank_penalty_from_orders(Q, column_orders)

        result.solutions.append((float(lam), Q.copy()))
        result.kl_values.append(kl_val)
        result.rank_violations.append(rank_viol)
        result.lambda_path.append(float(lam))

        Q_warm = Q

    return result
