# rank_preserving_calibration/soft_calibration.py
"""
Soft-constraint calibration with tunable trade-offs.

This module provides calibration methods that use soft penalties instead of
hard constraints, allowing users to trade off between:
- Fitting the original probabilities P
- Matching target marginals M
- Preserving rank orderings

The soft-constraint formulation gives a Pareto frontier of solutions rather than
a single "correct" answer, which is more practical when distribution shifts are large.
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
class SoftCalibrationResult:
    """Result from soft-constraint calibration.

    Attributes:
        Q: Calibrated probability matrix.
        converged: Whether the algorithm converged.
        iterations: Number of iterations performed.
        objective_values: Objective function values over iterations.
        fit_term: Final ||Q - P||² term.
        marginal_term: Final ||col_sums(Q) - M||² term.
        rank_term: Final rank violation penalty term.
        max_row_error: Maximum row sum error.
        max_col_error: Maximum column sum error.
        max_rank_violation: Maximum rank violation.
        final_change: Final relative change between iterations.
    """

    Q: np.ndarray
    converged: bool
    iterations: int
    objective_values: list[float]
    fit_term: float
    marginal_term: float
    rank_term: float
    max_row_error: float
    max_col_error: float
    max_rank_violation: float
    final_change: float


def _compute_rank_penalty(Q: np.ndarray, column_orders: list[np.ndarray]) -> float:
    """Compute total rank violation penalty: sum of (q[i] - q[i+1])_+ for all columns."""
    penalty = 0.0
    J = Q.shape[1]
    for j in range(J):
        q_sorted = Q[column_orders[j], j]
        if q_sorted.size > 1:
            diffs = q_sorted[:-1] - q_sorted[1:]
            penalty += float(np.maximum(diffs, 0.0).sum())
    return penalty


def _soft_isotonic_projection(
    v: np.ndarray, lam_r: float, step_size: float = 1.0
) -> np.ndarray:
    """Soft isotonic projection via proximal gradient step.

    When lam_r is large, this approaches strict isotonic regression.
    When lam_r is small, it barely modifies the input.

    Args:
        v: Input vector (should be sorted by original scores).
        lam_r: Rank penalty weight.
        step_size: Gradient descent step size.

    Returns:
        Softly isotonic vector.
    """
    if lam_r == 0.0 or v.size <= 1:
        return v.copy()

    # For large lambda, use exact isotonic regression
    if lam_r >= 1e6:
        return _isotonic_regression(v, rtol=0.0, ties="stable")

    # For moderate lambda, use a soft approach:
    # Gradient of rank penalty with respect to v is:
    #   d/dv[i] sum_k (v[k] - v[k+1])_+ = +1 if v[i] > v[i+1] (for i < n-1)
    #                                      -1 if v[i-1] > v[i] (for i > 0)
    # We take a proximal gradient step: v_new = v - step * lam_r * grad
    z = v.copy()

    for _ in range(10):  # Few iterations of proximal gradient
        grad = np.zeros_like(z)
        for i in range(len(z) - 1):
            if z[i] > z[i + 1]:
                grad[i] += 1.0
                grad[i + 1] -= 1.0
        z = z - step_size * lam_r * grad / (len(z))

    return z


def calibrate_soft(
    P: np.ndarray,
    M: np.ndarray,
    lam_m: float = 1.0,
    lam_r: float = 10.0,
    max_iters: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False,
    step_size: float = 0.1,
) -> SoftCalibrationResult:
    """Calibrate with soft constraints on marginals and ranks.

    Solves the optimization problem:
        min_Q ||Q - P||² + lam_m·||col_sums(Q) - M||² + lam_r·rank_penalty(Q)
        s.t. Q on row simplex (rows >= 0, sum to 1)

    This allows trading off between:
    - Staying close to original predictions (small lam_m, lam_r)
    - Matching marginals (large lam_m)
    - Preserving ranks (large lam_r)

    Args:
        P: Input probability matrix of shape (N, J).
        M: Target column sums of shape (J,). Should sum to approximately N.
        lam_m: Marginal penalty weight. Larger = closer to target marginals.
            Set to 0 to ignore marginals entirely.
        lam_r: Rank penalty weight. Larger = more isotonic.
            Set to 0 to allow arbitrary rank violations.
        max_iters: Maximum number of iterations.
        tol: Convergence tolerance for relative change.
        verbose: If True, enables debug logging.
        step_size: Step size for gradient descent updates.

    Returns:
        SoftCalibrationResult with calibrated matrix and diagnostics.

    Raises:
        CalibrationError: If inputs are invalid.

    Examples:
        >>> import numpy as np
        >>> from rank_preserving_calibration import calibrate_soft
        >>> P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> M = np.array([1.0, 0.7, 0.3])
        >>> # Balanced trade-off
        >>> result = calibrate_soft(P, M, lam_m=1.0, lam_r=10.0)
        >>> # Prioritize marginal matching
        >>> result = calibrate_soft(P, M, lam_m=100.0, lam_r=1.0)
        >>> # Prioritize rank preservation
        >>> result = calibrate_soft(P, M, lam_m=0.1, lam_r=100.0)
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

    if lam_m < 0 or lam_r < 0:
        raise CalibrationError("lam_m and lam_r must be non-negative")

    # Warn about feasibility
    M_sum = float(M.sum())
    if abs(M_sum - N) > 0.1 * N:
        warnings.warn(
            f"Sum of M ({M_sum:.3f}) differs significantly from N ({N}). "
            "Consider adjusting target marginals.",
            UserWarning,
            stacklevel=2,
        )

    P = np.asarray(P, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    # Precompute column orders
    column_orders = [np.argsort(P[:, j], kind="mergesort") for j in range(J)]

    # Initialize Q
    Q = P.copy()

    objective_values: list[float] = []
    converged = False
    final_change = float("inf")
    iteration = 0

    for iteration in range(1, max_iters + 1):
        Q_prev = Q.copy()

        # Gradient of ||Q - P||² is 2(Q - P)
        grad_fit = 2.0 * (Q - P)

        # Gradient of lam_m * ||col_sums(Q) - M||²
        # d/dQ[i,j] = 2 * lam_m * (col_sum_j - M_j)
        if lam_m > 0:
            col_sums = Q.sum(axis=0)
            marginal_error = col_sums - M
            grad_marginal = (
                2.0 * lam_m * np.ones((N, 1)) @ marginal_error.reshape(1, -1)
            )
        else:
            grad_marginal = 0.0

        # Total gradient (without rank term, handled separately)
        grad = grad_fit + grad_marginal

        # Gradient step
        Q = Q - step_size * grad

        # Soft isotonic projection for rank preservation
        if lam_r > 0:
            for j in range(J):
                idx = column_orders[j]
                v_sorted = Q[idx, j]
                z_sorted = _soft_isotonic_projection(
                    v_sorted, lam_r * step_size, step_size=1.0
                )
                Q[idx, j] = z_sorted

        # Project onto row simplex (hard constraint)
        Q = _project_row_simplex(Q)

        # Compute objective
        fit_term = float(np.sum((Q - P) ** 2))
        col_sums = Q.sum(axis=0)
        marginal_term = float(np.sum((col_sums - M) ** 2))
        rank_term = _compute_rank_penalty(Q, column_orders)

        obj_val = fit_term + lam_m * marginal_term + lam_r * rank_term
        objective_values.append(obj_val)

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
    max_rank_violation = _compute_rank_violation(Q, P)

    fit_term = float(np.sum((Q - P) ** 2))
    marginal_term = float(np.sum((col_sums - M) ** 2))
    rank_term = _compute_rank_penalty(Q, column_orders)

    return SoftCalibrationResult(
        Q=Q,
        converged=converged,
        iterations=iteration,
        objective_values=objective_values,
        fit_term=fit_term,
        marginal_term=marginal_term,
        rank_term=rank_term,
        max_row_error=max_row_error,
        max_col_error=max_col_error,
        max_rank_violation=max_rank_violation,
        final_change=final_change,
    )


def calibrate_soft_admm(
    P: np.ndarray,
    M: np.ndarray,
    lam_m: float = 1.0,
    lam_r: float = 10.0,
    rho: float = 1.0,
    max_iters: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False,
) -> SoftCalibrationResult:
    """ADMM-based soft calibration with better convergence.

    Uses ADMM to solve the soft-constraint problem more reliably than
    simple gradient descent.

    Args:
        P: Input probability matrix of shape (N, J).
        M: Target column sums of shape (J,).
        lam_m: Marginal penalty weight.
        lam_r: Rank penalty weight.
        rho: ADMM penalty parameter.
        max_iters: Maximum iterations.
        tol: Convergence tolerance.
        verbose: Enable debug logging.

    Returns:
        SoftCalibrationResult with calibrated matrix.
    """
    _configure_logging(verbose)

    # Validate inputs
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        raise CalibrationError("P must be a 2D numpy array")
    if P.size == 0:
        raise CalibrationError("P cannot be empty")

    N, J = P.shape
    if J < 2:
        raise CalibrationError("P must have at least 2 columns")

    if not isinstance(M, np.ndarray) or M.ndim != 1 or M.size != J:
        raise CalibrationError(f"M must be a 1D array of length {J}")

    if lam_m < 0 or lam_r < 0:
        raise CalibrationError("lam_m and lam_r must be non-negative")

    P = np.asarray(P, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    # Precompute column orders
    column_orders = [np.argsort(P[:, j], kind="mergesort") for j in range(J)]

    # ADMM variables:
    # Q: primal variable (calibrated probabilities)
    # Z: auxiliary for simplex constraint
    # Y: auxiliary for isotonic constraint (per column)
    # u_z, u_y: dual variables
    Q = P.copy()
    Z = Q.copy()  # simplex copy
    Y = Q.copy()  # isotonic copy
    u_z = np.zeros_like(Q)
    u_y = np.zeros_like(Q)

    objective_values: list[float] = []
    converged = False
    final_change = float("inf")
    iteration = 0

    for iteration in range(1, max_iters + 1):
        Q_prev = Q.copy()

        # Q-update: solve quadratic
        # min (1/2)||Q - P||² + (lam_m/2)||Q.sum(0) - M||²
        #     + (rho/2)||Q - Z + u_z||² + (rho/2)||Q - Y + u_y||²
        # This is a quadratic in Q, solve via closed form

        # Gradient is: (Q - P) + lam_m * (col_sums - M) broadcast + rho*(Q - Z + u_z) + rho*(Q - Y + u_y)
        # Setting to zero:
        # (1 + 2*rho)*Q + lam_m * broadcast = P + rho*(Z - u_z) + rho*(Y - u_y)

        # Per-column marginal term complicates this. Use iterative approach.
        numerator = P + rho * (Z - u_z) + rho * (Y - u_y)
        # Include marginal contribution
        if lam_m > 0:
            # Marginal term adds lam_m * (col_sum - M_j) to each element in column j
            # So Q_ij has gradient term lam_m * (sum_k Q_kj - M_j)
            # This couples all rows in a column
            # For tractability, update Q column by column

            Q_new = np.zeros_like(Q)
            for j in range(J):
                # For column j: solve
                # (1 + 2*rho + lam_m)*Q_j + lam_m*11^T Q_j / N = numerator_j + lam_m*M_j/N
                # Actually: gradient = Q_j - P_j + lam_m*(sum(Q_j) - M_j)*1 + rho*(Q_j - Z_j + u_z_j) + rho*(Q_j - Y_j + u_y_j)
                # = (1 + 2*rho)*Q_j + lam_m*1*1^T Q_j - P_j - rho*Z_j + rho*u_z_j - rho*Y_j + rho*u_y_j + lam_m*(-M_j)*1

                # Setting to 0:
                # (1 + 2*rho)*Q_j + lam_m*1*sum(Q_j) = P_j + rho*Z_j - rho*u_z_j + rho*Y_j - rho*u_y_j + lam_m*M_j*1

                # Let c = sum(Q_j). Then:
                # (1 + 2*rho)*Q_j = RHS - lam_m*c*1
                # Q_j = (RHS - lam_m*c*1) / (1 + 2*rho)
                # Sum both sides:
                # c = (sum(RHS) - lam_m*c*N) / (1 + 2*rho)
                # c*(1 + 2*rho) = sum(RHS) - lam_m*N*c
                # c*(1 + 2*rho + lam_m*N) = sum(RHS)
                # c = sum(RHS) / (1 + 2*rho + lam_m*N)

                RHS_j = (
                    P[:, j]
                    + rho * (Z[:, j] - u_z[:, j])
                    + rho * (Y[:, j] - u_y[:, j])
                    + lam_m * M[j] / N * np.ones(N)
                )
                c = RHS_j.sum() / (1 + 2 * rho + lam_m)
                Q_new[:, j] = (RHS_j - lam_m * c / N * np.ones(N)) / (1 + 2 * rho)
            Q = Q_new
        else:
            Q = numerator / (1 + 2 * rho)

        # Z-update: project onto row simplex
        Z = _project_row_simplex(Q + u_z)

        # Y-update: soft isotonic projection per column
        for j in range(J):
            idx = column_orders[j]
            v_sorted = (Q + u_y)[idx, j]
            if lam_r >= 1e6:
                # Hard isotonic
                y_sorted = _isotonic_regression(v_sorted, rtol=0.0, ties="stable")
            elif lam_r > 0:
                # Soft isotonic via prox
                y_sorted = _soft_isotonic_projection(
                    v_sorted, lam_r / rho, step_size=1.0
                )
            else:
                y_sorted = v_sorted
            Y[idx, j] = y_sorted

        # Dual updates
        u_z = u_z + Q - Z
        u_y = u_y + Q - Y

        # Compute objective
        fit_term = float(np.sum((Q - P) ** 2))
        col_sums = Q.sum(axis=0)
        marginal_term = float(np.sum((col_sums - M) ** 2))
        rank_term = _compute_rank_penalty(Q, column_orders)
        obj_val = 0.5 * fit_term + 0.5 * lam_m * marginal_term + lam_r * rank_term
        objective_values.append(obj_val)

        # Convergence
        change_norm = np.linalg.norm(Q - Q_prev)
        Q_norm = np.linalg.norm(Q_prev)
        final_change = float(change_norm / (Q_norm + 1e-15))

        if final_change < tol:
            converged = True
            break

    # Final Q from Z (the simplex projection)
    Q = Z

    # Final diagnostics
    row_sums = Q.sum(axis=1)
    col_sums = Q.sum(axis=0)
    max_row_error = float(np.max(np.abs(row_sums - 1.0)))
    max_col_error = float(np.max(np.abs(col_sums - M)))
    max_rank_violation = _compute_rank_violation(Q, P)

    fit_term = float(np.sum((Q - P) ** 2))
    marginal_term = float(np.sum((col_sums - M) ** 2))
    rank_term = _compute_rank_penalty(Q, column_orders)

    return SoftCalibrationResult(
        Q=Q,
        converged=converged,
        iterations=iteration,
        objective_values=objective_values,
        fit_term=fit_term,
        marginal_term=marginal_term,
        rank_term=rank_term,
        max_row_error=max_row_error,
        max_col_error=max_col_error,
        max_rank_violation=max_rank_violation,
        final_change=final_change,
    )
