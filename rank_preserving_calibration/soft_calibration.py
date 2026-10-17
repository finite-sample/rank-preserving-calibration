# rank_preserving_calibration/soft_calibration.py
"""Soft-constraint calibration with tunable trade-offs.

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
    _project_row_simplex,
    _validate_arrays,
)
from .nearly import prox_near_isotonic


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
    """Sum the rank violations (q[i] - q[i+1])_+ over every column."""
    penalty = 0.0
    J = Q.shape[1]
    for j in range(J):
        q_sorted = Q[column_orders[j], j]
        if q_sorted.size > 1:
            diffs = q_sorted[:-1] - q_sorted[1:]
            penalty += float(np.maximum(diffs, 0.0).sum())
    return penalty


def calibrate_soft(
    P: np.ndarray,
    M: np.ndarray,
    lam_m: float = 1.0,
    lam_r: float = 10.0,
    *,
    rho: float = 1.0,
    max_iters: int = 5000,
    tol: float = 1e-8,
    verbose: bool = False,
) -> SoftCalibrationResult:
    """Calibrate with soft penalties on the marginals and on rank violations.

    Solves the convex problem::

        minimise  0.5 ||Q - P||^2 + 0.5 lam_m ||Q^T 1 - M||^2
                  + lam_r * sum_j sum_i (q_(i),j - q_(i+1),j)_+
        subject to each row of Q on the probability simplex,

    where q_(i),j is column j in the order of P's column j, so the last term
    charges every adjacent downward step. Rows stay exact probability
    distributions; the marginals and the ranking are traded off against
    staying close to P.

    By ADMM on the split Q = Z (row simplex, exact projection) and Q = Y (rank
    penalty, exact prox of the nearly-isotonic term per column).

    Args:
        P: Input probability matrix of shape (N, J).
        M: Target column sums of shape (J,).
        lam_m: Marginal penalty weight. 0 ignores the marginals.
        lam_r: Rank penalty weight. 0 ignores the ranking; large values
            approach strict isotonicity.
        rho: ADMM penalty parameter.
        max_iters: Maximum number of iterations.
        tol: Tolerance on the primal and dual residuals, relative to the
            iterate's norm.
        verbose: If True, enables debug logging.

    Returns:
        SoftCalibrationResult with the calibrated matrix, taken from the
        simplex block so that every row is a probability distribution.

    Raises:
        CalibrationError: If P or M are malformed, or a weight is negative.

    Examples:
        >>> import numpy as np
        >>> from rank_preserving_calibration import calibrate_soft
        >>> P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> M = np.array([0.8, 0.8, 0.4])
        >>> result = calibrate_soft(P, M, lam_m=100.0, lam_r=1.0)
        >>> result.converged
        True
        >>> np.round(result.Q.sum(axis=0), 2).tolist()
        [0.8, 0.8, 0.4]
    """
    _configure_logging(verbose)

    N, J = _validate_arrays(P, M)

    if lam_m < 0 or lam_r < 0 or rho <= 0:
        raise CalibrationError("lam_m and lam_r must be non-negative and rho positive")

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
    column_orders = [np.argsort(P[:, j], kind="mergesort") for j in range(J)]

    Q = P.copy()
    Z = P.copy()
    Y = P.copy()
    u_z = np.zeros_like(P)
    u_y = np.zeros_like(P)

    objective_values: list[float] = []
    converged = False
    final_change = float("inf")
    iteration = 0

    for iteration in range(1, max_iters + 1):  # noqa: B007  # reported count
        Q_prev = Q

        # Q-update. Setting the gradient to zero column by column, with B the
        # unpenalised right-hand side and c = sum(Q_j):
        #   (1 + 2 rho) Q_j + lam_m (c - M_j) 1 = B_j
        # Summing over rows gives c, then Q_j follows.
        B = P + rho * (Z - u_z) + rho * (Y - u_y)
        c = (B.sum(axis=0) + lam_m * N * M) / (1 + 2 * rho + lam_m * N)
        Q = (B + lam_m * (M - c)) / (1 + 2 * rho)

        Z_prev, Y_prev = Z, Y
        Z = _project_row_simplex(Q + u_z)
        V = Q + u_y
        Y = V.copy()
        if lam_r > 0:
            for j, idx in enumerate(column_orders):
                Y[idx, j] = prox_near_isotonic(V[idx, j], lam_r / rho)

        u_z = u_z + Q - Z
        u_y = u_y + Q - Y

        objective_values.append(_objective(Z, P, M, lam_m, lam_r, column_orders))

        scale = 1.0 + float(np.linalg.norm(Q))
        primal = max(float(np.linalg.norm(Q - Z)), float(np.linalg.norm(Q - Y)))
        dual = rho * (
            float(np.linalg.norm(Z - Z_prev)) + float(np.linalg.norm(Y - Y_prev))
        )
        final_change = float(np.linalg.norm(Q - Q_prev)) / scale
        if primal <= tol * scale and dual <= tol * scale:
            converged = True
            break

    if not converged:
        warnings.warn(
            f"Soft calibration did not converge in {iteration} iterations; "
            f"final relative change {final_change:.2e}.",
            UserWarning,
            stacklevel=2,
        )

    Q = Z
    col_sums = Q.sum(axis=0)
    return SoftCalibrationResult(
        Q=Q,
        converged=converged,
        iterations=iteration,
        objective_values=objective_values,
        fit_term=float(np.sum((Q - P) ** 2)),
        marginal_term=float(np.sum((col_sums - M) ** 2)),
        rank_term=_compute_rank_penalty(Q, column_orders),
        max_row_error=float(np.max(np.abs(Q.sum(axis=1) - 1.0))),
        max_col_error=float(np.max(np.abs(col_sums - M))),
        max_rank_violation=_compute_rank_violation(Q, P),
        final_change=final_change,
    )


def _objective(
    Q: np.ndarray,
    P: np.ndarray,
    M: np.ndarray,
    lam_m: float,
    lam_r: float,
    column_orders: list[np.ndarray],
) -> float:
    """Evaluate the objective calibrate_soft minimises.

    Args:
        Q: Candidate matrix.
        P: Input probability matrix.
        M: Target column sums.
        lam_m: Marginal penalty weight.
        lam_r: Rank penalty weight.
        column_orders: Per-column orderings of P.

    Returns:
        The objective value at Q.
    """
    return float(
        0.5 * np.sum((Q - P) ** 2)
        + 0.5 * lam_m * np.sum((Q.sum(axis=0) - M) ** 2)
        + lam_r * _compute_rank_penalty(Q, column_orders)
    )
