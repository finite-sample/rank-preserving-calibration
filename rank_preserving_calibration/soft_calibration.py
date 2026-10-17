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
    _validate_arrays,
)


@dataclass(slots=True)
class SoftCalibrationResult:
    """Result from soft-constraint calibration.

    Attributes:
        Q: Calibrated probability matrix.
        converged: Whether the solver solved the program and every row of Q is
            a probability distribution to within 1e-7.
        iterations: Interior-point iterations.
        objective: The minimised objective at Q.
        fit_term: ||Q - P||^2 at Q.
        marginal_term: ||col_sums(Q) - M||^2 at Q.
        rank_term: Summed downward steps at Q.
        max_row_error: Maximum row sum error.
        max_col_error: Maximum column sum error.
        max_rank_violation: Maximum rank violation.
    """

    Q: np.ndarray
    converged: bool
    iterations: int
    objective: float
    fit_term: float
    marginal_term: float
    rank_term: float
    max_row_error: float
    max_col_error: float
    max_rank_violation: float


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
    tol: float = 1e-10,
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

    It is a quadratic program, solved by Clarabel's interior-point method as
    calibrate_qp is: variables Q, the marginal gaps s = Q^T 1 - M, and a hinge
    v >= q_lower - q_upper, v >= 0 per adjacent pair. An ADMM with an exact
    rank prox reached the same optimum at N = 30 but stalled at its iteration
    cap from N = 300, 25-90x slower and up to 2e-2 above the optimum.

    Args:
        P: Input probability matrix of shape (N, J).
        M: Target column sums of shape (J,).
        lam_m: Marginal penalty weight. 0 ignores the marginals.
        lam_r: Rank penalty weight. 0 ignores the ranking; large values
            approach strict isotonicity.
        tol: Solver gap and feasibility tolerance.
        verbose: Print the solver log.

    Returns:
        SoftCalibrationResult with the calibrated matrix.

    Raises:
        CalibrationError: If P or M are malformed, a weight is negative, or the
            solver fails.

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
    import clarabel
    import scipy.sparse as sp

    _configure_logging(verbose)

    N, J = _validate_arrays(P, M)

    if lam_m < 0 or lam_r < 0:
        raise CalibrationError("lam_m and lam_r must be non-negative")

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

    # x = [q (N*J, row-major), s (J), v (J*(N-1))]; Clarabel wants
    # A x + slack = b with the slack in the listed cones.
    n, m = N * J, J * (N - 1)
    n_var = n + J + m
    k = np.arange(n)
    h = np.arange(m)
    lower = np.concatenate([o[:-1] * J + j for j, o in enumerate(column_orders)])
    upper = np.concatenate([o[1:] * J + j for j, o in enumerate(column_orders)])
    A = sp.vstack(
        [
            # Zero cone: row sums equal 1, and s - Q^T 1 = -M.
            sp.csr_matrix((np.ones(n), (k // J, k)), shape=(N, n_var)),
            sp.csr_matrix(
                (
                    np.r_[-np.ones(n), np.ones(J)],
                    (np.r_[k % J, np.arange(J)], np.r_[k, n + np.arange(J)]),
                ),
                shape=(J, n_var),
            ),
            # Non-negative cone: q_lower - q_upper - v <= 0, -v <= 0, -q <= 0.
            sp.csr_matrix(
                (
                    np.r_[np.ones(m), -np.ones(m), -np.ones(m)],
                    (np.r_[h, h, h], np.r_[lower, upper, n + J + h]),
                ),
                shape=(m, n_var),
            ),
            sp.csr_matrix((-np.ones(m), (h, n + J + h)), shape=(m, n_var)),
            sp.csr_matrix((-np.ones(n), (k, k)), shape=(n, n_var)),
        ],
        format="csc",
    )
    b = np.r_[np.ones(N), -M, np.zeros(2 * m + n)]
    quad = sp.diags(np.r_[np.ones(n), np.full(J, float(lam_m)), np.zeros(m)]).tocsc()
    linear = np.r_[-P.reshape(-1), np.zeros(J), np.full(m, float(lam_r))]
    cones = [
        clarabel.ZeroConeT(N + J),  # type: ignore[attr-defined]
        clarabel.NonnegativeConeT(2 * m + n),  # type: ignore[attr-defined]
    ]

    settings = clarabel.DefaultSettings()  # type: ignore[attr-defined]
    settings.verbose = bool(verbose)
    settings.tol_gap_abs = tol
    settings.tol_gap_rel = tol
    settings.tol_feas = tol
    # AlmostSolved means these looser tolerances were met instead.
    settings.reduced_tol_gap_abs = 100 * tol
    settings.reduced_tol_gap_rel = 100 * tol
    settings.reduced_tol_feas = 100 * tol
    solution = clarabel.DefaultSolver(  # type: ignore[attr-defined]
        quad, linear, A, b, cones, settings
    ).solve()
    status = str(solution.status)
    if status not in ("Solved", "AlmostSolved"):
        raise CalibrationError(f"Soft calibration solver returned status {status!r}.")

    # Interior-point output can sit 1e-12 below zero; clipping keeps the rows.
    Q = np.maximum(np.asarray(solution.x[:n], dtype=float), 0.0).reshape(N, J)
    col_sums = Q.sum(axis=0)
    fit_term = float(np.sum((Q - P) ** 2))
    marginal_term = float(np.sum((col_sums - M) ** 2))
    rank_term = _compute_rank_penalty(Q, column_orders)
    max_row_error = float(np.max(np.abs(Q.sum(axis=1) - 1.0)))
    return SoftCalibrationResult(
        Q=Q,
        converged=max_row_error <= 1e-7,
        iterations=int(getattr(solution, "iterations", 0) or 0),
        objective=0.5 * fit_term + 0.5 * lam_m * marginal_term + lam_r * rank_term,
        fit_term=fit_term,
        marginal_term=marginal_term,
        rank_term=rank_term,
        max_row_error=max_row_error,
        max_col_error=float(np.max(np.abs(col_sums - M))),
        max_rank_violation=_compute_rank_violation(Q, P),
    )
