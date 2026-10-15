# rank_preserving_calibration/kl_calibration.py
"""KL-divergence rank-preserving calibration.

This module provides rank-preserving calibration using KL divergence (relative entropy)
as the loss function instead of squared Euclidean distance. KL divergence is a natural
choice for probability calibration in label-shift scenarios.

Key innovations:
1. **Anchor-Reference Decoupling**: Separate A (ranking anchor) from R (the KL
   reference)
2. **Geometric Mean Pooling**: KL isotonic regression uses the geometric, not
   the arithmetic, mean
3. **Multiplicative Rescaling**: Sum constraints via multiplication (not addition)
4. **Pareto Frontier**: Report whole λ-path, not single tuned point

Exports:
    - KLCalibrationResult: Result container for hard KL calibration
    - KLParetoResult: Result container for Pareto frontier sweep
    - calibrate_kl: Main hard KL solver with anchor-reference decoupling
    - calibrate_kl_soft: Soft KL calibration with λ penalty, solved exactly
    - calibrate_kl_pareto: The KL / rank-violation frontier over a λ grid
"""

from __future__ import annotations

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
    # SIM300 fires because ruff reads the bare capital `Q` as a constant; it
    # is the probability matrix, so this is not a Yoda condition.
    mask = Q > eps  # noqa: SIM300
    kl = np.zeros_like(Q)
    kl[mask] = Q[mask] * (_safe_log(Q[mask], eps) - _safe_log(R[mask], eps))

    return float(np.sum(kl))


def _prepare_kl(
    P: np.ndarray,
    M: np.ndarray,
    R: np.ndarray | None,
    A: np.ndarray | None,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate the KL solvers' inputs and fill in the defaults R = A = P.

    Args:
        P: Input probability matrix of shape (N, J).
        M: Target column sums of shape (J,).
        R: Reference for KL divergence, or None for P.
        A: Anchor for rank ordering, or None for P.
        tol: Solver tolerance.

    Returns:
        Tuple of (P, M, R, A) as float arrays.

    Raises:
        CalibrationError: If R or A has the wrong shape or R is negative.
    """
    _validate_inputs(P, M, max_iters=1, tol=tol, feasibility_tol=0.0)
    P = np.asarray(P, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    R = P.copy() if R is None else np.asarray(R, dtype=np.float64)
    A = P.copy() if A is None else np.asarray(A, dtype=np.float64)
    if R.shape != P.shape or A.shape != P.shape:
        raise CalibrationError(f"R and A must have shape {P.shape}")
    if np.any(R < 0) or not np.isfinite(R).all():
        raise CalibrationError("R must be finite and non-negative")
    return P, M, R, A


def _kl_result(
    Q: np.ndarray, R: np.ndarray, A: np.ndarray, M: np.ndarray, iterations: int
) -> KLCalibrationResult:
    """Package a solved matrix with its diagnostics.

    Args:
        Q: Solved matrix.
        R: KL reference.
        A: Ranking anchor.
        M: Target column sums.
        iterations: Solver iterations.

    Returns:
        KLCalibrationResult for Q.
    """
    return KLCalibrationResult(
        Q=Q,
        converged=True,
        iterations=iterations,
        kl_divergence=_kl_div_matrix(Q, R),
        max_row_error=float(np.max(np.abs(Q.sum(axis=1) - 1.0))),
        max_col_error=float(np.max(np.abs(Q.sum(axis=0) - M))),
        max_rank_violation=_compute_rank_violation(Q, A),
        final_change=0.0,
    )


def calibrate_kl(
    P: np.ndarray,
    M: np.ndarray,
    R: np.ndarray | None = None,
    A: np.ndarray | None = None,
    tol: float = 1e-10,
    verbose: bool = False,
) -> KLCalibrationResult:
    """Calibrate by minimising KL divergence under rank preservation.

    Solves exactly, as an exponential-cone program::

        min KL(Q||R)  s.t.  rows of Q on the simplex, column sums = M,
                            each column non-decreasing in the order of A

    The anchor-reference decoupling separates what divergence is measured
    from (R) and what fixes the rank order (A).

    Args:
        P: Input probability matrix of shape (N, J).
        M: Target column sums of shape (J,). Must sum to N.
        R: Reference distribution for KL divergence. If None, uses P.
        A: Anchor for rank ordering. If None, uses P.
        tol: Solver gap and feasibility tolerance.
        verbose: Print the solver log.

    Returns:
        KLCalibrationResult with calibrated matrix and diagnostics.

    Raises:
        CalibrationError: If the inputs are invalid or the solver fails.

    Examples:
        >>> import numpy as np
        >>> from rank_preserving_calibration import calibrate_kl
        >>> P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> M = np.array([0.8, 0.8, 0.4])
        >>> result = calibrate_kl(P, M)
        >>> round(result.max_col_error, 6), result.max_rank_violation < 1e-8
        (0.0, True)

        With anchor-reference decoupling:

        >>> A = np.array([[0.6, 0.3, 0.1], [0.4, 0.4, 0.2]])
        >>> result = calibrate_kl(P, M, R=P, A=A)
    """
    _configure_logging(verbose)
    P, M, R, A = _prepare_kl(P, M, R, A, tol)
    # Exact penalty with a certificate. The penalised problem relaxes the
    # constrained one, so a penalised minimiser with no rank violations is the
    # constrained minimiser. Posing the constraints directly stalls the
    # interior-point solver when many of them are active at once; the hinge
    # variables keep the problem strictly interior.
    for lam in (1e3, 1e5, 1e7):
        Q, status, iterations = _solve_kl(R, A, M, lam, tol, verbose)
        if status not in ("Solved", "AlmostSolved"):
            raise CalibrationError(
                f"KL solver returned status {status!r}. Check that sum(M) == N "
                "and that R is positive wherever Q needs mass."
            )
        if _compute_rank_violation(Q, A) <= 1e-9:
            return _kl_result(Q, R, A, M, iterations)
    raise CalibrationError(
        "KL calibration could not certify rank preservation: violations remain "
        "at a penalty weight of 1e7."
    )


def _compute_rank_penalty_from_orders(
    Q: np.ndarray, column_orders: list[np.ndarray]
) -> float:
    """Sum the rank violations (q[i] - q[i+1])_+ over every column."""
    penalty = 0.0
    J = Q.shape[1]
    for j in range(J):
        q_sorted = Q[column_orders[j], j]
        if q_sorted.size > 1:
            diffs = q_sorted[:-1] - q_sorted[1:]
            penalty += float(np.maximum(diffs, 0.0).sum())
    return penalty


def _solve_kl(
    R: np.ndarray,
    A: np.ndarray,
    M: np.ndarray,
    lam: float,
    tol: float,
    verbose: bool,
) -> tuple[np.ndarray, str, int]:
    """Minimise KL(Q||R) + lam * V_A(Q) over C(M) as an exponential-cone program.

    Variables are x = [q, t, v]: the N*J entries of Q (row-major), one
    epigraph variable per entry with t >= q log(q / r), and one hinge per
    adjacent pair in anchor order with v >= q_lower - q_upper, v >= 0. The
    objective is sum(t) + lam * sum(v); KL's -sum(q) + sum(r) is constant
    because the row sums fix sum(q) = N. Clarabel's cone is
    s3 >= s2 exp(s1 / s2) for s = b - A x, so (s1, s2, s3) = (-t, q, r)
    encodes the epigraph. An entry with r = 0 forces
    q = 0, imposed directly since the cone has no interior there.

    Args:
        R: KL reference of shape (N, J), non-negative.
        A: Ranking anchor of shape (N, J).
        M: Target column sums of shape (J,).
        lam: Rank penalty weight.
        tol: Solver gap and feasibility tolerance.
        verbose: Print the solver log.

    Returns:
        Tuple of (Q, solver status, iterations).
    """
    import clarabel
    import scipy.sparse as sp

    N, J = R.shape
    n = N * J
    m = J * (N - 1)
    n_var = 2 * n + m
    r = R.reshape(-1)
    zero_r = np.flatnonzero(r == 0)
    pos_r = np.flatnonzero(r > 0)

    blocks = []
    b_parts = []

    # Zero cone: row sums, column sums, and q = t = 0 where r = 0.
    k = np.arange(n)
    blocks.append(sp.csr_matrix((np.ones(n), (k // J, k)), shape=(N, n_var)))
    b_parts.append(np.ones(N))
    blocks.append(sp.csr_matrix((np.ones(n), (k % J, k)), shape=(J, n_var)))
    b_parts.append(M)
    z = zero_r.size
    blocks.append(
        sp.csr_matrix(
            (np.ones(2 * z), (np.arange(2 * z), np.r_[zero_r, n + zero_r])),
            shape=(2 * z, n_var),
        )
    )
    b_parts.append(np.zeros(2 * z))
    n_zero = N + J + 2 * z

    # Non-negative cone: q_lower - q_upper - v <= 0, then -v <= 0.
    lower = np.concatenate(
        [np.argsort(A[:, j], kind="mergesort")[:-1] * J + j for j in range(J)]
    )
    upper = np.concatenate(
        [np.argsort(A[:, j], kind="mergesort")[1:] * J + j for j in range(J)]
    )
    h = np.arange(m)
    blocks.append(
        sp.csr_matrix(
            (
                np.r_[np.ones(m), -np.ones(m), -np.ones(m)],
                (np.r_[h, h, h], np.r_[lower, upper, 2 * n + h]),
            ),
            shape=(m, n_var),
        )
    )
    blocks.append(sp.csr_matrix((-np.ones(m), (h, 2 * n + h)), shape=(m, n_var)))
    b_parts.extend([np.zeros(m), np.zeros(m)])

    # Exponential cones, three rows per entry with r > 0: (-t, q, r).
    e = pos_r.size
    rows = 3 * np.arange(e)
    blocks.append(
        sp.csr_matrix(
            (
                np.r_[np.ones(e), -np.ones(e)],
                (np.r_[rows, rows + 1], np.r_[n + pos_r, pos_r]),
            ),
            shape=(3 * e, n_var),
        )
    )
    b_exp = np.zeros(3 * e)
    b_exp[2::3] = r[pos_r]
    b_parts.append(b_exp)

    A_mat = sp.vstack(blocks, format="csc")
    b = np.concatenate(b_parts)
    c = np.r_[np.zeros(n), np.ones(n), np.full(m, lam)]
    cones = [
        clarabel.ZeroConeT(n_zero),  # type: ignore[attr-defined]
        clarabel.NonnegativeConeT(2 * m),  # type: ignore[attr-defined]
    ] + [clarabel.ExponentialConeT()] * e  # type: ignore[attr-defined]

    settings = clarabel.DefaultSettings()  # type: ignore[attr-defined]
    settings.verbose = bool(verbose)
    settings.tol_gap_abs = tol
    settings.tol_gap_rel = tol
    settings.tol_feas = tol
    # Clarabel reports AlmostSolved when it meets these looser tolerances but
    # not the primary ones; the exponential cone does that at the odd lambda.
    # At 100x the primary tolerance Q is still accurate to about 1e-7.
    settings.reduced_tol_gap_abs = 100 * tol
    settings.reduced_tol_gap_rel = 100 * tol
    settings.reduced_tol_feas = 100 * tol

    solver = clarabel.DefaultSolver(  # type: ignore[attr-defined]
        sp.csc_matrix((n_var, n_var)), c, A_mat, b, cones, settings
    )
    solution = solver.solve()
    Q = np.maximum(np.asarray(solution.x[:n], dtype=float), 0.0).reshape(N, J)
    return Q, str(solution.status), int(getattr(solution, "iterations", 0) or 0)


def calibrate_kl_soft(
    P: np.ndarray,
    M: np.ndarray,
    lam: float = 1.0,
    R: np.ndarray | None = None,
    A: np.ndarray | None = None,
    tol: float = 1e-10,
    verbose: bool = False,
) -> KLCalibrationResult:
    """Soft KL calibration with a λ-weighted rank penalty.

    Solves exactly, as an exponential-cone program::

        min KL(Q||R) + λ·V_A(Q)  s.t.  Q ∈ C(M)

    Where:
        - KL(Q||R) = sum q log(q / r) is the divergence from the reference R
        - V_A(Q) sums every adjacent downward step, (q_lower - q_upper)_+,
          within each column in the order of the anchor A
        - C(M) is rows on the simplex with column sums equal to M

    At λ = 0 this is the KL projection onto C(M), which IPF also computes;
    for λ large enough the penalty is exact and the result equals
    :func:`calibrate_kl`.

    Args:
        P: Input probability matrix of shape (N, J).
        M: Target column sums of shape (J,). Must sum to N.
        lam: Rank penalty weight. Larger = more isotonic.
        R: Reference for KL divergence. Default: P.
        A: Anchor for rank ordering. Default: P.
        tol: Solver gap and feasibility tolerance.
        verbose: Print the solver log.

    Returns:
        KLCalibrationResult with the calibrated matrix.

    Raises:
        CalibrationError: If the inputs are invalid, λ is negative, R or A has
            the wrong shape or R a negative entry, or the solver fails.

    Examples:
        >>> import numpy as np
        >>> from rank_preserving_calibration import calibrate_kl_soft
        >>> P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> M = np.array([0.8, 0.8, 0.4])
        >>> strong = calibrate_kl_soft(P, M, lam=10.0)
        >>> weak = calibrate_kl_soft(P, M, lam=0.0)
        >>> bool(strong.max_rank_violation <= weak.max_rank_violation)
        True
    """
    _configure_logging(verbose)
    if lam < 0:
        raise CalibrationError("lam must be non-negative")
    P, M, R, A = _prepare_kl(P, M, R, A, tol)
    Q, status, iterations = _solve_kl(R, A, M, float(lam), tol, verbose)
    if status not in ("Solved", "AlmostSolved"):
        raise CalibrationError(
            f"KL soft solver returned status {status!r}. Check that sum(M) == N "
            "and that R is positive wherever Q needs mass."
        )
    return _kl_result(Q, R, A, M, iterations)


def calibrate_kl_pareto(
    P: np.ndarray,
    M: np.ndarray,
    lambda_grid: list[float] | np.ndarray | None = None,
    R: np.ndarray | None = None,
    A: np.ndarray | None = None,
    tol: float = 1e-10,
    verbose: bool = False,
) -> KLParetoResult:
    """Compute the frontier of KL divergence against rank violation.

    Solves :func:`calibrate_kl_soft` at each λ, reporting the whole path rather
    than a single tuned point. Because each solve is exact, KL is
    non-decreasing and the rank penalty non-increasing along the path.

    Args:
        P: Input probability matrix of shape (N, J).
        M: Target column sums of shape (J,).
        lambda_grid: λ values to solve at. Default: 20 points from 0.01 to 1000.
        R: Reference for KL divergence. Default: P.
        A: Anchor for rank ordering. Default: P.
        tol: Solver gap and feasibility tolerance.
        verbose: Print the solver logs.

    Returns:
        KLParetoResult with solutions along the frontier, in increasing λ.

    Examples:
        >>> import numpy as np
        >>> from rank_preserving_calibration import calibrate_kl_pareto
        >>> P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> M = np.array([0.8, 0.8, 0.4])
        >>> result = calibrate_kl_pareto(P, M)
        >>> len(result.solutions) == len(result.kl_values) == len(result.lambda_path)
        True
    """
    _validate_inputs(P, M, max_iters=1, tol=tol, feasibility_tol=0.0)
    grid = np.logspace(-2, 3, 20) if lambda_grid is None else lambda_grid
    A_ = np.asarray(P if A is None else A, dtype=np.float64)
    column_orders = [np.argsort(A_[:, j], kind="mergesort") for j in range(A_.shape[1])]

    result = KLParetoResult()
    for lam in sorted(float(x) for x in grid):
        sol = calibrate_kl_soft(P, M, lam=lam, R=R, A=A, tol=tol, verbose=verbose)
        result.solutions.append((lam, sol.Q.copy()))
        result.kl_values.append(sol.kl_divergence)
        result.rank_violations.append(
            _compute_rank_penalty_from_orders(sol.Q, column_orders)
        )
        result.lambda_path.append(lam)
    return result
