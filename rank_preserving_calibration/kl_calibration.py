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

# Row and column error a returned matrix must meet to count as converged, and
# the rank violation calibrate_kl accepts as none: solver precision, sized
# from the worst cases over 40 peaked problems (2.3e-8 and 2.5e-9).
_FEASIBILITY_TOL = 1e-7
_RANK_TOL = 1e-8

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
        converged: True if the row and column errors are both at most 1e-7.
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


def _compute_weak_rank_violation(Q: np.ndarray, A: np.ndarray) -> float:
    """Maximum strict weak-order violation: max_{a_i < a_k} (q_i - q_k)_+."""
    max_viol = 0.0
    for j in range(Q.shape[1]):
        _, inv = np.unique(A[:, j], return_inverse=True)
        K = int(inv.max()) + 1
        if K <= 1:
            continue
        q_col = Q[:, j]
        b_max = np.full(K, -np.inf)
        b_min = np.full(K, np.inf)
        np.maximum.at(b_max, inv, q_col)
        np.minimum.at(b_min, inv, q_col)
        prefix_max = np.maximum.accumulate(b_max)
        viol = float(np.max(np.maximum(0.0, prefix_max[:-1] - b_min[1:])))
        if viol > max_viol:
            max_viol = viol
    return max_viol


def _kl_result(
    Q: np.ndarray,
    R: np.ndarray,
    A: np.ndarray,
    M: np.ndarray,
    iterations: int,
    *,
    ties: str = "weak",
    feasibility_tol: float = _FEASIBILITY_TOL,
) -> KLCalibrationResult:
    """Package a solved matrix with its diagnostics."""
    max_row_error = float(np.max(np.abs(Q.sum(axis=1) - 1.0)))
    max_col_error = float(np.max(np.abs(Q.sum(axis=0) - M)))
    max_rank_violation = (
        _compute_weak_rank_violation(Q, A)
        if ties == "weak"
        else _compute_rank_violation(Q, A)
    )
    return KLCalibrationResult(
        Q=Q,
        converged=max(max_row_error, max_col_error) <= feasibility_tol,
        iterations=iterations,
        kl_divergence=_kl_div_matrix(Q, R),
        max_row_error=max_row_error,
        max_col_error=max_col_error,
        max_rank_violation=max_rank_violation,
        final_change=0.0,
    )


def calibrate_kl(
    P: np.ndarray,
    M: np.ndarray,
    R: np.ndarray | None = None,
    A: np.ndarray | None = None,
    tol: float = 1e-10,
    verbose: bool = False,
    *,
    ties: str = "weak",
) -> KLCalibrationResult:
    """Calibrate by minimizing KL divergence under rank preservation.

    Solves exactly, as an exponential-cone program::

        min KL(Q||R)  s.t.  rows of Q on the simplex, column sums = M,
                            each column non-decreasing in the order of A

    With ``ties="weak"`` (default), tied anchor entries ``A[i1, j] == A[i2, j]``
    are unconstrained relative to each other and only strictly ordered blocks
    are constrained via block-boundary variables. With ``ties="stable"``, ties
    follow row-index order (legacy behavior).
    """
    _configure_logging(verbose)
    if ties not in ("weak", "stable"):
        raise CalibrationError(f"ties must be 'weak' or 'stable', got {ties!r}")
    P, M, R, A = _prepare_kl(P, M, R, A, tol)
    feas_tol = max(_FEASIBILITY_TOL, 100.0 * tol)
    rank_tol = max(1e-7, 100.0 * tol)
    if ties == "weak":
        try:
            Q, iterations, _ = _solve_kl(R, A, M, None, tol, verbose)
        except CalibrationError as exc:
            if "PrimalInfeasible" in str(exc):
                raise CalibrationError(
                    "KL calibration could not certify rank preservation: "
                    "the rank constraints are probably infeasible with these "
                    f"targets ({exc})."
                ) from exc
            raise
        result = _kl_result(
            Q, R, A, M, iterations, ties="weak", feasibility_tol=feas_tol
        )
        if result.converged and result.max_rank_violation <= rank_tol:
            return result
    # Fallback / stable-tie exact-penalty path.
    for lam in (1e3, 1e4, 1e5):
        Q, iterations, _ = (
            _solve_kl(R, A, M, lam, tol, verbose)
            if ties == "weak"
            else _solve_kl(R, A, M, lam, tol, verbose, ties=ties)
        )
        result = _kl_result(Q, R, A, M, iterations, ties=ties, feasibility_tol=feas_tol)
        if result.converged and result.max_rank_violation <= rank_tol:
            return result
    raise CalibrationError(
        "KL calibration could not certify rank preservation: violations remain "
        "at a penalty weight of 1e5, so the rank constraints are probably "
        "infeasible with these targets -- for instance a zero in R high in an "
        "anchor column forces every entry below it to zero."
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


def _solve_kl_once(
    R: np.ndarray,
    A: np.ndarray,
    M: np.ndarray,
    lam: float | None,
    tol: float,
    verbose: bool,
    *,
    ties: str = "weak",
) -> tuple[np.ndarray, str, int]:
    """Minimize KL(Q||R) (+ lam * V_A(Q)) over C(M) as an exponential-cone program."""
    import clarabel
    import scipy.sparse as sp

    N, J = R.shape
    n = N * J
    r = R.reshape(-1)
    zero_r = np.flatnonzero(r == 0)
    pos_r = np.flatnonzero(r > 0)

    has_ties = ties == "weak" and any(len(np.unique(A[:, j])) < N for j in range(J))
    soft = lam is not None

    n_u = 0
    n_ineq = 0
    n_mono = 0
    m = J * (N - 1)
    ineq_q: list[int] = []
    ineq_u: list[int] = []
    ineq_sq: list[float] = []
    ineq_su: list[float] = []
    u_lo: list[int] = []
    u_hi: list[int] = []

    if has_ties:
        for j in range(J):
            _, inv = np.unique(A[:, j], return_inverse=True)
            K = int(inv.max()) + 1
            if K <= 1:
                continue
            for i in range(N):
                k_blk = int(inv[i])
                q_idx = i * J + j
                if k_blk < K - 1:
                    ineq_q.append(q_idx)
                    ineq_u.append(n_u + k_blk)
                    ineq_sq.append(1.0)
                    ineq_su.append(-1.0)
                if k_blk > 0:
                    ineq_q.append(q_idx)
                    ineq_u.append(n_u + k_blk - 1)
                    ineq_sq.append(-1.0)
                    ineq_su.append(1.0)
            for k_blk in range(K - 2):
                u_lo.append(n_u + k_blk)
                u_hi.append(n_u + k_blk + 1)
            n_u += K - 1
        n_ineq = len(ineq_q)
        n_mono = len(u_lo)
        n_v = n_ineq if soft else 0
        n_var = 2 * n + n_u + n_v
    else:
        n_v = m if soft else 0
        n_var = 2 * n + n_v

    blocks = []
    b_parts = []

    # Zero cone: row sums, column sums, and q = t = 0 where r = 0.
    k = np.arange(n)
    blocks.append(sp.csr_matrix((np.ones(n), (k // J, k)), shape=(N, n_var)))
    b_parts.append(np.ones(N))
    blocks.append(sp.csr_matrix((np.ones(n), (k % J, k)), shape=(J, n_var)))
    b_parts.append(M)
    z = zero_r.size
    if z > 0:
        blocks.append(
            sp.csr_matrix(
                (np.ones(2 * z), (np.arange(2 * z), np.r_[zero_r, n + zero_r])),
                shape=(2 * z, n_var),
            )
        )
        b_parts.append(np.zeros(2 * z))
    n_zero = N + J + 2 * z

    # Non-negative cone: rank constraints (and hinge slacks if soft).
    if has_ties:
        r_idx = np.arange(n_ineq)
        if soft:
            blocks.append(
                sp.csr_matrix(
                    (
                        np.r_[ineq_sq, ineq_su, -np.ones(n_ineq)],
                        (
                            np.r_[r_idx, r_idx, r_idx],
                            np.r_[
                                ineq_q,
                                2 * n + np.array(ineq_u, dtype=int),
                                2 * n + n_u + r_idx,
                            ],
                        ),
                    ),
                    shape=(n_ineq, n_var),
                )
            )
        else:
            blocks.append(
                sp.csr_matrix(
                    (
                        np.r_[ineq_sq, ineq_su],
                        (
                            np.r_[r_idx, r_idx],
                            np.r_[ineq_q, 2 * n + np.array(ineq_u, dtype=int)],
                        ),
                    ),
                    shape=(n_ineq, n_var),
                )
            )
        b_parts.append(np.zeros(n_ineq))
        if n_mono > 0:
            rm = np.arange(n_mono)
            blocks.append(
                sp.csr_matrix(
                    (
                        np.r_[np.ones(n_mono), -np.ones(n_mono)],
                        (
                            np.r_[rm, rm],
                            np.r_[
                                2 * n + np.array(u_lo, dtype=int),
                                2 * n + np.array(u_hi, dtype=int),
                            ],
                        ),
                    ),
                    shape=(n_mono, n_var),
                )
            )
            b_parts.append(np.zeros(n_mono))
        if soft:
            rv = np.arange(n_v)
            blocks.append(
                sp.csr_matrix(
                    (-np.ones(n_v), (rv, 2 * n + n_u + rv)), shape=(n_v, n_var)
                )
            )
            b_parts.append(np.zeros(n_v))
        n_nonneg = n_ineq + n_mono + n_v
    else:
        lower = np.concatenate(
            [np.argsort(A[:, j], kind="mergesort")[:-1] * J + j for j in range(J)]
        )
        upper = np.concatenate(
            [np.argsort(A[:, j], kind="mergesort")[1:] * J + j for j in range(J)]
        )
        h = np.arange(m)
        if soft:
            blocks.append(
                sp.csr_matrix(
                    (
                        np.r_[np.ones(m), -np.ones(m), -np.ones(m)],
                        (np.r_[h, h, h], np.r_[lower, upper, 2 * n + h]),
                    ),
                    shape=(m, n_var),
                )
            )
            blocks.append(
                sp.csr_matrix((-np.ones(m), (h, 2 * n + h)), shape=(m, n_var))
            )
            b_parts.extend([np.zeros(m), np.zeros(m)])
            n_nonneg = 2 * m
        else:
            blocks.append(
                sp.csr_matrix(
                    (
                        np.r_[np.ones(m), -np.ones(m)],
                        (np.r_[h, h], np.r_[lower, upper]),
                    ),
                    shape=(m, n_var),
                )
            )
            b_parts.append(np.zeros(m))
            n_nonneg = m

    # Exponential cones, three rows per entry with r > 0: (-t, q, 1).
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
    b_exp[2::3] = 1.0
    b_parts.append(b_exp)

    A_mat = sp.vstack(blocks, format="csc")
    b = np.concatenate(b_parts)
    log_r = np.zeros(n)
    log_r[pos_r] = np.log(r[pos_r])
    c = np.zeros(n_var)
    c[:n] = -log_r
    c[n : 2 * n] = 1.0
    if soft:
        c[2 * n + n_u :] = float(lam)
    cones = [
        clarabel.ZeroConeT(n_zero),  # type: ignore[attr-defined]
        clarabel.NonnegativeConeT(n_nonneg),  # type: ignore[attr-defined]
    ] + [clarabel.ExponentialConeT()] * e  # type: ignore[attr-defined]

    settings = clarabel.DefaultSettings()  # type: ignore[attr-defined]
    settings.verbose = bool(verbose)
    settings.tol_gap_abs = tol
    settings.tol_gap_rel = tol
    settings.tol_feas = tol
    settings.reduced_tol_gap_abs = 100 * tol
    settings.reduced_tol_gap_rel = 100 * tol
    settings.reduced_tol_feas = 100 * tol

    solver = clarabel.DefaultSolver(  # type: ignore[attr-defined]
        sp.csc_matrix((n_var, n_var)), c, A_mat, b, cones, settings
    )
    solution = solver.solve()
    Q = np.maximum(np.asarray(solution.x[:n], dtype=float), 0.0).reshape(N, J)
    return Q, str(solution.status), int(getattr(solution, "iterations", 0) or 0)


def _solve_kl(
    R: np.ndarray,
    A: np.ndarray,
    M: np.ndarray,
    lam: float | None,
    tol: float,
    verbose: bool,
    *,
    ties: str = "weak",
) -> tuple[np.ndarray, int, float]:
    """Solve the KL program, loosening the tolerance if the solver stalls."""
    status = ""
    for attempt in (tol, 100 * tol, 1e4 * tol):
        Q, status, iterations = _solve_kl_once(
            R, A, M, lam, attempt, verbose, ties=ties
        )
        if status in ("Solved", "AlmostSolved"):
            return Q, iterations, attempt
    raise CalibrationError(
        f"KL solver returned status {status!r} down to tolerance {1e4 * tol:.0e}. "
        "Check that sum(M) == N and that R is positive wherever Q needs mass."
    )


def calibrate_kl_soft(
    P: np.ndarray,
    M: np.ndarray,
    lam: float = 1.0,
    R: np.ndarray | None = None,
    A: np.ndarray | None = None,
    tol: float = 1e-10,
    verbose: bool = False,
    *,
    ties: str = "weak",
) -> KLCalibrationResult:
    """Soft KL calibration with a λ-weighted rank penalty."""
    _configure_logging(verbose)
    if lam < 0:
        raise CalibrationError("lam must be non-negative")
    if ties not in ("weak", "stable"):
        raise CalibrationError(f"ties must be 'weak' or 'stable', got {ties!r}")
    P, M, R, A = _prepare_kl(P, M, R, A, tol)
    Q, iterations, _ = (
        _solve_kl(R, A, M, float(lam), tol, verbose)
        if ties == "weak"
        else _solve_kl(R, A, M, float(lam), tol, verbose, ties=ties)
    )
    feas_tol = max(_FEASIBILITY_TOL, 100.0 * tol)
    result = _kl_result(Q, R, A, M, iterations, ties=ties, feasibility_tol=feas_tol)
    if not result.converged:
        warnings.warn(
            f"KL soft solve missed the constraints: row error "
            f"{result.max_row_error:.1e}, column error {result.max_col_error:.1e}.",
            UserWarning,
            stacklevel=2,
        )
    return result


def _compute_weak_rank_penalty(Q: np.ndarray, A: np.ndarray) -> float:
    """Exact L1 weak-order penalty matching the soft KL objective."""
    N, J = Q.shape
    has_ties = any(len(np.unique(A[:, j])) < N for j in range(J))
    if not has_ties:
        orders = [np.argsort(A[:, j], kind="mergesort") for j in range(J)]
        return _compute_rank_penalty_from_orders(Q, orders)
    total = 0.0
    for j in range(J):
        _, inv = np.unique(A[:, j], return_inverse=True)
        K = int(inv.max()) + 1
        if K <= 1:
            continue
        q_col = Q[:, j]
        b_max = np.full(K, -np.inf)
        b_min = np.full(K, np.inf)
        np.maximum.at(b_max, inv, q_col)
        np.minimum.at(b_min, inv, q_col)
        total += float(np.maximum(0.0, b_max[:-1] - b_min[1:]).sum())
    return total


def calibrate_kl_pareto(
    P: np.ndarray,
    M: np.ndarray,
    lambda_grid: list[float] | np.ndarray | None = None,
    R: np.ndarray | None = None,
    A: np.ndarray | None = None,
    tol: float = 1e-10,
    verbose: bool = False,
    *,
    ties: str = "weak",
) -> KLParetoResult:
    """Compute the frontier of KL divergence against rank violation."""
    _validate_inputs(P, M, max_iters=1, tol=tol, feasibility_tol=0.0)
    if ties not in ("weak", "stable"):
        raise CalibrationError(f"ties must be 'weak' or 'stable', got {ties!r}")
    grid = np.logspace(-2, 3, 20) if lambda_grid is None else lambda_grid
    A_ = np.asarray(P if A is None else A, dtype=np.float64)
    column_orders = [np.argsort(A_[:, j], kind="mergesort") for j in range(A_.shape[1])]

    result = KLParetoResult()
    for lam in sorted(float(x) for x in grid):
        sol = calibrate_kl_soft(
            P, M, lam=lam, R=R, A=A, tol=tol, verbose=verbose, ties=ties
        )
        result.solutions.append((lam, sol.Q.copy()))
        result.kl_values.append(sol.kl_divergence)
        result.rank_violations.append(
            _compute_weak_rank_penalty(sol.Q, A_)
            if ties == "weak"
            else _compute_rank_penalty_from_orders(sol.Q, column_orders)
        )
        result.lambda_path.append(lam)
    return result
