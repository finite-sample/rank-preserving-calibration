"""Robust rank-preserving multiclass probability calibration.

This module provides numerically stable implementations of rank-preserving
calibration algorithms including Dykstra's alternating projections and ADMM.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._numba_utils import get_jit_functions
from .nearly import (
    project_near_isotonic_euclidean,  # epsilon-slack, with exact sum shift
    prox_near_isotonic,  # lambda-penalty (exact prox if provided version)
)

type NDArrayFloat = np.ndarray[Any, np.dtype[np.floating[Any]]]
type ColumnOrders = list[np.ndarray]
type CallbackFunction = Callable[[int, float, np.ndarray], bool] | None

_jit_funcs = get_jit_functions()

# Set up logging
logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    """Configure logging level based on verbosity setting."""
    if verbose:
        logger.setLevel(logging.DEBUG)
        # Ensure handler exists and is configured
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False
    else:
        logger.setLevel(logging.WARNING)
        # Remove any existing handlers when verbose=False to prevent leakage
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        logger.propagate = True  # Let parent loggers handle output


# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------


@dataclass(slots=True)
class CalibrationResult:
    """Result container for rank-preserving calibration algorithms.

    Returned by calibrate_dykstra() containing the calibrated probability matrix
    and diagnostic information about convergence and constraint satisfaction.

    Attributes:
        Q: Calibrated probability matrix of shape (N, J) where rows sum to 1
            and columns preserve rank ordering from original scores.
        converged: True if algorithm converged within specified tolerance.
        iterations: Number of iterations performed before termination.
        max_row_error: Maximum absolute error in row sum constraint (should be ≈0).
        max_col_error: Maximum absolute error in column sum constraint.
        max_rank_violation: Maximum rank-order violation across all columns.
        final_change: Final relative change in solution between iterations.

    Examples:
        >>> result = calibrate_dykstra(P, M)
        >>> if result.converged:
        ...     print(f"Calibration successful in {result.iterations} iterations")
        >>> print(f"Max rank violation: {result.max_rank_violation:.6f}")
    """

    Q: np.ndarray
    converged: bool
    iterations: int
    max_row_error: float
    max_col_error: float
    max_rank_violation: float
    final_change: float


@dataclass(slots=True)
class ADMMResult:
    """Result from ADMM optimization.

    Attributes:
        Q: Calibrated probability matrix.
        converged: Whether ADMM converged.
        iterations: Number of iterations performed.
        objective_values: Objective function values over iterations.
        primal_residuals: Primal residual norms over iterations.
        dual_residuals: Dual residual norms over iterations.
        max_row_error: Maximum row sum error.
        max_col_error: Maximum column sum error.
        max_rank_violation: Maximum rank violation.
        final_change: Final relative change between iterations.
    """

    Q: np.ndarray
    converged: bool
    iterations: int
    objective_values: list[float]
    primal_residuals: list[float]
    dual_residuals: list[float]
    max_row_error: float
    max_col_error: float
    max_rank_violation: float
    final_change: float


class CalibrationError(Exception):
    """Raised when calibration fails due to invalid inputs or numerical issues."""


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


def _validate_inputs(
    P: np.ndarray, M: np.ndarray, max_iters: int, tol: float, feasibility_tol: float
) -> tuple[int, int]:
    """Validate all inputs to calibration functions.

    Args:
        P: Input probability matrix to validate.
        M: Target column sums to validate.
        max_iters: Maximum iterations parameter to validate.
        tol: Tolerance parameter to validate.
        feasibility_tol: Feasibility tolerance to validate.

    Returns:
        Tuple of (N, J) where N is number of instances and J is number of classes.

    Raises:
        CalibrationError: If any input validation fails.
    """
    # Validate P using match statements
    match P:
        case x if not isinstance(x, np.ndarray):
            raise CalibrationError("P must be a numpy array")
        case x if x.ndim != 2:
            raise CalibrationError("P must be a 2D array of shape (N, J)")
        case x if x.size == 0:
            raise CalibrationError("P cannot be empty")
        case x if not np.isfinite(x).all():
            raise CalibrationError("P must not contain NaN or infinite values")
        case x if np.any(x < 0):
            raise CalibrationError("P must contain non-negative values")

    N, J = P.shape
    match J:
        case j if j < 2:
            raise CalibrationError("P must have at least 2 columns (classes)")

    # Validate M using match statements
    match M:
        case x if not isinstance(x, np.ndarray):
            raise CalibrationError("M must be a numpy array")
        case x if x.ndim != 1:
            raise CalibrationError("M must be a 1D array")
        case x if x.size != J:
            raise CalibrationError(f"M must have length {J} to match P.shape[1]")
        case x if not np.isfinite(x).all():
            raise CalibrationError("M must not contain NaN or infinite values")
        case x if np.any(x < 0):
            raise CalibrationError("M must contain non-negative values")

    # Feasibility is exact, not approximate: rows summing to 1 force the grand
    # total to N, so sum(M) != N makes the intersection *empty* -- there is no
    # nearest feasible point to return, and the solver will fail. Conversely,
    # when sum(M) == N the constant matrix Q[i, j] = M[j]/N always satisfies
    # every constraint, so the intersection is never empty and any failure is
    # conditioning rather than infeasibility.
    #
    # The warning therefore fires at the point where the problem stops being
    # solvable, scaled only by floating-point slack. It used to fire at
    # `feasibility_tol * N` (10% of N by default), so a 2% mismatch produced no
    # warning at all and then raised.
    M_sum = float(M.sum())
    difference = abs(M_sum - N)
    slack = max(1e-9 * max(N, 1.0), np.finfo(float).eps * 100.0 * max(M_sum, 1.0))
    if difference > slack:
        warnings.warn(
            f"Sum of M ({M_sum:.6g}) must equal N ({N}) for a feasible problem; "
            f"it differs by {difference:.3g}. Rows summing to 1 force the grand "
            f"total to N, so no matrix satisfies both constraint sets and "
            f"calibration will fail. Rescale with M = M * N / M.sum().",
            UserWarning,
            stacklevel=2,
        )

    # Validate other parameters using match statements
    match max_iters:
        case x if not isinstance(x, int) or x <= 0:
            raise CalibrationError("max_iters must be a positive integer")

    match tol:
        case x if not isinstance(x, int | float) or x <= 0:
            raise CalibrationError("tol must be a positive number")

    match feasibility_tol:
        case x if not isinstance(x, int | float) or x < 0:
            raise CalibrationError("feasibility_tol must be non-negative")

    return N, J


# ---------------------------------------------------------------------
# Core projections
# ---------------------------------------------------------------------


def _project_row_simplex(
    rows: np.ndarray, eps: float = 1e-15, use_jit: bool = True
) -> np.ndarray:
    """Project rows onto the probability simplex with numerical stability.

    Projects each row of the matrix onto the probability simplex, ensuring
    non-negative entries that sum to 1. Uses Euclidean projection algorithm
    with numerical stability improvements.

    Args:
        rows: Matrix of shape (N, J) where each row will be projected.
        eps: Small tolerance for numerical stability in computations.
        use_jit: Whether to use JIT-compiled version if available.

    Returns:
        Projected matrix with same shape, where each row sums to 1 and is non-negative.
    """
    # Use JIT version if available and requested
    if (
        use_jit
        and _jit_funcs["available"]
        and _jit_funcs["project_row_simplex"] is not None
    ):
        return _jit_funcs["project_row_simplex"](rows, eps)

    # Fallback to pure Python implementation
    N, J = rows.shape
    projected = np.empty_like(rows, dtype=np.float64)

    for i in range(N):
        v = rows[i]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1.0
        ind = np.arange(1, J + 1, dtype=np.float64)

        cond = u - cssv / ind > eps
        rho = np.nonzero(cond)[0][-1] if np.any(cond) else (J - 1)

        theta = cssv[rho] / (rho + 1)
        w = np.maximum(v - theta, 0.0)

        # Normalize defensively to exactly sum to 1
        sum_w = w.sum()
        if sum_w > eps:
            w /= sum_w
        else:
            w[:] = 1.0 / J

        projected[i] = w

    return projected


# ---------- Isotonic (PAV) -------------------------------------------


def _isotonic_regression(
    y: np.ndarray,
    rtol: float = 0.0,
    ties: str = "stable",
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Isotonic regression (nondecreasing) via stack-based PAV in O(n).

    Strict by default (rtol=0.0) to avoid micro-violations in tests.
    """

    def _tol(a: float, b: float) -> float:
        return rtol * (abs(a) + abs(b) + 1.0)

    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n <= 1:
        return y.copy()
    if rtol < 0:
        raise ValueError("rtol must be nonnegative.")
    if ties not in ("stable", "group"):
        raise ValueError("ties must be 'stable' or 'group'.")

    if weights is None:
        w_init = np.ones(n, dtype=np.int64)
    else:
        w_init = np.asarray(weights)
        if w_init.shape != y.shape:
            raise ValueError("weights must have the same shape as y")
        if np.any(w_init <= 0):
            raise ValueError("weights must be positive")
        if np.all(np.isclose(w_init, np.round(w_init))):
            w_init = np.round(w_init).astype(np.int64)
        else:
            w_init = w_init.astype(np.float64)

    # Optional pre-pooling of *contiguous exact equals in y*
    if ties == "group":
        vals: list[float] = []
        wts: list[float] = []
        i = 0
        while i < n:
            j = i + 1
            vi = y[i]
            total_w = float(w_init[i])
            while j < n and y[j] == vi:
                total_w += float(w_init[j])
                j += 1
            vals.append(float(vi))
            wts.append(total_w)
            i = j
        a = np.asarray(vals, dtype=np.float64)
        w0 = np.asarray(wts, dtype=np.float64)
    else:
        a = y
        w0 = np.asarray(w_init, dtype=np.float64)

    m = a.size
    # Block stacks
    start = np.empty(m, dtype=np.int64)  # start index in expanded output
    mean = np.empty(m, dtype=np.float64)  # block mean
    wsum = np.empty(m, dtype=np.float64)  # block weight
    top = -1
    idx = 0  # running start position

    for i in range(m):
        # push new block
        top += 1
        start[top] = idx
        mean[top] = a[i]
        wsum[top] = w0[i]
        idx += round(w0[i])

        # merge backward while violated beyond tolerance
        while top > 0 and mean[top - 1] > mean[top] + _tol(mean[top - 1], mean[top]):
            w1 = wsum[top - 1]
            w2 = wsum[top]
            mean[top - 1] = (w1 * mean[top - 1] + w2 * mean[top]) / (w1 + w2)
            wsum[top - 1] = w1 + w2
            top -= 1

    # expand pooled block means back to full length
    out = np.empty(n, dtype=np.float64)
    for j in range(top + 1):
        s = start[j]
        e = start[j + 1] if j < top else n
        out[s:e] = mean[j]
    return out


# ---------- Column projection (exact) --------------------------------


def _run_lengths_of_equals(x_sorted: np.ndarray) -> np.ndarray:
    """Run-lengths of contiguous exact equals in an already-sorted array."""
    n = x_sorted.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    lens: list[int] = []
    cnt = 1
    for i in range(1, n):
        if x_sorted[i] == x_sorted[i - 1]:
            cnt += 1
        else:
            lens.append(cnt)
            cnt = 1
    lens.append(cnt)
    return np.asarray(lens, dtype=np.int64)


def _project_column_isotonic_sum(
    column: np.ndarray,
    column_order: np.ndarray,
    target_sum: float,
    *,
    rtol: float = 0.0,
    nearly: dict | None = None,
    ties: str = "stable",
    score_sorted: np.ndarray | None = None,
) -> np.ndarray:
    """Project one column onto isotonic (by model-score order) with a fixed sum.

    Exact Euclidean projection: PAV, then a *uniform additive shift* c so the
    column sums to `target_sum`. If `ties=="group"` and `score_sorted` is given,
    we pre-pool equal-score runs, run weighted PAV, add the uniform shift, then expand.
    """
    if column.size == 0:
        return column.copy()

    y = column[column_order]

    # Nearly-isotonic mode selection using match
    match nearly:
        case {"mode": "epsilon", **rest}:
            eps = float(rest.get("eps", 1e-3))
            iso_shifted = project_near_isotonic_euclidean(y, eps, sum_target=target_sum)
        case None | _:
            if ties == "group" and score_sorted is not None:
                lens = _run_lengths_of_equals(score_sorted)
                k = lens.size
                y_group = np.empty(k, dtype=np.float64)
                pos = 0
                for g in range(k):
                    L = int(lens[g])
                    y_group[g] = float(np.mean(y[pos : pos + L]))
                    pos += L
                z_group = _isotonic_regression(
                    y_group, rtol=rtol, ties="stable", weights=lens.astype(np.float64)
                )
                total_n = int(lens.sum())
                c = (float(target_sum) - float(np.dot(z_group, lens))) / float(total_n)
                z_group_shift = z_group + c
                iso_shifted = np.repeat(z_group_shift, lens)
            else:
                iso = _isotonic_regression(y, rtol=rtol, ties=ties)
                c = (float(target_sum) - float(iso.sum())) / float(iso.size)
                iso_shifted = iso + c

    projected = np.empty_like(column, dtype=np.float64)
    projected[column_order] = iso_shifted
    return projected


# ---------------------------------------------------------------------
# Diagnostics & helpers
# ---------------------------------------------------------------------


def is_feasible(
    Q: np.ndarray,
    M: np.ndarray,
    *,
    row_atol: float = 1e-10,
    col_atol: float = 1e-10,
) -> bool:
    """Check that ``Q`` has unit row sums and column sums equal to ``M``.

    The tolerances are **absolute**. This function exists because that was not
    always true: every feasibility check in this module used to call
    ``np.allclose(x, target, atol=...)`` without passing ``rtol``, and
    ``np.allclose`` keeps its default ``rtol=1e-5``. The effective bound was
    ``atol + 1e-5 * |target|``, so a stated ``atol=1e-12`` on row sums behaved as
    ``1e-5``, and the column check -- measured against ``M_j``, which grows with
    ``N`` -- became *looser* as the problem got larger. Convergence was therefore
    reported least reliably exactly where it mattered most.

    Args:
        Q: Candidate matrix of shape (N, J).
        M: Target column sums of shape (J,).
        row_atol: Absolute tolerance on ``|row_sum - 1|``.
        col_atol: Absolute tolerance on ``|col_sum - M_j|``.

    Returns:
        True if every row sum and every column sum is within tolerance.

    Examples:
        >>> import numpy as np
        >>> Q = np.full((4, 2), 0.5)
        >>> is_feasible(Q, np.array([2.0, 2.0]))
        True

        A row error of 1e-6 is a real violation, and is reported as one:

        >>> Q_bad = Q.copy()
        >>> Q_bad[0, 0] += 1e-6
        >>> is_feasible(Q_bad, np.array([2.0, 2.0]))
        False
    """
    rows_ok = bool(np.all(np.abs(Q.sum(axis=1) - 1.0) <= row_atol))
    cols_ok = bool(
        np.all(np.abs(Q.sum(axis=0) - np.asarray(M, dtype=float)) <= col_atol)
    )
    return rows_ok and cols_ok


def _compute_rank_violation(Q: np.ndarray, P: np.ndarray) -> float:
    """Compute maximum rank violation across all columns (w.r.t. original scores)."""
    max_violation = 0.0
    _, J = Q.shape
    for j in range(J):
        idx = np.argsort(P[:, j])  # order by original model scores
        q_sorted = Q[idx, j]
        if q_sorted.size > 1:
            diffs = np.diff(q_sorted)
            violation = float(np.max(np.maximum(0.0, -diffs)))
            max_violation = max(max_violation, violation)
    return max_violation


def _detect_cycling(
    Q_history: list[NDArrayFloat], Q: NDArrayFloat, cycle_tol: float = 1e-12
) -> bool:
    """Very conservative cycle detection (usually disabled)."""
    matches = 0
    for prev_Q in Q_history:
        if np.allclose(Q, prev_Q, rtol=cycle_tol, atol=cycle_tol):
            matches += 1
            if matches >= 2:
                return True
    return False


def _polish_to_intersection(
    Q: np.ndarray,
    M: np.ndarray,
    column_orders: ColumnOrders,
    *,
    rtol: float = 0.0,
    ties: str = "stable",
    score_sorted: list[np.ndarray | None] | None = None,
    max_iters: int = 200,
    row_atol: float = 1e-8,
    col_atol: float = 1e-8,
) -> np.ndarray:
    """Small alternating-projection polish to hit constraints to machine tolerance."""
    _, J = Q.shape
    if score_sorted is None:
        score_sorted = [None] * J

    for _ in range(max_iters):
        Q = _project_row_simplex(Q)
        for j in range(J):
            Q[:, j] = _project_column_isotonic_sum(
                Q[:, j],
                column_orders[j],
                float(M[j]),
                rtol=rtol,
                ties=ties,
                score_sorted=score_sorted[j] if score_sorted else None,
            )
        if is_feasible(Q, M, row_atol=row_atol, col_atol=col_atol):
            break
    return Q


# ---------------------------------------------------------------------
# Dykstra calibration (exact projections onto each set)
# ---------------------------------------------------------------------


def calibrate_dykstra(
    P: np.ndarray,
    M: np.ndarray,
    max_iters: int = 3000,
    tol: float = 1e-7,
    rtol: float = 0.0,  # strict isotone by default
    feasibility_tol: float = 0.1,
    verbose: bool = False,
    callback: CallbackFunction = None,
    detect_cycles: bool = False,  # default off for determinism
    cycle_window: int = 10,
    nearly: dict | None = None,
    ties: str = "stable",
    use_jit: bool = True,
    row_atol: float = 1e-8,
    col_atol: float = 1e-8,
) -> CalibrationResult:
    """Calibrate using Dykstra's alternating projections.

    Projects multiclass probabilities onto the intersection of:
      (A) row simplex: {rows ≥ 0, rows sum to 1} and
      (B) column-wise isotone-by-score + fixed column sums:
          {nondecreasing in score order; column sum = M_j}.

    This is the recommended default method for rank-preserving calibration.
    The algorithm uses exact Euclidean projections via Pool Adjacent Violators
    (PAV) followed by uniform shifts to satisfy sum constraints.

    Args:
        P: Input probability matrix of shape (N, J). Each row represents predicted class
            probabilities for one instance. Rows need not sum to 1 initially.
        M: Target column sums of shape (J,). Should sum to approximately N for
            feasibility.
        max_iters: Maximum number of iterations. Default 3000 is usually sufficient.
        tol: Convergence tolerance for relative change in solution. Default 1e-7.
        rtol: Relative tolerance for isotonic violations in PAV. Default 0.0 (strict).
        feasibility_tol: Tolerance for feasibility warnings when sum(M) differs from N.
        verbose: If True, enables debug logging.
        callback: Optional function called each iteration as callback(iter, change, Q).
            Should return False to terminate early.
        detect_cycles: If True, detects and breaks cycles in the solution sequence.
        cycle_window: Number of iterations to look back for cycle detection.
        nearly: Optional dict for nearly-isotonic constraints. Use
            {"mode": "epsilon", "eps": 0.01} to allow small isotonicity violations.
        ties: How to handle tied scores. "stable" preserves input order, "group" pools
            equal-score instances.
        use_jit: If True and numba is available, uses JIT-compiled functions for speed.
        row_atol: Absolute tolerance on ``|row sum - 1|`` for convergence.
            Absolute, not relative -- see `is_feasible`. The 1e-8 default is what
            alternating projections can actually reach: each sweep ends on a
            column projection, which perturbs the row sums, so no finite iterate
            satisfies both constraint sets exactly. It is still three orders
            tighter than the 1e-5 the previous `np.allclose` check allowed.
        col_atol: Absolute tolerance on ``|column sum - M_j|`` for declaring
            convergence. Being absolute, it does not loosen as M_j grows with N,
            which the previous check did.

    Returns:
        CalibrationResult object containing:
            - Q: Calibrated probability matrix of shape (N, J)
            - converged: Always True (failures raise CalibrationError instead)
            - iterations: Number of iterations performed
            - max_row_error: Maximum absolute row sum error
            - max_col_error: Maximum absolute column sum error
            - max_rank_violation: Maximum rank order violation
            - final_change: Final relative change in solution

    Raises:
        CalibrationError: If inputs are invalid, the algorithm fails to converge,
            or other errors occur.
        ValueError: If ties parameter is not "stable" or "group"

    Examples:
        Basic calibration:

        >>> import numpy as np
        >>> from rank_preserving_calibration import calibrate_dykstra
        >>> P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> M = np.array([1.0, 0.7, 0.3])  # Target column sums
        >>> result = calibrate_dykstra(P, M)
        >>> print(f"Converged: {result.converged}")
        >>> print(f"Row sums: {result.Q.sum(axis=1)}")
        >>> print(f"Column sums: {result.Q.sum(axis=0)}")

        With nearly-isotonic constraints:

        >>> result = calibrate_dykstra(P, M, nearly={"mode": "epsilon", "eps": 0.05})

    Notes:
        - Converges to the exact intersection of the constraint sets
        - Preserves ranking within each class (column) by original model scores
        - Memory complexity is O(N*J) for the probability matrices
        - Time complexity per iteration is O(N*J*log(N)) due to sorting
        - For best performance, ensure sum(M) ≈ N and use numba if available
        - Raises CalibrationError on convergence failure rather than returning
          an unreliable result
    """
    _configure_logging(verbose)
    _, J = _validate_inputs(P, M, max_iters, tol, feasibility_tol)
    if ties not in ("stable", "group"):
        raise ValueError(f"ties must be 'stable' or 'group', got '{ties}'")

    P = np.asarray(P, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    # Initialize Dykstra variables
    Q = P.copy()
    U = np.zeros_like(P, dtype=np.float64)  # row memory
    V = np.zeros_like(P, dtype=np.float64)  # col memory
    Q_prev = np.empty_like(Q)

    # Precompute column orders once (stable)
    column_orders = [np.argsort(P[:, j], kind="mergesort") for j in range(J)]
    score_sorted: list[np.ndarray | None] = (
        [P[ord_j, j] for j, ord_j in enumerate(column_orders)]
        if ties == "group"
        else [None] * J
    )

    Q_history: list[NDArrayFloat] | None = [] if detect_cycles else None
    converged = False
    final_change = float("inf")
    iteration = 0

    for iteration in range(1, max_iters + 1):
        np.copyto(Q_prev, Q)

        # Project onto row simplex
        Y = Q + U
        Q = _project_row_simplex(Y, use_jit=use_jit)
        U = Y - Q

        # Project onto column constraints
        Y = Q + V
        for j in range(J):
            Q[:, j] = _project_column_isotonic_sum(
                Y[:, j],
                column_orders[j],
                float(M[j]),
                rtol=rtol,
                nearly=nearly,
                ties=ties,
                score_sorted=score_sorted[j],
            )
        V = Y - Q

        # Convergence check (relative change + feasibility)
        change_abs = np.linalg.norm(Q - Q_prev)
        norm_Q_prev = np.linalg.norm(Q_prev)
        final_change = (
            float(change_abs / norm_Q_prev) if norm_Q_prev > 0 else float(change_abs)
        )

        feasible = is_feasible(Q, M, row_atol=row_atol, col_atol=col_atol)

        if final_change < tol and feasible:
            converged = True
            logger.info("Dykstra converged at iteration %d", iteration)
            break

        # Cycle detection (optional)
        if detect_cycles and iteration > cycle_window and Q_history is not None:
            if _detect_cycling(Q_history, Q):
                warnings.warn(
                    f"Cycling detected at iteration {iteration}",
                    UserWarning,
                    stacklevel=2,
                )
                break
            Q_history.append(Q.copy())
            if len(Q_history) > cycle_window:
                Q_history.pop(0)

        if iteration % 100 == 0 or iteration <= 10:
            logger.debug("Dykstra iteration %d: change = %.2e", iteration, final_change)

        if callback is not None and not callback(iteration, final_change, Q):
            break

    # If not strictly feasible, polish to the intersection
    if not is_feasible(Q, M, row_atol=row_atol, col_atol=col_atol):
        Q = _polish_to_intersection(
            Q,
            M,
            column_orders,
            rtol=rtol,
            ties=ties,
            score_sorted=score_sorted,
            max_iters=100,
            row_atol=row_atol,
            col_atol=col_atol,
        )

        # If now feasible, count as converged for reporting
        if is_feasible(Q, M, row_atol=row_atol, col_atol=col_atol):
            converged = True

    # Diagnostics
    row_sums = Q.sum(axis=1)
    col_sums = Q.sum(axis=0)
    max_row_error = float(np.max(np.abs(row_sums - 1.0)))
    max_col_error = float(np.max(np.abs(col_sums - M)))
    max_rank_violation = _compute_rank_violation(Q, P)

    # Fail fast on non-convergence instead of returning unreliable results
    if not converged:
        raise CalibrationError(
            f"Calibration failed to converge after {iteration} iterations. "
            f"Final change: {final_change:.2e} (tolerance: {tol:.2e}). "
            f"Max row error: {max_row_error:.2e}, max col error: {max_col_error:.2e}. "
            "Try: increasing max_iters, relaxing tol, using nearly-isotonic "
            "constraints (nearly={'mode': 'epsilon', 'eps': 0.01}), or consider "
            "temperature scaling."
        )

    return CalibrationResult(
        Q=Q,
        converged=converged,
        iterations=iteration,
        max_row_error=max_row_error,
        max_col_error=max_col_error,
        max_rank_violation=max_rank_violation,
        final_change=final_change,
    )


# ---------------------------------------------------------------------
# ADMM calibration (penalty-based, snaps to exact projection)
# ---------------------------------------------------------------------


def calibrate_admm(
    P: np.ndarray,
    M: np.ndarray,
    rho: float = 1.0,
    max_iters: int = 1000,
    tol: float = 1e-6,
    rtol: float = 0.0,
    feasibility_tol: float = 0.1,
    verbose: bool = False,
    nearly: dict | None = None,
    ties: str = "stable",
    use_jit: bool = True,
) -> ADMMResult:
    """Calibrate using ADMM-style optimization with penalty methods.

    An alternative to Dykstra's projections that handles row/column sum constraints via
    Lagrange multipliers and rank-preservation through either strict isotonic regression
    or lambda-penalty nearly-isotonic proximal operators.

    The algorithm minimizes ||Q - P||² subject to constraint sets using an augmented
    Lagrangian approach. For final optimality verification, the solution is snapped
    to the exact intersection using a short Dykstra polish.

    Args:
        P: Input probability matrix of shape (N, J). Each row represents predicted class
            probabilities for one instance. Rows need not sum to 1 initially.
        M: Target column sums of shape (J,). Should sum to approximately N for
            feasibility.
        rho: ADMM penalty parameter. Larger values enforce constraints more
            aggressively.
            Default 1.0 works well for most problems.
        max_iters: Maximum number of iterations. Default 1000 is usually sufficient.
        tol: Convergence tolerance for primal/dual residuals. Default 1e-6.
        rtol: Relative tolerance for isotonic violations in PAV. Default 0.0 (strict).
        feasibility_tol: Tolerance for feasibility warnings when sum(M) differs from N.
        verbose: If True, enables debug logging.
        nearly: Optional dict for nearly-isotonic constraints. Use
            {"mode": "lambda", "lam": 1.0} for the lambda-penalty approach,
            allowing soft isotonicity violations.
        ties: How to handle tied scores. "stable" preserves input order, "group" pools
            equal-score instances.
        use_jit: If True and numba is available, uses JIT-compiled functions for speed.

    Returns:
        ADMMResult object containing:
            - Q: Calibrated probability matrix of shape (N, J)
            - converged: Always True (failures raise CalibrationError instead)
            - iterations: Number of iterations performed
            - max_row_error: Maximum absolute row sum error
            - max_col_error: Maximum absolute column sum error
            - max_rank_violation: Maximum rank order violation
            - final_change: Final relative change in solution
            - objective_values: List of objective function values per iteration
            - primal_residuals: List of primal residual norms per iteration
            - dual_residuals: List of dual residual norms per iteration

    Raises:
        CalibrationError: If inputs are invalid, the algorithm fails to converge,
            or other errors occur.
        ValueError: If ties parameter is not "stable" or "group"

    Examples:
        Basic ADMM calibration:

        >>> import numpy as np
        >>> from rank_preserving_calibration import calibrate_admm
        >>> P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> M = np.array([1.0, 0.7, 0.3])
        >>> result = calibrate_admm(P, M)
        >>> print(f"Converged: {result.converged}")
        >>> print(f"Objective values: {result.objective_values[-5:]}")

        With lambda-penalty for soft isotonicity:

        >>> result = calibrate_admm(P, M, nearly={"mode": "lambda", "lam": 2.0})

        Adjusting penalty parameter:

        >>> result = calibrate_admm(P, M, rho=5.0)  # Stronger constraint enforcement

    Notes:
        - Often converges faster than Dykstra for well-conditioned problems
        - Provides convergence diagnostics via objective and residual histories
        - Lambda-penalty mode allows trading off isotonicity for fit quality
        - Final solution is snapped to exact feasible set for optimality
        - Experimental: may need parameter tuning for difficult problems
    """
    _configure_logging(verbose)
    N, J = _validate_inputs(P, M, max_iters, tol, feasibility_tol)
    if ties not in ("stable", "group"):
        raise ValueError(f"ties must be 'stable' or 'group', got '{ties}'")

    P = np.asarray(P, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    # Precompute column orders once (stable)
    column_orders = [np.argsort(P[:, j], kind="mergesort") for j in range(J)]
    score_sorted: list[np.ndarray | None] = (
        [P[ord_j, j] for j, ord_j in enumerate(column_orders)]
        if ties == "group"
        else [None] * J
    )

    # Initialize ADMM variables
    Q = P.copy()
    Z1 = np.ones(N)  # row-sum auxiliaries
    Z2 = M.copy()  # col-sum auxiliaries
    lambda1 = np.zeros(N)  # row multipliers
    lambda2 = np.zeros(J)  # col multipliers

    objective_values: list[float] = []
    primal_residuals: list[float] = []
    dual_residuals: list[float] = []

    # Initialize lambda penalty using match
    match nearly:
        case {"mode": "lambda", **rest}:
            lam_pen = float(rest.get("lam", 1.0))
        case _:
            lam_pen = None

    converged = False
    iteration = 0
    Q_prev = Q.copy()
    for iteration in range(max_iters):
        Q_prev = Q.copy()

        # Q-update: exact minimiser of
        #     0.5||Q - P||_F^2 + (rho/2)||Q 1_J - a||^2 + (rho/2)||Q^T 1_N - b||^2
        # with a = Z1 - lambda1/rho and b = Z2 - lambda2/rho.
        #
        # The previous version computed (P + rho*(a[:, None] + b[None, :])) /
        # (1 + 2*rho), which pulls every ELEMENT toward the row target and the
        # column target. But a[i] targets the row *sum* and b[j] the column
        # *sum*, so that update drove each row toward J*(1 + M_j)/2 rather than
        # 1. It could not satisfy its own constraints for any rho, and
        # calibrate_admm failed on problems as small as N=25.
        #
        # Stationarity gives Q + rho (Q 1)1^T + rho 1(Q^T 1)^T = R, which closes
        # in the row sums, column sums and grand total, so the exact solution
        # needs no linear system and stays O(NJ).
        a_vec = Z1 - lambda1 / rho
        b_vec = Z2 - lambda2 / rho
        R = P + rho * a_vec.reshape(-1, 1) + rho * b_vec.reshape(1, -1)
        s_tot = float(R.sum()) / (1.0 + rho * J + rho * N)
        r_vec = (R.sum(axis=1) - rho * s_tot) / (1.0 + rho * J)
        c_vec = (R.sum(axis=0) - rho * s_tot) / (1.0 + rho * N)
        Q_unconstrained = R - rho * r_vec.reshape(-1, 1) - rho * c_vec.reshape(1, -1)

        # Rank-preserving + nonnegativity
        if lam_pen is not None:
            for j in range(J):
                idx = column_orders[j]
                v_sorted = Q_unconstrained[idx, j]
                # prox_near_isotonic's rho and max_iters belong to the inner
                # ADMM that solves the prox subproblem. They are not this
                # function's outer penalty and iteration budget, and forcing
                # those in would change the prox itself.
                z = prox_near_isotonic(v_sorted, lam_pen)  # preen: allow-dropped-arg
                if isinstance(z, tuple):  # safety with return_info variants
                    z = z[0]
                Q_unconstrained[idx, j] = z
        else:
            for j in range(J):
                idx = column_orders[j]
                v_sorted = Q_unconstrained[idx, j]
                if ties == "group" and score_sorted[j] is not None:
                    score_j = score_sorted[j]
                    if score_j is None:
                        raise ValueError("score_j unexpectedly None")
                    lens = _run_lengths_of_equals(score_j)
                    pos = 0
                    y_group = np.empty(lens.size, dtype=np.float64)
                    for g, L in enumerate(lens):
                        y_group[g] = float(np.mean(v_sorted[pos : pos + L]))
                        pos += int(L)
                    z_group = _isotonic_regression(
                        y_group,
                        rtol=rtol,
                        ties="stable",
                        weights=lens.astype(np.float64),
                    )
                    v_sorted[:] = np.repeat(z_group, lens)
                else:
                    v_sorted[:] = _isotonic_regression(
                        v_sorted, rtol=rtol, ties="stable"
                    )

        Q = np.maximum(Q_unconstrained, 0.0)

        # Z-updates (hard equality constraints)
        row_sums = Q.sum(axis=1)
        col_sums = Q.sum(axis=0)

        Z1_prev = Z1.copy()
        Z2_prev = Z2.copy()

        Z1 = np.ones(N)  # enforce row sums = 1
        Z2 = M.copy()  # enforce col sums = M

        # Multiplier updates
        lambda1 += rho * (row_sums - Z1)
        lambda2 += rho * (col_sums - Z2)

        # Residuals & objective (include λ-penalty term if used)
        primal_res = np.linalg.norm(np.concatenate([row_sums - Z1, col_sums - Z2]))
        dual_res = rho * (np.linalg.norm(Z1 - Z1_prev) + np.linalg.norm(Z2 - Z2_prev))

        obj_val = 0.5 * np.linalg.norm(Q - P) ** 2
        if lam_pen is not None:
            pen = 0.0
            for j in range(J):
                idx = column_orders[j]
                qj = Q[idx, j]
                pen += float(np.maximum(qj[:-1] - qj[1:], 0.0).sum())
            obj_val += lam_pen * pen

        objective_values.append(float(obj_val))
        primal_residuals.append(float(primal_res))
        dual_residuals.append(float(dual_res))

        if iteration % 100 == 0:
            logger.debug(
                "ADMM iter %d: obj=%.3e, primal=%.3e, dual=%.3e",
                iteration,
                obj_val,
                primal_res,
                dual_res,
            )

        if primal_res < tol and dual_res < tol:
            converged = True
            break

    if not converged and verbose:
        warnings.warn(
            f"ADMM failed to converge after {max_iters} iterations",
            UserWarning,
            stacklevel=2,
        )

    # Final change (w.r.t. last iterate in the loop)
    if iteration > 0:
        final_change = float(
            np.linalg.norm(Q - Q_prev) / (1.0 + np.linalg.norm(Q_prev))
        )
    else:
        final_change = float("inf")

    # Snap to the exact projection (guarantees distance optimality over feasible set)
    try:
        # The snap is what makes the returned matrix an actual projection, so it
        # gets a budget proportional to the problem rather than a flat 1500 --
        # Dykstra's iteration count grows superlinearly in N, and a fixed budget
        # silently became inadequate somewhere above N=25.
        snap = calibrate_dykstra(
            P,
            M,
            max_iters=max(1500, 200 * N),
            tol=1e-10,
            rtol=0.0,
            verbose=False,
            detect_cycles=False,
            ties="stable",
            use_jit=use_jit,
            # The snap's Q is what gets returned, so a caller asking for a
            # nearly-isotonic relaxation has to reach it -- otherwise the
            # snap re-imposes strict rank preservation and `nearly` is a
            # silent no-op. tol stays at 1e-10: that is the snap's own
            # exactness budget, not the caller's convergence tolerance.
            nearly=nearly,
            feasibility_tol=feasibility_tol,
        )
        Q = snap.Q
    except CalibrationError as exc:
        # Never fall back silently. The ADMM iterate is not a projection onto the
        # constraint set: its objective can sit *below* the true optimum precisely
        # because it is infeasible. Returning it while reporting converged=True
        # hands back a matrix that violates the guarantee this package exists for
        # -- measured at 27 rank violations on a 25-row problem -- with no signal
        # at the default verbosity.
        raise CalibrationError(
            "ADMM converged but the final projection onto the constraint set did "
            f"not: {exc}. The raw ADMM iterate is not rank-preserving and is not "
            "returned. Use calibrate_dykstra directly, raise max_iters, or relax "
            "the isotonic constraint with nearly={'mode': 'epsilon', 'eps': 0.01}."
        ) from exc

    # Diagnostics
    row_sums = Q.sum(axis=1)
    col_sums = Q.sum(axis=0)
    max_row_error = float(np.max(np.abs(row_sums - 1.0)))
    max_col_error = float(np.max(np.abs(col_sums - M)))
    max_rank_violation = _compute_rank_violation(Q, P)

    # Fail fast on non-convergence instead of returning unreliable results
    if not converged:
        raise CalibrationError(
            f"ADMM calibration failed to converge after {iteration + 1} iterations. "
            f"Final primal residual: {primal_residuals[-1]:.2e}, "
            f"dual residual: {dual_residuals[-1]:.2e} (tolerance: {tol:.2e}). "
            f"Try: increasing max_iters, adjusting rho parameter, relaxing tol, "
            f"or consider Dykstra's method with nearly-isotonic constraints."
        )

    return ADMMResult(
        Q=Q,
        converged=converged,
        iterations=iteration + 1,
        objective_values=objective_values,
        primal_residuals=primal_residuals,
        dual_residuals=dual_residuals,
        max_row_error=max_row_error,
        max_col_error=max_col_error,
        max_rank_violation=max_rank_violation,
        final_change=final_change,
    )
