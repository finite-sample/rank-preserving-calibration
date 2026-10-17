"""Robust rank-preserving multiclass probability calibration.

This module provides numerically stable implementations of rank-preserving
calibration by Dykstra's alternating projections.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._isotonic import _isotonic_regression
from .nearly import project_near_isotonic_euclidean

type NDArrayFloat = np.ndarray[Any, np.dtype[np.floating[Any]]]
type ColumnOrders = list[np.ndarray]
type CallbackFunction = Callable[[int, float, np.ndarray], bool] | None

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
        >>> import numpy as np
        >>> from rank_preserving_calibration import calibrate_dykstra
        >>> P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> M = np.array([0.8, 0.8, 0.4])
        >>> result = calibrate_dykstra(P, M)
        >>> if result.converged:
        ...     print(f"Calibration successful in {result.iterations} iterations")
        Calibration successful in 2 iterations
        >>> print(f"Max rank violation: {result.max_rank_violation:.6f}")
        Max rank violation: 0.000000
    """

    Q: np.ndarray
    converged: bool
    iterations: int
    max_row_error: float
    max_col_error: float
    max_rank_violation: float
    final_change: float


class CalibrationError(Exception):
    """Raised when calibration fails due to invalid inputs or numerical issues."""


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


def _validate_arrays(P: np.ndarray, M: np.ndarray) -> tuple[int, int]:
    """Check the shapes and values of P and M, shared by every solver.

    Args:
        P: Input probability matrix to validate.
        M: Target column sums to validate.

    Returns:
        Tuple of (N, J) where N is number of instances and J is number of classes.

    Raises:
        CalibrationError: If P or M is malformed.
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

    return N, J


def _epsilon_from(nearly: dict | None) -> float:
    """Read the isotonic slack out of a ``nearly`` specification.

    Args:
        nearly: None for a strict isotonic constraint, or
            ``{"mode": "epsilon", "eps": float}`` to permit adjacent decreases
            of at most ``eps``.

    Returns:
        The permitted decrease, 0.0 for the strict constraint.

    Raises:
        CalibrationError: If the mode is unsupported or ``eps`` is negative.
    """
    if nearly is None:
        return 0.0
    mode = nearly.get("mode")
    if mode != "epsilon":
        raise CalibrationError(
            f"nearly supports only {{'mode': 'epsilon', 'eps': ...}}; got mode="
            f"{mode!r}. To penalise rank violations instead of bounding them, "
            "use calibrate_soft."
        )
    eps = float(nearly.get("eps", 1e-3))
    if eps < 0.0:
        raise CalibrationError(f"eps must be non-negative, got {eps}")
    return eps


def _validate_inputs(
    P: np.ndarray, M: np.ndarray, max_iters: int, tol: float, feasibility_tol: float
) -> tuple[int, int]:
    """Validate the inputs of a solver whose row and column constraints are hard.

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
    N, J = _validate_arrays(P, M)

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


def _project_row_simplex(rows: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Project rows onto the probability simplex with numerical stability.

    Euclidean projection of each row onto {w >= 0, sum(w) = 1} by the
    sort-and-threshold method (Duchi et al., 2008), done for all rows at once.

    Args:
        rows: Matrix of shape (N, J) where each row will be projected.
        eps: Small tolerance for numerical stability in computations.

    Returns:
        Projected matrix with same shape, where each row sums to 1 and is non-negative.
    """
    rows = np.asarray(rows, dtype=np.float64)
    N, J = rows.shape
    u = -np.sort(-rows, axis=1)
    cssv = np.cumsum(u, axis=1) - 1.0
    cond = u - cssv / np.arange(1, J + 1) > eps
    # rho is the last index meeting the condition; cond[:, 0] is always true
    # since u[0] - (u[0] - 1) = 1.
    rho = J - 1 - np.argmax(cond[:, ::-1], axis=1)
    theta = cssv[np.arange(N), rho] / (rho + 1.0)
    projected = np.maximum(rows - theta[:, None], 0.0)

    # Normalize defensively to exactly sum to 1
    sums = projected.sum(axis=1, keepdims=True)
    ok = sums[:, 0] > eps
    projected[ok] /= sums[ok]
    projected[~ok] = 1.0 / J
    return projected


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
                z_group = _isotonic_regression(y_group, weights=lens.astype(np.float64))
                total_n = int(lens.sum())
                c = (float(target_sum) - float(np.dot(z_group, lens))) / float(total_n)
                z_group_shift = z_group + c
                iso_shifted = np.repeat(z_group_shift, lens)
            else:
                iso = _isotonic_regression(y)
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
        >>> from rank_preserving_calibration import is_feasible
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
        # Stable, like the constraints: tied scores keep their row order, so
        # a tie never reads as a violation.
        idx = np.argsort(P[:, j], kind="mergesort")
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
    feasibility_tol: float = 0.1,
    verbose: bool = False,
    callback: CallbackFunction = None,
    detect_cycles: bool = False,  # default off for determinism
    cycle_window: int = 10,
    nearly: dict | None = None,
    ties: str = "stable",
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
        >>> M = np.array([0.8, 0.8, 0.4])  # target column sums; P's are 1.0, 0.7, 0.3
        >>> result = calibrate_dykstra(P, M)
        >>> print(f"Converged: {result.converged}")
        Converged: True
        >>> result.Q.sum(axis=1).round(6).tolist()
        [1.0, 1.0]
        >>> result.Q.sum(axis=0).round(6).tolist()
        [0.8, 0.8, 0.4]

        With nearly-isotonic constraints:

        >>> result = calibrate_dykstra(P, M, nearly={"mode": "epsilon", "eps": 0.05})

    Notes:
        - Converges to the exact intersection of the constraint sets
        - Preserves ranking within each class (column) by original model scores
        - Memory complexity is O(N*J) for the probability matrices
        - Time complexity per iteration is O(N*J*log(N)) due to sorting
        - Raises CalibrationError on convergence failure rather than returning
          an unreliable result
    """
    _configure_logging(verbose)
    _, J = _validate_inputs(P, M, max_iters, tol, feasibility_tol)
    _epsilon_from(nearly)
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
        Q = _project_row_simplex(Y)
        U = Y - Q

        # Project onto column constraints
        Y = Q + V
        for j in range(J):
            Q[:, j] = _project_column_isotonic_sum(
                Y[:, j],
                column_orders[j],
                float(M[j]),
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
