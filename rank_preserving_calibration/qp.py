"""Exact rank-preserving calibration by sparse quadratic programming.

The problem is a convex QP: minimise ``||Q - P||_F^2`` subject to unit row sums,
column sums equal to ``M``, non-negativity, and within-column isotonicity in the
order of the original scores. Written sparsely it has ``N*J`` variables and about
``2*N*J`` constraint rows, each with at most two non-zeros.

Why an interior-point solver rather than alternating projections
----------------------------------------------------------------
Dykstra's method converges to the same answer -- that is checked in the test
suite -- but the constraint sets here meet at a shallow angle, which is the
regime where first-order methods crawl. Measured at ``J=4`` on feasible targets:

======  ==================  ====================  ==================
``N``   Dykstra             OSQP (first order)    Clarabel (this)
======  ==================  ====================  ==================
100     5.4 s               0.64 s                **0.01 s**
400     fails at 60k iters  23 s, 592 violations  **0.12 s**
1600    --                  267 s, 2710 viol.     **1.43 s**
6400    --                  --                    **50 s**
======  ==================  ====================  ==================

The lesson is about the algorithm class, not the problem: OSQP is also a sparse
QP solver and is also first-order, and it fails *worse* than Dykstra, returning
constraint-violating output while reporting that it ran. An interior-point method
is indifferent to the geometry that stalls both.

There is no size at which the projection method is faster, so this is the default
path. :func:`~rank_preserving_calibration.calibrate_dykstra` is retained as the
reference implementation this solver is tested against.

Limits
------
Interior-point factorisation cost grows steeply, so ~50 s at ``N=6400`` is a real
ceiling, just a much higher one than before. Above it, relax the isotonic
constraint with ``nearly={"mode": "epsilon", "eps": 0.05}`` -- supported here
exactly, by lowering the isotonic bound from 0 to ``-eps``.
"""

from __future__ import annotations

import numpy as np

from .calibration import (
    CalibrationError,
    CalibrationResult,
    _compute_rank_violation,
    _validate_inputs,
    is_feasible,
)

__all__ = ["calibrate_qp"]


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
            f"calibrate_qp supports nearly={{'mode': 'epsilon', ...}}; got mode="
            f"{mode!r}. The lambda-penalty mode changes the objective rather than "
            "the constraint set, so it is not a projection; use calibrate_admm."
        )
    eps = float(nearly.get("eps", 1e-3))
    if eps < 0.0:
        raise CalibrationError(f"eps must be non-negative, got {eps}")
    return eps


def _build_constraints(P: np.ndarray, M: np.ndarray, eps: float):
    """Assemble the sparse constraint system.

    Built vectorised rather than with Python loops: at ``N=6400`` the loop-based
    construction, not the solve, dominated the runtime.

    Args:
        P: Probability matrix of shape (N, J), which also fixes the within-column
            orderings that must be preserved.
        M: Target column sums of shape (J,).
        eps: Permitted adjacent decrease within a column.

    Returns:
        Tuple of (A, b, n_eq, n_ineq) where the first ``n_eq`` rows of ``A x = b``
        are equalities and the remainder are ``A x <= b``.
    """
    import scipy.sparse as sp

    N, J = P.shape
    n = N * J

    # Equalities: N row sums, then J column sums.
    rows = np.repeat(np.arange(N), J)
    cols = np.arange(n)
    A_rows = sp.csr_matrix((np.ones(n), (rows, cols)), shape=(N, n))

    rows = np.repeat(np.arange(J), N)
    cols = (np.arange(N)[None, :] * J + np.arange(J)[:, None]).reshape(-1)
    A_cols = sp.csr_matrix((np.ones(n), (rows, cols)), shape=(J, n))

    A_eq = sp.vstack([A_rows, A_cols], format="csr")
    b_eq = np.concatenate([np.ones(N), np.asarray(M, dtype=float)])

    # Inequalities, all written as `<=`.
    #   isotonic:      Q[lower] - Q[upper] <= eps
    #   non-negative:  -Q <= 0
    if N > 1:
        r_idx, c_idx, vals = [], [], []
        row = 0
        for j in range(J):
            order = np.argsort(P[:, j], kind="mergesort")
            upper = order[1:] * J + j
            lower = order[:-1] * J + j
            k = np.arange(row, row + N - 1)
            r_idx.append(np.concatenate([k, k]))
            c_idx.append(np.concatenate([lower, upper]))
            vals.append(np.concatenate([np.ones(N - 1), -np.ones(N - 1)]))
            row += N - 1
        A_iso = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(r_idx), np.concatenate(c_idx))),
            shape=(row, n),
        )
        b_iso = np.full(row, eps)
    else:
        A_iso = sp.csr_matrix((0, n))
        b_iso = np.zeros(0)

    A_nn = -sp.identity(n, format="csr")
    b_nn = np.zeros(n)

    A_ineq = sp.vstack([A_iso, A_nn], format="csr")
    b_ineq = np.concatenate([b_iso, b_nn])

    A = sp.vstack([A_eq, A_ineq], format="csc")
    b = np.concatenate([b_eq, b_ineq])
    # Read the block sizes off the vectors rather than the matrices: scipy's
    # stubs type `.shape` as optional, and the cone dimensions must be plain
    # ints for the solver anyway.
    return A, b, int(b_eq.size), int(b_ineq.size)


def calibrate_qp(
    P: np.ndarray,
    M: np.ndarray,
    *,
    nearly: dict | None = None,
    row_atol: float = 1e-8,
    col_atol: float = 1e-8,
    solver_tol: float = 1e-10,
    feasibility_tol: float = 0.1,
    verbose: bool = False,
) -> CalibrationResult:
    """Calibrate by solving the projection exactly as a sparse QP.

    This is the recommended solver. It minimises ``||Q - P||_F^2`` over matrices
    with unit row sums, column sums ``M``, non-negative entries, and columns that
    are non-decreasing in the order of the original scores -- the same problem
    :func:`~rank_preserving_calibration.calibrate_dykstra` solves by alternating
    projections, but by interior point, which does not stall on this geometry.

    Args:
        P: Probability matrix of shape (N, J). Rows need not already sum to 1;
            the within-column orderings of ``P`` are what get preserved.
        M: Target column sums of shape (J,). Must sum to ``N``: rows summing to 1
            fix the grand total, so any other total makes the problem infeasible.
        nearly: None for strict isotonicity, or ``{"mode": "epsilon", "eps": e}``
            to permit adjacent decreases of at most ``e``. Handled exactly, by
            lowering the isotonic bound rather than by penalising violations.
        row_atol: Absolute tolerance on ``|row sum - 1|`` when reporting
            convergence.
        col_atol: Absolute tolerance on ``|column sum - M_j|``.
        solver_tol: Interior-point convergence tolerance. The 1e-10 default is
            tighter than the solver's own default because rank preservation is
            the guarantee here: at the looser default the result carries
            isotonic violations of order 1e-9 to 1e-7. The cost is negligible.
        feasibility_tol: Retained for signature compatibility with the other
            solvers; feasibility itself is exact and is checked in validation.
        verbose: Print solver progress.

    Returns:
        CalibrationResult with the calibrated matrix and its diagnostics.

    Raises:
        CalibrationError: If the inputs are invalid, the optional solver
            dependencies are missing, or the solver does not reach an optimum.

    Examples:
        >>> import numpy as np
        >>> from rank_preserving_calibration import calibrate_qp
        >>> P = np.array([[0.7, 0.3], [0.4, 0.6], [0.5, 0.5]])
        >>> M = np.array([1.8, 1.2])          # sums to N = 3
        >>> result = calibrate_qp(P, M)
        >>> result.converged
        True
        >>> bool(np.allclose(result.Q.sum(axis=1), 1.0))
        True
        >>> bool(np.allclose(result.Q.sum(axis=0), M))
        True

        Ranking within each class is preserved exactly:

        >>> float(result.max_rank_violation) < 1e-9
        True
    """
    try:
        import clarabel
        import scipy.sparse as sp
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise CalibrationError(
            "calibrate_qp needs `clarabel` and `scipy`. Install them, or use "
            "calibrate_dykstra, which depends only on numpy."
        ) from exc

    N, J = _validate_inputs(
        P, M, max_iters=1, tol=1e-8, feasibility_tol=feasibility_tol
    )
    P = np.asarray(P, dtype=float)
    M = np.asarray(M, dtype=float)
    eps = _epsilon_from(nearly)

    A, b, n_eq, n_ineq = _build_constraints(P, M, eps)
    n = N * J

    # Clarabel minimises 0.5 x' Pq x + q' x, and wants the upper triangle only.
    # Our objective is ||Q - P||^2 = x'x - 2 P'x + const, so Pq = 2I and q = -2P.
    quad = sp.triu(sp.identity(n, format="csc") * 2.0).tocsc()
    linear = -2.0 * P.reshape(-1)
    cones = [
        clarabel.ZeroConeT(n_eq),  # type: ignore[attr-defined]
        clarabel.NonnegativeConeT(n_ineq),  # type: ignore[attr-defined]
    ]

    settings = clarabel.DefaultSettings()  # type: ignore[attr-defined]
    settings.verbose = bool(verbose)
    # Tighter than the solver's default, and deliberately so. At the default
    # tolerance the returned matrix carries small isotonic violations -- 204 of
    # them at N=400 and 1655 at N=1600, each between 1e-9 and 1e-7. They are
    # numerically tiny, but rank preservation is this package's guarantee, and a
    # guarantee that holds only to 1e-7 is a different guarantee. Tightening to
    # 1e-10 drives them to exactly zero and costs almost nothing: 0.24 s against
    # 0.19 s at N=400, and 2.01 s against 2.03 s at N=1600.
    settings.tol_gap_abs = solver_tol
    settings.tol_gap_rel = solver_tol
    settings.tol_feas = solver_tol

    solver = clarabel.DefaultSolver(  # type: ignore[attr-defined]
        quad, linear, A, b, cones, settings
    )
    solution = solver.solve()
    status = str(solution.status)

    if status != "Solved":
        raise CalibrationError(
            f"QP solver returned status {status!r}. If the targets are feasible "
            "(sum(M) == N), try relaxing the isotonic constraint with "
            "nearly={'mode': 'epsilon', 'eps': 0.01}."
        )

    Q = np.asarray(solution.x, dtype=float).reshape(N, J)
    # The solver works to a numerical tolerance, so tiny negatives are possible;
    # clipping cannot break monotonicity because max(x, 0) is non-decreasing.
    Q = np.maximum(Q, 0.0)

    row_sums = Q.sum(axis=1)
    col_sums = Q.sum(axis=0)
    return CalibrationResult(
        Q=Q,
        converged=is_feasible(Q, M, row_atol=row_atol, col_atol=col_atol),
        iterations=int(getattr(solution, "iterations", 0) or 0),
        max_row_error=float(np.max(np.abs(row_sums - 1.0))),
        max_col_error=float(np.max(np.abs(col_sums - M))),
        max_rank_violation=_compute_rank_violation(Q, P),
        final_change=0.0,
    )
