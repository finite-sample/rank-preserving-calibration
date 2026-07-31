"""The recommended entry point, dispatching to the right solver.

Three solvers exist and they solve the same convex problem. The differences are
practical rather than mathematical:

``qp``
    Sparse quadratic program via an interior-point solver. Exact, and faster
    than the alternatives at every size tested -- 4x at ``N=10``, 13x at
    ``N=25``, 2230x at ``N=50``. The default.
``dykstra``
    Dykstra's alternating projections. Depends only on numpy, and is the
    reference implementation the QP path is tested against. Correct, but the
    iteration count grows superlinearly: 159 iterations at ``N=25``, 6,982 at
    ``N=100``, and past 60,000 at ``N=200``.
``admm``
    Augmented Lagrangian, then a projection. Retained for its residual
    diagnostics; it offers no accuracy or speed advantage.
"""

from __future__ import annotations

import numpy as np

from .calibration import (
    ADMMResult,
    CalibrationError,
    CalibrationResult,
    calibrate_admm,
    calibrate_dykstra,
)
from .qp import calibrate_qp

__all__ = ["calibrate"]

_METHODS = ("auto", "qp", "dykstra", "admm")


def calibrate(
    P: np.ndarray,
    M: np.ndarray,
    *,
    method: str = "auto",
    nearly: dict | None = None,
    **kwargs,
) -> CalibrationResult | ADMMResult:
    """Calibrate a probability matrix, preserving within-class rankings.

    Finds the matrix closest to ``P`` in Frobenius norm whose rows are
    probability distributions, whose column sums equal ``M``, and whose columns
    are non-decreasing in the order of the original scores -- so the ranking of
    individuals within each class survives calibration.

    Args:
        P: Probability matrix of shape (N, J). The within-column orderings of
            ``P`` are what get preserved.
        M: Target column sums of shape (J,). Must sum to ``N``; rows summing to 1
            fix the grand total, so any other value is infeasible. Rescale with
            ``M = M * N / M.sum()``.
        method: ``"auto"`` (default, currently ``"qp"``), ``"qp"``, ``"dykstra"``
            or ``"admm"``. See the module docstring for the trade-offs.
        nearly: None for strict rank preservation, or
            ``{"mode": "epsilon", "eps": e}`` to permit adjacent decreases of at
            most ``e``. Worth reaching for above the size where the exact solve
            becomes expensive.
        **kwargs: Passed through to the chosen solver.

    Returns:
        CalibrationResult (or ADMMResult for ``method="admm"``) holding the
        calibrated matrix and its diagnostics.

    Raises:
        CalibrationError: If ``method`` is unknown, the inputs are invalid, or
            the chosen solver fails.

    Examples:
        >>> import numpy as np
        >>> from rank_preserving_calibration import calibrate
        >>> P = np.array([[0.7, 0.3], [0.4, 0.6], [0.5, 0.5]])
        >>> M = np.array([1.8, 1.2])          # must sum to N = 3
        >>> result = calibrate(P, M)
        >>> result.converged
        True

        Ranking within each class is preserved exactly:

        >>> float(result.max_rank_violation) < 1e-9
        True

        The solvers agree, because they solve the same problem:

        >>> other = calibrate(P, M, method="dykstra")
        >>> bool(np.allclose(result.Q, other.Q, atol=1e-6))
        True
    """
    if method not in _METHODS:
        raise CalibrationError(f"method must be one of {_METHODS}, got {method!r}")

    resolved = "qp" if method == "auto" else method

    if resolved == "qp":
        return calibrate_qp(P, M, nearly=nearly, **kwargs)
    if resolved == "dykstra":
        return calibrate_dykstra(P, M, nearly=nearly, **kwargs)
    return calibrate_admm(P, M, nearly=nearly, **kwargs)
