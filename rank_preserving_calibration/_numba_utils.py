"""Optional Numba JIT-compiled functions for performance optimization.

This module provides JIT-compiled versions of performance-critical functions.
If Numba is not installed, it gracefully falls back to pure Python versions.
"""

from __future__ import annotations

import numpy as np

# Try to import numba, set flag for availability
try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    # Define identity decorator for graceful fallback
    def njit(*args, **_kwargs):
        """Identity decorator when numba is not available."""

        def decorator(func):
            return func

        return (
            decorator
            if not args
            else decorator(args[0])
            if callable(args[0])
            else decorator
        )


# ---------------------------------------------------------------------
# JIT-compiled simplex projection
# ---------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def project_row_simplex_jit(rows: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """JIT-compiled version of row simplex projection.

    Projects each row onto the probability simplex (sums to 1, non-negative).

    Args:
        rows: Matrix of shape (N, J) to project.
        eps: Small tolerance for numerical stability.

    Returns:
        Projected matrix with the same shape as the input.
    """
    N, J = rows.shape
    projected = np.empty_like(rows)

    for i in range(N):
        v = rows[i].copy()
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1.0

        # Find rho (pivot index)
        rho = J - 1
        for k in range(J):
            if u[k] - cssv[k] / (k + 1.0) > eps:
                rho = k

        # Compute threshold theta
        theta = cssv[rho] / (rho + 1.0)

        # Apply soft thresholding
        for j in range(J):
            projected[i, j] = max(rows[i, j] - theta, 0.0)

        # Normalize to sum to 1
        sum_w = np.sum(projected[i])
        if sum_w > eps:
            for j in range(J):
                projected[i, j] /= sum_w
        else:
            for j in range(J):
                projected[i, j] = 1.0 / J

    return projected


# ---------------------------------------------------------------------
# Export appropriate functions based on Numba availability
# ---------------------------------------------------------------------


def get_jit_functions():
    """Get JIT-compiled functions if available, otherwise None.

    Returns:
        Dictionary mapping function names to implementations. Values are None
        if Numba is not available.
    """
    if HAS_NUMBA:
        return {
            "project_row_simplex": project_row_simplex_jit,
            "available": True,
        }
    return {
        "project_row_simplex": None,
        "available": False,
    }
