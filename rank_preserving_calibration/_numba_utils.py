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
    def njit(*args, **kwargs):
        """Identity decorator when numba is not available."""
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0]) if callable(args[0]) else decorator


# ---------------------------------------------------------------------
# JIT-compiled simplex projection
# ---------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def project_row_simplex_jit(rows: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """JIT-compiled version of row simplex projection.

    Projects each row onto the probability simplex (sums to 1, non-negative).

    Parameters
    ----------
    rows : np.ndarray
        Matrix of shape (N, J) to project.
    eps : float
        Small tolerance for numerical stability.

    Returns
    -------
    np.ndarray
        Projected matrix with same shape as input.
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
# JIT-compiled isotonic regression (PAV algorithm)
# ---------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def isotonic_regression_jit(
    y: np.ndarray,
    weights: np.ndarray | None = None,
    rtol: float = 0.0
) -> np.ndarray:
    """JIT-compiled isotonic regression via Pool Adjacent Violators.

    Computes isotonic (non-decreasing) regression in O(n) time.

    Parameters
    ----------
    y : np.ndarray
        Input values to make isotonic.
    weights : np.ndarray, optional
        Weights for weighted isotonic regression.
    rtol : float
        Relative tolerance for violations.

    Returns
    -------
    np.ndarray
        Isotonic fit with same shape as y.
    """
    n = y.size
    if n <= 1:
        return y.copy()

    # Initialize weights if not provided
    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = weights.astype(np.float64)

    # PAV algorithm using block pooling
    # Arrays for tracking blocks
    block_vals = np.empty(n, dtype=np.float64)
    block_weights = np.empty(n, dtype=np.float64)
    block_starts = np.empty(n, dtype=np.int64)

    n_blocks = 0

    for i in range(n):
        # Start new block
        block_vals[n_blocks] = y[i]
        block_weights[n_blocks] = w[i]
        block_starts[n_blocks] = i
        n_blocks += 1

        # Pool adjacent violators
        while n_blocks > 1:
            # Check if we should merge with previous block
            tol = rtol * (abs(block_vals[n_blocks-2]) + abs(block_vals[n_blocks-1]) + 1.0)
            if block_vals[n_blocks-2] <= block_vals[n_blocks-1] + tol:
                break

            # Merge blocks: weighted average
            w1 = block_weights[n_blocks-2]
            w2 = block_weights[n_blocks-1]
            v1 = block_vals[n_blocks-2]
            v2 = block_vals[n_blocks-1]

            block_vals[n_blocks-2] = (w1 * v1 + w2 * v2) / (w1 + w2)
            block_weights[n_blocks-2] = w1 + w2
            n_blocks -= 1

    # Expand blocks to output
    result = np.empty_like(y)
    for i in range(n_blocks):
        start = block_starts[i]
        end = block_starts[i+1] if i < n_blocks-1 else n
        for j in range(start, end):
            result[j] = block_vals[i]

    return result


# ---------------------------------------------------------------------
# JIT-compiled helper for run-length encoding
# ---------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def run_lengths_jit(x_sorted: np.ndarray) -> np.ndarray:
    """JIT-compiled run-length encoding for sorted arrays.

    Computes lengths of consecutive equal values.

    Parameters
    ----------
    x_sorted : np.ndarray
        Sorted input array.

    Returns
    -------
    np.ndarray
        Array of run lengths.
    """
    n = x_sorted.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    # Count number of runs first
    n_runs = 1
    for i in range(1, n):
        if x_sorted[i] != x_sorted[i-1]:
            n_runs += 1

    # Allocate output
    lengths = np.empty(n_runs, dtype=np.int64)

    # Fill run lengths
    run_idx = 0
    cnt = 1
    for i in range(1, n):
        if x_sorted[i] == x_sorted[i-1]:
            cnt += 1
        else:
            lengths[run_idx] = cnt
            run_idx += 1
            cnt = 1
    lengths[run_idx] = cnt

    return lengths


# ---------------------------------------------------------------------
# Export appropriate functions based on Numba availability
# ---------------------------------------------------------------------

def get_jit_functions():
    """Get JIT-compiled functions if available, otherwise None.

    Returns
    -------
    dict
        Dictionary mapping function names to implementations.
        Values are None if Numba is not available.
    """
    if HAS_NUMBA:
        return {
            'project_row_simplex': project_row_simplex_jit,
            'isotonic_regression': isotonic_regression_jit,
            'run_lengths': run_lengths_jit,
            'available': True
        }
    else:
        return {
            'project_row_simplex': None,
            'isotonic_regression': None,
            'run_lengths': None,
            'available': False
        }
