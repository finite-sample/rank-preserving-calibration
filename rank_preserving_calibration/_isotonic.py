"""Weighted isotonic regression, the one implementation every solver shares.

scipy's PAVA is compiled and widely used; the package carried three
pure-Python copies until 0.11, one of which mishandled weights.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import isotonic_regression


def _isotonic_regression(
    y: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """Fit a nondecreasing sequence to y by weighted least squares.

    Args:
        y: Sequence of shape (n,), already in the order to make monotone.
        weights: Positive weights of shape (n,). If None, all ones.

    Returns:
        The fit z minimising sum_i w_i (z_i - y_i)^2 subject to z
        nondecreasing.

    Raises:
        ValueError: If the weights do not match y or are not positive.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.size <= 1:
        return y.copy()
    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != y.shape:
            raise ValueError("weights must have the same shape as y")
        if np.any(w <= 0):
            raise ValueError("weights must be positive")
    return np.asarray(isotonic_regression(y, weights=w).x, dtype=np.float64)
