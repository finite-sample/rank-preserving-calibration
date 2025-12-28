# rank_preserving_calibration/ovr_isotonic.py
# This file will contain the implementation of the One-vs-Rest Isotonic Regression calibrator.
from __future__ import annotations

from typing import Any

import numpy as np

from .calibration import _isotonic_regression


def calibrate_ovr_isotonic(
    y: np.ndarray,
    probs: np.ndarray,
) -> dict[str, Any]:
    """
    Calibrates multiclass probabilities using One-vs-Rest Isotonic Regression.

    For each class, this method trains a separate isotonic regression model on
    the binary problem of that class vs. all other classes. The resulting
    calibrated probabilities are then normalized to sum to 1. This is a common
    approach for multiclass calibration and is used by libraries like scikit-learn.

    Args:
        y: True class labels as integers of shape (N,).
        probs: Original probability matrix of shape (N, J).

    Returns:
        A dictionary containing the calibrated probabilities 'Q'.
    """
    y = np.asarray(y, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    _, J = probs.shape

    calibrated_probs = np.zeros_like(probs)

    for j in range(J):
        # 1. Prepare data for the binary (one-vs-rest) problem.
        y_binary = (y == j).astype(int)
        p_j = probs[:, j]

        # 2. Sort the data based on the probabilities for the current class.
        # Use a stable sort to handle ties in probabilities correctly.
        order = np.argsort(p_j, kind="mergesort")
        p_j_sorted = p_j[order]
        y_binary_sorted = y_binary[order]

        # 3. Fit the isotonic regression model.
        # This finds an isotonic (non-decreasing) sequence that best fits the
        # binary labels. This sequence represents the calibrated probabilities
        # for the sorted input probabilities.
        calibrated_p_j_sorted = _isotonic_regression(y_binary_sorted, ties="stable")

        # 4. Create an interpolation function.
        # The sorted probabilities `p_j_sorted` and the calibrated probabilities
        # `calibrated_p_j_sorted` define a step function. We use interpolation
        # to map the original (unsorted) probabilities to their calibrated values.
        # We need to handle duplicate values in `p_j_sorted`.

        unique_p, unique_indices = np.unique(p_j_sorted, return_index=True)
        unique_calibrated_p = calibrated_p_j_sorted[unique_indices]

        calibrated_probs[:, j] = np.interp(p_j, unique_p, unique_calibrated_p)

    # 5. Normalize the rows to sum to 1, as the one-vs-rest procedure
    # does not guarantee that the calibrated probabilities for each
    # instance will sum to 1.
    row_sums = calibrated_probs.sum(axis=1)

    # Avoid division by zero for rows that sum to 0.
    # In such cases, assign uniform probabilities.
    zero_sum_mask = row_sums == 0
    if np.any(zero_sum_mask):
        calibrated_probs[zero_sum_mask, :] = 1.0 / J
        row_sums[zero_sum_mask] = 1.0

    calibrated_probs = calibrated_probs / row_sums[:, np.newaxis]

    return {
        "Q": calibrated_probs,
    }
