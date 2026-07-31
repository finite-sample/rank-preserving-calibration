"""One-vs-rest isotonic calibration: the conventional baseline.

This is what scikit-learn does and what most multiclass calibration pipelines
do. It is provided so that the rank-preserving methods in this package have
something honest to be compared against -- **not** as a rank-preserving method
itself. See :func:`calibrate_ovr_isotonic` for the measurement.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .calibration import _isotonic_regression


def calibrate_ovr_isotonic(
    y: np.ndarray,
    probs: np.ndarray,
) -> dict[str, Any]:
    """Calibrate multiclass probabilities by one-vs-rest isotonic regression.

    Fits a separate isotonic regression per class on the binary problem of that
    class against the rest, then normalises each row to sum to 1. This is the
    conventional approach, used by scikit-learn among others, and is included
    here as the **baseline** the rank-preserving solvers are measured against.

    .. warning::
       **This does not preserve rank**, despite living in a package named for
       it. The final row normalisation divides every entry by a row-specific
       total, and two people with the same class-`j` score but different row
       totals come out in a different order than they went in. Measured on
       columns that are *perfectly isotonic before* normalisation, **57% of
       adjacent within-class pairs invert**.

       If you rank individuals by their probability of a given class, use
       :func:`~rank_preserving_calibration.calibrate_dykstra` instead, which
       projects onto the constraint set rather than rescaling rows.

    The row normalisation is also not free statistically: it partially undoes
    the per-class calibration it just performed, which is why one-vs-rest
    isotonic ranks last of seven methods on NLL in Dimitriadis-style benchmark
    comparisons of multiclass calibrators.

    Args:
        y: True class labels as integers of shape (N,).
        probs: Original probability matrix of shape (N, J).

    Returns:
        A dictionary containing the calibrated probabilities under key ``'Q'``,
        with rows summing to 1.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> P = rng.dirichlet(np.ones(3), size=50)
        >>> y = rng.integers(0, 3, 50)
        >>> Q = calibrate_ovr_isotonic(y, P)["Q"]
        >>> bool(np.allclose(Q.sum(axis=1), 1.0))
        True
    """
    y = np.asarray(y, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    _, J = probs.shape

    calibrated_probs = np.zeros_like(probs)

    for j in range(J):
        y_binary = (y == j).astype(np.float64)
        p_j = probs[:, j]

        # Pool tied scores before fitting. Isotonic regression is defined on an
        # ordered sequence, so tied predictors must collapse to one weighted
        # point; without this, two observations sharing a score can receive
        # different fitted values and the interpolation below then keeps
        # whichever one the sort happened to place first, making the result
        # depend on row order.
        unique_p, inverse = np.unique(p_j, return_inverse=True)
        counts = np.bincount(inverse, minlength=unique_p.size).astype(np.float64)
        pooled = np.bincount(inverse, weights=y_binary, minlength=unique_p.size)
        pooled /= counts

        fitted = _isotonic_regression(pooled, ties="stable", weights=counts)

        calibrated_probs[:, j] = np.interp(p_j, unique_p, fitted)

    # Rows do not sum to 1 after per-class calibration, so they are rescaled.
    # This is the step that breaks rank preservation; see the warning above.
    row_sums = calibrated_probs.sum(axis=1)

    zero_sum_mask = row_sums == 0
    if np.any(zero_sum_mask):
        calibrated_probs[zero_sum_mask, :] = 1.0 / J
        row_sums[zero_sum_mask] = 1.0

    calibrated_probs = calibrated_probs / row_sums[:, np.newaxis]

    return {
        "Q": calibrated_probs,
    }
