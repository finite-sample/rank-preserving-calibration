"""Nearly isotonic regression utilities.

This module provides "relaxed" isotonic constraints that allow small
violations of monotonicity—useful when strict isotonicity is too restrictive.

Exports:
    - project_near_isotonic_euclidean: ε-slack projection (exact L2), with an
      optional sum target.
    - prox_near_isotonic: exact proximal operator for λ * sum (z_i - z_{i+1})_+.
    - prox_near_isotonic_with_sum: same prox with an exact sum constraint,
      imposed by translation.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ._isotonic import _isotonic_regression

__all__ = [
    "project_near_isotonic_euclidean",
    "prox_near_isotonic",
    "prox_near_isotonic_with_sum",
]

# ---------------------------------------------------------------------
# (A) Hard-slack (ε) nearly isotonic projection
# ---------------------------------------------------------------------


def project_near_isotonic_euclidean(
    v: np.ndarray,
    eps: float,
    sum_target: float | None = None,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Project v onto the set { z : z_{i+1} >= z_i - eps } in L2.

    Reduction: define w_i = v_i + i * eps. Then the constraint becomes standard
    isotonic on w. Let w* = isotonic(w) (weighted if weights are provided).
    The exact projection is z* = w* - i * eps. If sum_target is given, apply a
    uniform shift so 1^T z* = sum_target. That is exact for uniform weights
    only: the sum constraint's multiplier shifts entry i by mu / w_i.

    Args:
        v: Input vector of shape (n,) to project.
        eps: Slack parameter (>= 0). Allows z[i+1] >= z[i] - eps rather
            than the strict z[i+1] >= z[i].
        sum_target: If provided, apply an exact uniform shift so the
            result sums to this value. Requires uniform weights.
        weights: Positive weights of shape (n,) for a weighted L2
            projection; cannot be combined with sum_target unless uniform.

    Returns:
        Projected vector of shape (n,) satisfying the near-isotonic
        constraint, and the sum constraint if one was requested.

    Raises:
        ValueError: If eps is negative, the weights are invalid, or
            sum_target is combined with non-uniform weights.

    Notes:
        - Idempotent: applying the projection twice gives the same result.
    """
    v = np.asarray(v, dtype=np.float64)
    n = v.size
    if n <= 1:
        z = v.copy()
        if sum_target is not None:
            z += (float(sum_target) - float(z.sum())) / max(n, 1)
        return z
    if eps < 0:
        raise ValueError("eps must be nonnegative.")
    if sum_target is not None and weights is not None and np.ptp(weights) > 0:
        raise ValueError("sum_target requires uniform weights")

    ar = np.arange(n, dtype=np.float64)
    w = v + eps * ar
    iw = _isotonic_regression(w, weights)
    z = iw - eps * ar

    if sum_target is not None:
        z += (float(sum_target) - float(z.sum())) / n
    return z


# ---------------------------------------------------------------------
# (B) Penalized (λ) nearly isotonic proximal operator
# ---------------------------------------------------------------------


def prox_near_isotonic(y: np.ndarray, lam: float) -> np.ndarray:
    r"""Compute the prox of the λ-penalty nearly-isotonic term exactly.

    Solves::

        minimize_z  0.5 * ||z - y||^2 + lam * sum_i (z_i - z_{i+1})_+

    which charges every downward step and leaves upward ones free; lam = 0
    returns y, and lam large enough returns the isotonic regression of y.

    By dynamic programming along the chain, as in Johnson's (2013) algorithm
    for the fused lasso. With f_i the optimal cost of the first i terms as a
    function of z_i, the message passed on is
    m_i(b) = min_a f_i(a) + lam * (a - b)_+, whose derivative is f_i'
    clipped to [-lam, 0] (the fused lasso's symmetric penalty clips to
    [-lam, lam]). f_i' is piecewise linear, so its knots are kept in a deque
    and each step costs amortized O(1). Going back, z_i is z_{i+1} clipped to
    the points where f_i' equals -lam and 0.

    No maintained Python package solves this problem; the R package neariso
    (Hoefling, Tibshirani & Tibshirani) implements their path algorithm.

    Args:
        y: Input vector of shape (n,).
        lam: Penalty weight on downward steps, non-negative.

    Returns:
        The minimiser z, of shape (n,).

    Raises:
        ValueError: If lam is negative.
    """
    y = np.asarray(y, dtype=np.float64)
    if lam < 0:
        raise ValueError("lam must be nonnegative.")
    n = y.size
    if n <= 1 or lam == 0.0:
        return y.copy()

    # f' = a x + b on the segment left of the first knot (aL, bL) and right of
    # the last (aR, bR); each knot stores the change (da, db) in the slope and
    # intercept as x crosses it. Every segment has slope at least 1.
    knots: deque[tuple[float, float, float]] = deque()
    aL, bL = 1.0, -float(y[0])
    aR, bR = 1.0, -float(y[0])
    lower = np.empty(n - 1)
    upper = np.empty(n - 1)

    for i in range(n - 1):
        while knots:
            x, da, db = knots[0]
            if aL * x + bL >= -lam:
                break
            knots.popleft()
            aL += da
            bL += db
        lower[i] = (-lam - bL) / aL
        while knots:
            x, da, db = knots[-1]
            if aR * x + bR <= 0.0:
                break
            knots.pop()
            aR -= da
            bR -= db
        upper[i] = -bR / aR

        # Clip f' to [-lam, 0], then add the next data term's x - y.
        knots.appendleft((lower[i], aL, bL + lam))
        knots.append((upper[i], -aR, -bR))
        yi = float(y[i + 1])
        aL, bL = 1.0, -lam - yi
        aR, bR = 1.0, -yi

    while knots:
        x, da, db = knots[-1]
        if aR * x + bR <= 0.0:
            break
        knots.pop()
        aR -= da
        bR -= db

    z = np.empty(n)
    z[-1] = -bR / aR
    for i in range(n - 2, -1, -1):
        z[i] = min(max(z[i + 1], lower[i]), upper[i])
    return z


def prox_near_isotonic_with_sum(
    y: np.ndarray, lam: float, sum_target: float
) -> np.ndarray:
    """Compute the prox under a sum constraint.

    Solves::

        minimize_z  0.5 * ||z - y||^2 + lam * sum_i (z_i - z_{i+1})_+
        subject to  sum(z) = sum_target

    The penalty depends only on differences, so it is translation-invariant,
    and the constrained prox is the unconstrained one shifted by a constant.

    Args:
        y: Input vector of shape (n,).
        lam: Penalty weight on downward steps.
        sum_target: Value the result must sum to.

    Returns:
        The constrained minimiser.
    """
    z = prox_near_isotonic(y, lam)
    return z + (float(sum_target) - float(z.sum())) / z.size
