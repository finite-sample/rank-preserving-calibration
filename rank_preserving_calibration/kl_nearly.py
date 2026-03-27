# rank_preserving_calibration/kl_nearly.py
"""
Nearly isotonic utilities for KL divergence.

This module provides "relaxed" isotonic constraints for KL geometry that allow
small violations of monotonicity. The key difference from Euclidean nearly-isotonic:

- **Euclidean slack**: z[i+1] >= z[i] - eps (additive)
- **KL slack**: z[i+1] >= z[i] * (1 - eps) (multiplicative)

The multiplicative form is natural for KL divergence since KL is invariant
under positive scaling.

Exports:
    - project_near_kl_isotonic: ε-slack KL projection with multiplicative slack
    - prox_kl_near_isotonic: λ-penalty proximal operator for KL
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "project_near_kl_isotonic",
    "prox_kl_near_isotonic",
]


def _kl_pav_increasing(
    y: np.ndarray,
    w: np.ndarray | None = None,
    eps: float = 1e-300,
) -> np.ndarray:
    """KL isotonic regression via Pool Adjacent Violators with geometric mean.

    Args:
        y: Input sequence (positive values).
        w: Positive weights.
        eps: Numerical stability constant.

    Returns:
        Isotonic fit minimizing weighted KL divergence.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n <= 1:
        return np.maximum(y.copy(), eps)

    y = np.maximum(y, eps)

    if w is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(w, dtype=np.float64)

    log_y = np.log(y)

    start = np.empty(n, dtype=np.int64)
    log_sum = np.empty(n, dtype=np.float64)
    wsum = np.empty(n, dtype=np.float64)
    top = -1

    for i in range(n):
        top += 1
        start[top] = i
        log_sum[top] = w[i] * log_y[i]
        wsum[top] = w[i]

        while top > 0:
            left_mean = log_sum[top - 1] / wsum[top - 1]
            right_mean = log_sum[top] / wsum[top]
            if left_mean <= right_mean:
                break
            log_sum[top - 1] += log_sum[top]
            wsum[top - 1] += wsum[top]
            top -= 1

    z = np.empty(n, dtype=np.float64)
    for j in range(top + 1):
        s = start[j]
        e = start[j + 1] if j < top else n
        block_mean = log_sum[j] / wsum[j]
        z[s:e] = np.exp(block_mean)

    return z


def project_near_kl_isotonic(
    v: np.ndarray,
    eps_slack: float,
    sum_target: float | None = None,
    weights: np.ndarray | None = None,
    eps: float = 1e-300,
) -> np.ndarray:
    """Project v onto near-isotonic set with multiplicative slack under KL.

    The constraint is: z[i+1] >= z[i] * (1 - eps_slack)

    This is the KL analog of the Euclidean nearly-isotonic projection.
    The key insight is to transform to log space:
        log(z[i+1]) >= log(z[i]) + log(1 - eps_slack)

    Let delta = -log(1 - eps_slack) > 0. Then:
        log(z[i+1]) >= log(z[i]) - delta

    This is equivalent to the additive slack problem in log space.

    Args:
        v: Input vector (positive values).
        eps_slack: Multiplicative slack parameter (0 to 1).
            eps_slack=0 is strict isotonic, eps_slack=0.1 allows 10% violation.
        sum_target: If provided, apply multiplicative rescaling to hit this sum.
        weights: Optional weights for weighted projection.
        eps: Numerical stability constant.

    Returns:
        Near-isotonic projection in KL sense.
    """
    v = np.asarray(v, dtype=np.float64)
    n = v.size
    if n <= 1:
        z = np.maximum(v.copy(), eps)
        if sum_target is not None:
            z *= sum_target / max(z.sum(), eps)
        return z

    if eps_slack < 0 or eps_slack >= 1:
        raise ValueError("eps_slack must be in [0, 1)")

    v = np.maximum(v, eps)

    if eps_slack == 0:
        z = _kl_pav_increasing(v, weights, eps)
    else:
        delta = -np.log(1 - eps_slack)
        log_v = np.log(v)
        w_shifted = log_v + delta * np.arange(n)

        if weights is None:
            w_weights = np.ones(n, dtype=np.float64)
        else:
            w_weights = np.asarray(weights, dtype=np.float64)

        start = np.empty(n, dtype=np.int64)
        wlog_sum = np.empty(n, dtype=np.float64)
        wsum = np.empty(n, dtype=np.float64)
        top = -1

        for i in range(n):
            top += 1
            start[top] = i
            wlog_sum[top] = w_weights[i] * w_shifted[i]
            wsum[top] = w_weights[i]

            while top > 0:
                left_mean = wlog_sum[top - 1] / wsum[top - 1]
                right_mean = wlog_sum[top] / wsum[top]
                if left_mean <= right_mean:
                    break
                wlog_sum[top - 1] += wlog_sum[top]
                wsum[top - 1] += wsum[top]
                top -= 1

        log_z_shifted = np.empty(n, dtype=np.float64)
        for j in range(top + 1):
            s = start[j]
            e = start[j + 1] if j < top else n
            block_mean = wlog_sum[j] / wsum[j]
            log_z_shifted[s:e] = block_mean

        log_z = log_z_shifted - delta * np.arange(n)
        z = np.exp(log_z)

    if sum_target is not None:
        current_sum = z.sum()
        if current_sum > eps:
            z *= sum_target / current_sum

    return z


def prox_kl_near_isotonic(
    y: np.ndarray,
    lam: float,
    rho: float = 1.0,
    max_iters: int = 1000,
    tol: float = 1e-8,
    eps: float = 1e-300,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Proximal operator for λ-penalty nearly-isotonic term under KL.

    Solves:
        min_z  KL(z||y) + λ * sum_i (z_i - z_{i+1})_+

    where (x)_+ = max(0, x) penalizes rank violations.

    Uses ADMM in the dual space. This is more experimental than the
    epsilon-slack approach.

    Args:
        y: Input vector (positive values).
        lam: Penalty weight for rank violations.
        rho: ADMM penalty parameter.
        max_iters: Maximum ADMM iterations.
        tol: Convergence tolerance.
        eps: Numerical stability constant.
        return_info: If True, also return convergence info dict.

    Returns:
        If return_info=False: z (ndarray)
        If return_info=True: (z, info_dict)
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n <= 1:
        z = np.maximum(y.copy(), eps)
        if return_info:
            return z, {"iterations": 0, "converged": True}
        return z

    if lam < 0:
        raise ValueError("lam must be nonnegative")
    if lam == 0:
        z = np.maximum(y.copy(), eps)
        if return_info:
            return z, {"iterations": 0, "converged": True}
        return z

    y = np.maximum(y, eps)

    z = y.copy()
    r = np.zeros(n - 1, dtype=np.float64)
    u = np.zeros(n - 1, dtype=np.float64)

    converged = False
    for iteration in range(1, max_iters + 1):
        z_prev = z.copy()

        Dz = z[:-1] - z[1:]
        v = Dz + u

        t = lam / rho
        r_new = v.copy()
        mask_pos = v > t
        mask_mid = (v >= 0.0) & (~mask_pos)
        r_new[mask_pos] = v[mask_pos] - t
        r_new[mask_mid] = 0.0
        r = r_new

        target = r - u
        for _ in range(20):
            grad = np.zeros(n, dtype=np.float64)
            grad += np.log(z / y)
            Dz = z[:-1] - z[1:]
            diff_err = Dz - target
            grad[:-1] += rho * diff_err
            grad[1:] -= rho * diff_err

            step = 0.1 / (1 + iteration * 0.1)
            z = z * np.exp(-step * grad)
            z = np.maximum(z, eps)

        Dz = z[:-1] - z[1:]
        u = u + Dz - r

        change = np.linalg.norm(z - z_prev) / (np.linalg.norm(z_prev) + 1e-15)
        if change < tol:
            converged = True
            break

    if return_info:
        return z, {"iterations": iteration, "converged": converged}
    return z
