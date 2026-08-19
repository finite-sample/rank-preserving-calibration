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

import numpy as np

__all__ = [
    "project_near_isotonic_euclidean",
    "prox_near_isotonic",
    "prox_near_isotonic_with_sum",
]

# ---------------------------------------------------------------------
# Weighted isotonic regression (PAV) in O(n)
# ---------------------------------------------------------------------


def _pav_increasing(
    y: np.ndarray, w: np.ndarray | None = None, rtol: float = 0.0
) -> np.ndarray:
    """Pool-Adjacent-Violators (L2) for a nondecreasing 1D fit, with weights.

    Args:
        y: Sequence of shape (n,) to fit monotonically, already in the
            order you care about.
        w: Positive weights of shape (n,). If None, all ones.
        rtol: Tolerance used in the merge decision -- blocks count as
            monotone when left_mean <= right_mean + rtol *
            (|left_mean| + |right_mean| + 1).

    Returns:
        Isotonic fit of shape (n,) minimizing sum_i w_i * (z_i - y_i)^2
        subject to z nondecreasing.

    Raises:
        ValueError: If the weights are not positive or do not match y.

    Notes:
        - Strict by default (rtol=0.0) to avoid micro-violations in tests.
        - Idempotent: applying to an already isotone sequence returns it
          unchanged.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n <= 1:
        return y.copy()

    if w is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(w, dtype=np.float64)
        if w.shape != y.shape:
            raise ValueError("weights must have the same shape as y")
        if np.any(w <= 0):
            raise ValueError("weights must be strictly positive")

    # Block stacks: start index (in original index space), weighted mean, weight sum.
    start = np.empty(n, dtype=np.int64)
    mean = np.empty(n, dtype=np.float64)
    wsum = np.empty(n, dtype=np.float64)
    top = -1

    def _tol(a: float, b: float) -> float:
        return rtol * (abs(a) + abs(b) + 1.0)

    for i in range(n):
        top += 1
        start[top] = i
        mean[top] = y[i]
        wsum[top] = w[i]

        # Merge backward while violating monotonicity beyond tolerance
        while top > 0:
            left, right = mean[top - 1], mean[top]
            if left <= right + _tol(left, right):
                break
            new_w = wsum[top - 1] + wsum[top]
            mean[top - 1] = (wsum[top - 1] * left + wsum[top] * right) / new_w
            wsum[top - 1] = new_w
            top -= 1

    # Expand the piecewise-constant block means back to length n
    z = np.empty(n, dtype=np.float64)
    for j in range(top + 1):
        s = start[j]
        e = start[j + 1] if j < top else n
        z[s:e] = mean[j]
    return z


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
    uniform shift so 1^T z* = sum_target (translation invariance makes this exact).

    Args:
        v: Input vector of shape (n,) to project.
        eps: Slack parameter (>= 0). Allows z[i+1] >= z[i] - eps rather
            than the strict z[i+1] >= z[i].
        sum_target: If provided, apply an exact uniform shift so the
            result sums to this value.
        weights: Positive weights of shape (n,) for a weighted L2
            projection.

    Returns:
        Projected vector of shape (n,) satisfying the near-isotonic
        constraint, and the sum constraint if one was requested.

    Raises:
        ValueError: If eps is negative or the weights are invalid.

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

    ar = np.arange(n, dtype=np.float64)
    w = v + eps * ar
    iw = _pav_increasing(w, weights, rtol=0.0)
    z = iw - eps * ar

    if sum_target is not None:
        z += (float(sum_target) - float(z.sum())) / n
    return z


# ---------------------------------------------------------------------
# (B) Penalized (λ) nearly isotonic proximal operator
# ---------------------------------------------------------------------


def _diff(z: np.ndarray) -> np.ndarray:
    """Forward differences Dz with (Dz)_i = z_i - z_{i+1}."""
    return z[:-1] - z[1:]


def _diff_adjoint(p: np.ndarray, n: int) -> np.ndarray:
    """Apply the adjoint of the forward difference, D^T p.

    For D with rows [0..0,1,-1,0..0], we have:
      (D^T p)_0     = p_0
      (D^T p)_i     = p_i - p_{i-1},  i = 1..n-2
      (D^T p)_{n-1} = -p_{n-1}
    """
    out = np.empty(n, dtype=p.dtype)
    out[0] = p[0]
    out[1:-1] = p[1:] - p[:-1]
    out[-1] = -p[-1]
    return out


def _solve_tridiag(
    low: np.ndarray, d: np.ndarray, u: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Solve a tridiagonal system with the Thomas algorithm.

      low: sub-diagonal (length n-1)
      d: main diagonal (length n)
      u: super-diagonal (length n-1)
      b: RHS (length n)
    Returns x solving T x = b where T has (low, d, u).
    """
    n = d.size
    # Working copies
    c = u.copy()
    dd = d.copy()
    bb = b.copy()

    # Forward elimination
    for i in range(1, n):
        w = low[i - 1] / dd[i - 1]
        dd[i] -= w * c[i - 1]
        bb[i] -= w * bb[i - 1]

    # Back substitution
    x = np.empty_like(bb)
    x[-1] = bb[-1] / dd[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (bb[i] - c[i] * x[i + 1]) / dd[i]
    return x


def prox_near_isotonic(
    y: np.ndarray,
    lam: float,
    *,
    rho: float = 1.0,
    max_iters: int = 2000,
    abstol: float = 1e-8,
    reltol: float = 1e-6,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    r"""Compute the exact prox of the λ-penalty nearly-isotonic term.

        prox_{λ R}(y),  R(z) = ∑_{i=1}^{n-1} (z_i - z_{i+1})_+,

    where (x)_+ = max(0, x). This penalizes *downward* steps, relaxing strict
    isotonicity.

    We solve:
        minimize_z  0.5 * ||z - y||_2^2  +  λ * ∑ (Dz)_+   with  (Dz)_i = z_i - z_{i+1}
    by ADMM on the split Dz = r:
        z-update:   (I + ρ D^T D) z = y + ρ D^T (r - u)     [SPD tridiagonal solve]
        r-update:   r = prox_{(λ/ρ)‖·‖_{+,1}}(Dz + u)        [one-sided soft-threshold]
        u-update:   u ← u + Dz - r

    One-sided soft-threshold (elementwise):
        prox_{t(·)_+}(v) = { v - t,  if v > t
                             0,       if 0 ≤ v ≤ t
                             v,       if v < 0 }
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n <= 1:
        return (
            (y.copy(), {"iterations": 0, "pr_res": 0.0, "du_res": 0.0})
            if return_info
            else y.copy()
        )
    if lam < 0:
        raise ValueError("lam must be nonnegative.")
    if lam == 0.0:
        return (
            (y.copy(), {"iterations": 0, "pr_res": 0.0, "du_res": 0.0})
            if return_info
            else y.copy()
        )

    # Precompute tridiagonal system for z-update: A = I + ρ D^T D (SPD).
    main = np.empty(n, dtype=np.float64)
    main[0] = 1.0 + rho
    main[1:-1] = 1.0 + 2.0 * rho
    main[-1] = 1.0 + rho
    off = -rho * np.ones(n - 1, dtype=np.float64)

    # Initialize
    z = y.copy()
    r = _diff(z)
    u = np.zeros(n - 1, dtype=np.float64)

    pr_res = du_res = np.nan
    for k in range(1, max_iters + 1):
        # z-update
        rhs = y + rho * _diff_adjoint(r - u, n)
        z = _solve_tridiag(off, main, off, rhs)

        # r-update: one-sided soft-threshold at t = λ/ρ
        v = _diff(z) + u
        t = lam / rho
        r_prev = r
        r = v.copy()
        mask_pos = v > t
        mask_mid = (v >= 0.0) & (~mask_pos)
        r[mask_pos] = v[mask_pos] - t
        r[mask_mid] = 0.0
        # v < 0 unchanged (no penalty)

        # u-update
        Dz = _diff(z)
        u += Dz - r

        # Stopping criteria (Boyd et al.)
        pr = Dz - r
        dr = rho * _diff_adjoint(r - r_prev, n)

        pr_res = np.linalg.norm(pr)
        du_res = np.linalg.norm(dr)
        m = n - 1
        eps_pr = np.sqrt(m) * abstol + reltol * max(
            np.linalg.norm(Dz), np.linalg.norm(r)
        )
        eps_du = np.sqrt(n) * abstol + reltol * np.linalg.norm(
            rho * _diff_adjoint(u, n)
        )

        if pr_res <= eps_pr and du_res <= eps_du:
            if return_info:
                return z, {
                    "iterations": k,
                    "pr_res": float(pr_res),
                    "du_res": float(du_res),
                }
            return z

    # Hit max_iters
    if return_info:
        return z, {
            "iterations": max_iters,
            "pr_res": float(pr_res),
            "du_res": float(du_res),
        }
    return z


def prox_near_isotonic_with_sum(
    y: np.ndarray,
    lam: float,
    sum_target: float,
    *,
    return_info: bool = False,
    **kwargs,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Compute the prox under a **sum constraint**.

        minimize_z  0.5 * ||z - y||_2^2 + λ ∑ (z_i - z_{i+1})_+
        subject to  1^T z = sum_target

    Because R(z) depends only on *differences*, it is translation-invariant:
        R(z + c·1) = R(z).
    Hence the constrained prox is obtained exactly by:
        z* = prox_{λR}(y) + ((sum_target - 1^T prox_{λR}(y)) / n) · 1

    Args:
        y: Input vector of shape (n,).
        lam: Penalty weight on downward steps.
        sum_target: Value the result must sum to.
        return_info: If True, also return the solver's info dict.
        **kwargs: Forwarded to :func:`prox_near_isotonic`.

    Returns:
        ``z*`` when return_info is False (the default), otherwise the
        pair ``(z*, info_dict)``.
    """
    ret = prox_near_isotonic(y, lam, return_info=return_info, **kwargs)
    if return_info:
        z, info = ret if isinstance(ret, tuple) else (ret, {})
        c = (float(sum_target) - float(z.sum())) / z.size
        z_shifted = z + c
        return z_shifted, info
    z = ret if isinstance(ret, np.ndarray) else ret[0]
    c = (float(sum_target) - float(z.sum())) / z.size
    return z + c
