# rank_preserving_calibration/metrics.py
from __future__ import annotations

from typing import Any

import numpy as np


def _safe_log(x: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    return np.log(np.clip(x, eps, 1.0))


# ---------- KL Divergence ----------


def kl_divergence(Q: np.ndarray, R: np.ndarray, eps: float = 1e-15) -> float:
    """Compute KL divergence KL(Q||R) = sum Q * log(Q/R).

    The KL divergence measures how much Q differs from R, where R is
    the reference distribution. Uses convention 0 * log(0/x) = 0.

    Args:
        Q: Probability matrix of shape (N, J).
        R: Reference probability matrix of shape (N, J).
        eps: Small constant for numerical stability.

    Returns:
        KL divergence value (non-negative).

    Examples:
        >>> import numpy as np
        >>> from rank_preserving_calibration import kl_divergence
        >>> P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> Q = np.array([[0.6, 0.3, 0.1], [0.4, 0.4, 0.2]])
        >>> kl = kl_divergence(Q, P)
        >>> print(f"KL divergence: {kl:.4f}")
    """
    Q = np.asarray(Q, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    mask = Q > eps
    kl = np.zeros_like(Q)
    kl[mask] = Q[mask] * (_safe_log(Q[mask]) - _safe_log(R[mask]))

    return float(np.sum(kl))


def reverse_kl_divergence(Q: np.ndarray, R: np.ndarray, eps: float = 1e-15) -> float:
    """Compute reverse KL divergence KL(R||Q) = sum R * log(R/Q).

    The reverse KL divergence has different properties than forward KL:
    - Forward KL(Q||R) is mean-seeking (Q spreads over modes of R)
    - Reverse KL(R||Q) is mode-seeking (Q focuses on main mode of R)

    Args:
        Q: Probability matrix of shape (N, J).
        R: Reference probability matrix of shape (N, J).
        eps: Small constant for numerical stability.

    Returns:
        Reverse KL divergence value (non-negative).

    Examples:
        >>> import numpy as np
        >>> from rank_preserving_calibration import reverse_kl_divergence
        >>> P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> Q = np.array([[0.6, 0.3, 0.1], [0.4, 0.4, 0.2]])
        >>> rkl = reverse_kl_divergence(Q, P)
    """
    return kl_divergence(R, Q, eps=eps)


def symmetrized_kl(Q: np.ndarray, R: np.ndarray, eps: float = 1e-15) -> float:
    """Compute symmetrized KL divergence (Jensen-Shannon related).

    Symmetrized KL = 0.5 * (KL(Q||R) + KL(R||Q))

    This is symmetric in Q and R, unlike the standard KL divergence.

    Args:
        Q: Probability matrix of shape (N, J).
        R: Reference probability matrix of shape (N, J).
        eps: Small constant for numerical stability.

    Returns:
        Symmetrized KL divergence value (non-negative).

    Examples:
        >>> import numpy as np
        >>> from rank_preserving_calibration import symmetrized_kl
        >>> P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> Q = np.array([[0.6, 0.3, 0.1], [0.4, 0.4, 0.2]])
        >>> skl = symmetrized_kl(Q, P)
    """
    return 0.5 * (kl_divergence(Q, R, eps) + kl_divergence(R, Q, eps))


# ---------- Feasibility & invariants ----------


def feasibility_metrics(Q: np.ndarray, M: np.ndarray | None = None) -> dict[str, Any]:
    """Compute feasibility metrics for calibrated probability matrix.

    Analyzes how well the probability matrix Q satisfies the row and column
    sum constraints required by rank-preserving calibration.

    Args:
        Q: Calibrated probability matrix of shape (N, J). Should have rows summing
            to 1 and non-negative entries.
        M: Target column sums of shape (J,). If provided, analyzes column sum
            constraint satisfaction.

    Returns:
        Dictionary containing feasibility metrics:
            - "row": Row constraint metrics including max/mean absolute errors and min value
            - "col": Column constraint metrics (if M provided) including various error norms

    Examples:
        >>> import numpy as np
        >>> from rank_preserving_calibration import feasibility_metrics
        >>> Q = np.array([[0.6, 0.3, 0.1], [0.4, 0.4, 0.2]])
        >>> M = np.array([1.0, 0.7, 0.3])
        >>> metrics = feasibility_metrics(Q, M)
        >>> print(f"Max row error: {metrics['row']['max_abs_error']:.6f}")
        >>> print(f"Max col error: {metrics['col']['max_abs_error']:.6f}")
    """
    Q = np.asarray(Q, dtype=np.float64)
    _N, _J = Q.shape
    row_sums = Q.sum(axis=1)
    out = {
        "row": {
            "max_abs_error": float(np.max(np.abs(row_sums - 1.0))),
            "mean_abs_error": float(np.mean(np.abs(row_sums - 1.0))),
            "min_value": float(Q.min()),
        }
    }
    if M is not None:
        M = np.asarray(M, dtype=np.float64)
        col_sums = Q.sum(axis=0)
        err = col_sums - M
        out["col"] = {
            "max_abs_error": float(np.max(np.abs(err))),
            "l1_error": float(np.sum(np.abs(err))),
            "l2_error": float(np.linalg.norm(err)),
            "rel_max_abs_error": float(np.max(np.abs(err) / (np.abs(M) + 1e-15))),
        }
    return out


def isotonic_metrics(Q: np.ndarray, P: np.ndarray) -> dict[str, Any]:
    """Analyze isotonic (rank-preserving) properties of calibrated probabilities.

    Verifies that calibrated probabilities are nondecreasing when ordered by original
    model scores within each class. Computes violation statistics and flatness measures.

    Args:
        Q: Calibrated probability matrix of shape (N, J).
        P: Original probability matrix of shape (N, J) used for score ordering.

    Returns:
        Dictionary containing isotonic metrics:
            - "max_rank_violation": Maximum rank-order violation across all columns
            - "mass_weighted_violation": Sum of violation magnitudes weighted by mass
            - "flat_fraction": Fraction of adjacent pairs that are equal (flatness)
            - "per_class_max_violation": List of max violations per class
            - "per_class_mass_violation": List of mass-weighted violations per class

    Examples:
        >>> import numpy as np
        >>> from rank_preserving_calibration import isotonic_metrics
        >>> P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> Q = np.array([[0.65, 0.25, 0.1], [0.35, 0.45, 0.2]])
        >>> metrics = isotonic_metrics(Q, P)
        >>> print(f"Max violation: {metrics['max_rank_violation']:.6f}")
    """
    Q = np.asarray(Q, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    _N, J = Q.shape
    max_viols = []
    mass_viols = []
    flat_fracs = []
    for j in range(J):
        order = np.argsort(P[:, j], kind="mergesort")
        q = Q[order, j]
        if q.size <= 1:
            max_viols.append(0.0)
            mass_viols.append(0.0)
            flat_fracs.append(1.0)
            continue
        diffs = np.diff(q)
        max_viols.append(float(np.max(np.maximum(0.0, -diffs))))
        mass_viols.append(float(np.sum(np.maximum(0.0, -diffs))))
        flat_fracs.append(float(np.mean(np.isclose(diffs, 0.0))))
    return {
        "max_rank_violation": float(np.max(max_viols)),
        "total_violation_mass": float(np.sum(mass_viols)),
        "mean_flat_fraction": float(np.mean(flat_fracs)),
        "per_class": {
            "max_violation": max_viols,
            "violation_mass": mass_viols,
            "flat_fraction": flat_fracs,
        },
    }


def tie_group_variance(Q: np.ndarray, P: np.ndarray) -> dict[str, Any]:
    """Within-equal-score group variance of Q; for ties='group' this should be ~0."""
    Q = np.asarray(Q, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    _N, J = Q.shape
    per_class = []
    for j in range(J):
        scores = P[:, j]
        order = np.argsort(scores, kind="mergesort")
        s_sorted = scores[order]
        q_sorted = Q[order, j]
        # run-lengths of exact-equal scores
        lens = []
        n = len(s_sorted)
        i = 0
        vars_ = []
        while i < n:
            j2 = i + 1
            while j2 < n and s_sorted[j2] == s_sorted[i]:
                j2 += 1
            seg = q_sorted[i:j2]
            if (j2 - i) > 1:
                vars_.append(float(np.var(seg)))
                lens.append(j2 - i)
            i = j2
        per_class.append(
            {
                "group_count": len(lens),
                "mean_within_group_var": float(np.mean(vars_)) if vars_ else 0.0,
            }
        )
    return {"per_class": per_class}


# ---------- Distances ----------


def distance_metrics(Q: np.ndarray, P: np.ndarray) -> dict[str, float]:
    """Compute distance metrics between original and calibrated probabilities.

    Measures how much the calibration procedure has changed the probability estimates
    using various matrix norms and distance measures.

    Args:
        Q: Calibrated probability matrix of shape (N, J).
        P: Original probability matrix of shape (N, J).

    Returns:
        Dictionary containing distance metrics:
            - "frobenius": Frobenius norm ||Q - P||_F
            - "frobenius_sq": Squared Frobenius norm ||Q - P||²_F
            - "mean_abs": Mean absolute difference
            - "max_abs": Maximum absolute difference

    Examples:
        >>> import numpy as np
        >>> P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> Q = np.array([[0.65, 0.25, 0.1], [0.35, 0.45, 0.2]])
        >>> distances = distance_metrics(Q, P)
        >>> print(f"Frobenius distance: {distances['frobenius']:.4f}")
    """
    Q = np.asarray(Q, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    D = Q - P
    return {
        "frobenius": float(np.linalg.norm(D)),
        "frobenius_sq": float(np.sum(D * D)),
        "mean_abs": float(np.mean(np.abs(D))),
        "max_abs": float(np.max(np.abs(D))),
    }


# ---------- Proper scoring & calibration (labels required) ----------


def nll(y: np.ndarray, probs: np.ndarray) -> float:
    """Compute negative log-likelihood (cross-entropy loss).

    Args:
        y: True class labels as integers of shape (N,).
        probs: Probability matrix of shape (N, J).

    Returns:
        Average negative log-likelihood across all instances.

    Examples:
        >>> import numpy as np
        >>> y = np.array([0, 1, 2])
        >>> probs = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
        >>> loss = nll(y, probs)
        >>> print(f"NLL: {loss:.4f}")
    """
    y = np.asarray(y, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    return float(-np.mean(_safe_log(probs[np.arange(len(y)), y])))


def brier(y: np.ndarray, probs: np.ndarray) -> float:
    """Compute Brier score for multiclass classification.

    The Brier score measures the mean squared difference between predicted
    probabilities and one-hot encoded true labels. Lower is better.

    Args:
        y: True class labels as integers of shape (N,).
        probs: Probability matrix of shape (N, J).

    Returns:
        Brier score (mean squared error).

    Examples:
        >>> import numpy as np
        >>> y = np.array([0, 1, 2])
        >>> probs = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
        >>> score = brier(y, probs)
        >>> print(f"Brier score: {score:.4f}")
    """
    y = np.asarray(y, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    N, _J = probs.shape
    oh = np.zeros_like(probs)
    oh[np.arange(N), y] = 1.0
    return float(np.mean(np.sum((oh - probs) ** 2, axis=1)))


def top_label_ece(y: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> dict:
    """Compute Expected Calibration Error for top-label predictions.

    Bins predictions by confidence (max probability) and measures the gap
    between confidence and accuracy in each bin. ECE is the weighted average
    of these gaps, and MCE is the maximum gap.

    Args:
        y: True class labels as integers of shape (N,).
        probs: Probability matrix of shape (N, J).
        n_bins: Number of confidence bins (default 15).

    Returns:
        Dictionary containing:
            - "ece": Expected Calibration Error (weighted average gap)
            - "mce": Maximum Calibration Error (max gap across bins)
            - "bins": List of (count, mean_confidence, accuracy) per bin

    Examples:
        >>> import numpy as np
        >>> y = np.array([0, 1, 2])
        >>> probs = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
        >>> ece_result = top_label_ece(y, probs)
        >>> print(f"ECE: {ece_result['ece']:.4f}")
    """
    y = np.asarray(y, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    mce = 0.0
    table = []
    for b in range(n_bins):
        mask = idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            table.append((0, np.nan, np.nan))
            continue
        acc = float(correct[mask].mean())
        cbar = float(conf[mask].mean())
        gap = abs(acc - cbar)
        ece += (cnt / len(y)) * gap
        mce = max(mce, gap)
        table.append((cnt, cbar, acc))
    return {"ece": float(ece), "mce": float(mce), "bins": table}


def classwise_ece(
    y: np.ndarray, probs: np.ndarray, n_bins: int = 15, balanced: bool = False
) -> dict:
    """Compute class-wise Expected Calibration Error.

    For each class, bins predictions by the predicted probability for that class
    and measures the gap between predicted probability and actual frequency.
    Aggregates across all classes.

    Args:
        y: True class labels as integers of shape (N,).
        probs: Probability matrix of shape (N, J).
        n_bins: Number of probability bins per class (default 15).
        balanced: If True, weight classes equally; otherwise weight by sample count.

    Returns:
        Dictionary containing:
            - "ece": Aggregated Expected Calibration Error
            - "mce": Maximum Calibration Error across all classes
            - "per_class": List of per-class ECE/MCE metrics and bin tables

    Examples:
        >>> import numpy as np
        >>> y = np.array([0, 0, 1, 1, 2, 2])
        >>> probs = np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1],
        ...                   [0.2, 0.6, 0.2], [0.1, 0.8, 0.1],
        ...                   [0.1, 0.2, 0.7], [0.1, 0.1, 0.8]])
        >>> result = classwise_ece(y, probs)
        >>> print(f"Classwise ECE: {result['ece']:.4f}")
    """
    y = np.asarray(y, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    N, J = probs.shape
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total_weight = 0.0
    ece = 0.0
    mce = 0.0
    per_class = []
    for j in range(J):
        p = probs[:, j]
        label = (y == j).astype(np.float64)
        idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
        class_ece = 0.0
        class_weight = 0.0
        class_mce = 0.0
        rows = []
        for b in range(n_bins):
            mask = idx == b
            cnt = int(mask.sum())
            if cnt == 0:
                rows.append((0, np.nan, np.nan))
                continue
            conf = float(p[mask].mean())
            freq = float(label[mask].mean())
            gap = abs(conf - freq)
            w = (1.0 / J) * (cnt / cnt) if balanced else (cnt / N)
            # balanced: each nonempty bin contributes equally across classes
            class_ece += (1.0 / n_bins) * gap if balanced else w * gap
            class_weight += (1.0 / n_bins) if balanced else w
            class_mce = max(class_mce, gap)
            rows.append((cnt, conf, freq))
        if class_weight > 0:
            per_class.append(
                {
                    "class": j,
                    "ece": float(class_ece),
                    "mce": float(class_mce),
                    "bins": rows,
                }
            )
            ece += class_ece
            total_weight += 1.0 if balanced else 1.0  # both sum per class
        mce = max(mce, class_mce)
    # ece already aggregated per class under chosen scheme
    return {
        "ece": float(ece / (J if balanced else 1.0)),
        "mce": float(mce),
        "per_class": per_class,
    }


# ---------- Sharpness ----------


def sharpness_metrics(probs: np.ndarray) -> dict[str, float]:
    """Compute sharpness metrics for probability predictions.

    Sharpness measures how confident (peaked) the predictions are, independent
    of whether they are correct. Sharper predictions are more informative.

    Args:
        probs: Probability matrix of shape (N, J).

    Returns:
        Dictionary containing:
            - "mean_entropy": Average entropy across predictions (lower = sharper)
            - "mean_max_prob": Average of maximum probabilities (higher = sharper)
            - "mean_margin": Average difference between top two probabilities

    Examples:
        >>> import numpy as np
        >>> probs = np.array([[0.9, 0.05, 0.05], [0.5, 0.3, 0.2]])
        >>> sharp = sharpness_metrics(probs)
        >>> print(f"Mean max prob: {sharp['mean_max_prob']:.3f}")
    """
    probs = np.asarray(probs, dtype=np.float64)
    ent = -np.sum(np.where(probs > 0, probs * np.log(probs), 0.0), axis=1)
    mx = probs.max(axis=1)
    sort = np.sort(probs, axis=1)
    margin = sort[:, -1] - sort[:, -2] if probs.shape[1] >= 2 else sort[:, -1]
    return {
        "mean_entropy": float(np.mean(ent)),
        "mean_max_prob": float(np.mean(mx)),
        "mean_margin": float(np.mean(margin)),
    }


# ---------- Variance & Informativeness ----------


def column_variance(Q: np.ndarray) -> dict[str, Any]:
    """Compute per-column variance of a probability matrix.

    Useful for measuring how "spread out" the calibrated probabilities are
    within each class. Lower variance may indicate a "flatter" solution.

    Args:
        Q: Probability matrix of shape (N, J).

    Returns:
        Dictionary containing:
            - "per_column": List of variance values for each column
            - "mean": Mean variance across columns
            - "min": Minimum column variance
            - "max": Maximum column variance

    Examples:
        >>> import numpy as np
        >>> from rank_preserving_calibration import column_variance
        >>> Q = np.array([[0.6, 0.3, 0.1], [0.4, 0.4, 0.2]])
        >>> var = column_variance(Q)
        >>> print(f"Mean column variance: {var['mean']:.4f}")
    """
    Q = np.asarray(Q, dtype=np.float64)
    col_vars = np.var(Q, axis=0)
    return {
        "per_column": col_vars.tolist(),
        "mean": float(np.mean(col_vars)),
        "min": float(np.min(col_vars)),
        "max": float(np.max(col_vars)),
    }


def informativeness_ratio(Q: np.ndarray, P: np.ndarray) -> dict[str, Any]:
    """Compare variance of Q vs P to measure information preservation.

    The informativeness ratio measures how much of the original probability
    spread is preserved after calibration. A ratio close to 1 means the
    calibration preserved the discriminative structure; a ratio near 0
    indicates a "flat" solution with little discrimination between samples.

    Args:
        Q: Calibrated probability matrix of shape (N, J).
        P: Original probability matrix of shape (N, J).

    Returns:
        Dictionary containing:
            - "total_ratio": Var(Q) / Var(P)
            - "per_column_ratio": Variance ratio for each column
            - "mean_column_ratio": Mean of per-column ratios
            - "interpretation": String describing the result

    Examples:
        >>> import numpy as np
        >>> from rank_preserving_calibration import informativeness_ratio
        >>> P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> Q = np.array([[0.5, 0.35, 0.15], [0.5, 0.35, 0.15]])  # Flat
        >>> ratio = informativeness_ratio(Q, P)
        >>> print(f"Total ratio: {ratio['total_ratio']:.3f}")
    """
    Q = np.asarray(Q, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)

    var_q = float(np.var(Q))
    var_p = float(np.var(P))

    if var_p > 1e-15:
        total_ratio = var_q / var_p
    else:
        total_ratio = float("inf") if var_q > 0 else 1.0

    # Per-column ratios
    col_var_q = np.var(Q, axis=0)
    col_var_p = np.var(P, axis=0)
    per_col_ratios = []
    for vq, vp in zip(col_var_q, col_var_p, strict=True):
        if vp > 1e-15:
            per_col_ratios.append(vq / vp)
        else:
            per_col_ratios.append(float("inf") if vq > 0 else 1.0)

    mean_col_ratio = float(np.mean([r for r in per_col_ratios if np.isfinite(r)]))

    # Interpretation
    if total_ratio >= 0.8:
        interpretation = (
            "High informativeness: calibration preserved discrimination well."
        )
    elif total_ratio >= 0.5:
        interpretation = "Moderate informativeness: some loss of discrimination."
    elif total_ratio >= 0.2:
        interpretation = "Low informativeness: significant flattening occurred."
    else:
        interpretation = "Very low informativeness: solution is nearly flat/uniform."

    return {
        "total_ratio": total_ratio,
        "per_column_ratio": per_col_ratios,
        "mean_column_ratio": mean_col_ratio,
        "interpretation": interpretation,
    }


# ---------- Optional: AUC deltas (labels) ----------


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute one-vs-rest AUC using rank-based method.

    Args:
        scores: Predicted scores of shape (N,).
        labels: Binary labels of shape (N,).

    Returns:
        AUC value, or NaN if only one class is present.
    """
    order = np.argsort(scores, kind="mergesort")
    y = labels[order].astype(np.int64)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    # Rank-based AUC: sum of ranks of positives
    ranks = np.arange(1, len(y) + 1, dtype=np.float64)
    pos_ranks = ranks[y == 1]
    return float((pos_ranks.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auc_deltas(y: np.ndarray, P: np.ndarray, Q: np.ndarray) -> dict[str, Any]:
    """One-vs-rest AUC before vs after; useful to check rank effects of plateaus."""
    y = np.asarray(y, dtype=np.int64)
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    _N, J = Q.shape
    deltas = []
    before = []
    after = []
    for j in range(J):
        labels = (y == j).astype(np.int64)
        a0 = _binary_auc(P[:, j], labels)
        a1 = _binary_auc(Q[:, j], labels)
        before.append(float(a0))
        after.append(float(a1))
        if np.isfinite(a0) and np.isfinite(a1):
            deltas.append(float(a1 - a0))
    return {
        "mean_delta_auc": float(np.nanmean(deltas) if deltas else np.nan),
        "per_class": {
            "before": before,
            "after": after,
            "delta": [
                a1 - a0 if (np.isfinite(a0) and np.isfinite(a1)) else np.nan
                for a0, a1 in zip(before, after, strict=False)
            ],
        },
    }
