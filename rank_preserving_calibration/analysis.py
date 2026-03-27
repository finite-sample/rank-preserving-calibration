# rank_preserving_calibration/analysis.py
"""
Analysis utilities for understanding calibration behavior.

This module provides functions to diagnose and understand calibration outcomes,
particularly the "flatness" problem that occurs when distribution shifts are large.

When target marginals M differ significantly from the empirical marginals of P,
the calibrated solution Q may become "flat" (uniform across rows), losing the
discriminative power of the original predictions. These utilities help:

1. Measure how flat the solution is
2. Quantify the magnitude of the distribution shift
3. Estimate theoretical bounds on expected flatness
"""

from __future__ import annotations

from typing import Any

import numpy as np


def flatness_metrics(
    Q: np.ndarray, P: np.ndarray | None = None, M: np.ndarray | None = None
) -> dict[str, Any]:
    """Measure how "flat" (uninformative) the calibrated solution is.

    A perfectly flat solution has equal values within each column when sorted
    by original scores, meaning all samples receive the same probability for
    each class. This indicates loss of discriminative information.

    Args:
        Q: Calibrated probability matrix of shape (N, J).
        P: Original probability matrix of shape (N, J). If provided, enables
            variance ratio computation.
        M: Target column sums of shape (J,). If provided, enables comparison
            to the uniform solution Q_ij = M_j/N.

    Returns:
        Dictionary containing flatness metrics:
            - "column_variance": Per-column variance of Q values
            - "mean_column_variance": Average variance across columns
            - "total_variance": Total variance of all Q entries
            - "entropy_per_row": Mean entropy of each row
            - "mean_max_prob": Mean of maximum probability in each row
            If P is provided:
                - "variance_ratio": Var(Q) / Var(P), lower means flatter
            If M is provided:
                - "distance_to_uniform": Frobenius distance from uniform solution

    Examples:
        >>> import numpy as np
        >>> from rank_preserving_calibration import flatness_metrics
        >>> Q = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
        >>> P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> metrics = flatness_metrics(Q, P)
        >>> print(f"Variance ratio: {metrics['variance_ratio']:.3f}")
    """
    Q = np.asarray(Q, dtype=np.float64)
    N, _J = Q.shape

    # Per-column variance
    col_variances = np.var(Q, axis=0)

    # Total variance
    total_var = float(np.var(Q))

    # Entropy per row
    Q_safe = np.clip(Q, 1e-15, 1.0)
    row_entropies = -np.sum(Q_safe * np.log(Q_safe), axis=1)
    mean_entropy = float(np.mean(row_entropies))

    # Max prob per row
    mean_max_prob = float(np.mean(np.max(Q, axis=1)))

    result: dict[str, Any] = {
        "column_variance": col_variances.tolist(),
        "mean_column_variance": float(np.mean(col_variances)),
        "total_variance": total_var,
        "entropy_per_row": mean_entropy,
        "mean_max_prob": mean_max_prob,
    }

    # Variance ratio compared to P
    if P is not None:
        P = np.asarray(P, dtype=np.float64)
        P_total_var = float(np.var(P))
        if P_total_var > 1e-15:
            result["variance_ratio"] = total_var / P_total_var
        else:
            result["variance_ratio"] = float("inf") if total_var > 0 else 1.0

        # Per-column variance ratio
        P_col_variances = np.var(P, axis=0)
        col_var_ratios = []
        for q_var, p_var in zip(col_variances, P_col_variances, strict=True):
            if p_var > 1e-15:
                col_var_ratios.append(q_var / p_var)
            else:
                col_var_ratios.append(float("inf") if q_var > 0 else 1.0)
        result["column_variance_ratio"] = col_var_ratios

    # Distance to uniform solution
    if M is not None:
        M = np.asarray(M, dtype=np.float64)
        uniform = np.outer(np.ones(N), M / N)
        result["distance_to_uniform"] = float(np.linalg.norm(Q - uniform))
        result["relative_distance_to_uniform"] = float(
            np.linalg.norm(Q - uniform) / (np.linalg.norm(Q) + 1e-15)
        )

    return result


def marginal_shift_metrics(P: np.ndarray, M: np.ndarray) -> dict[str, Any]:
    """Quantify the distribution shift between P's empirical marginals and M.

    Larger shifts generally require more aggressive calibration adjustments,
    which can lead to flatter solutions. This function helps diagnose whether
    flatness is expected given the shift magnitude.

    Args:
        P: Original probability matrix of shape (N, J).
        M: Target column sums of shape (J,).

    Returns:
        Dictionary containing shift metrics:
            - "empirical_marginals": Column sums of P
            - "target_marginals": M values
            - "marginal_diff": Difference (M - empirical)
            - "l1_shift": L1 norm of the difference
            - "l2_shift": L2 norm of the difference
            - "max_shift": Maximum absolute shift in any column
            - "relative_l2_shift": L2 shift divided by sqrt(N)
            - "per_column_relative_shift": Relative shift per column

    Examples:
        >>> import numpy as np
        >>> from rank_preserving_calibration import marginal_shift_metrics
        >>> P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> M = np.array([0.8, 0.8, 0.4])
        >>> metrics = marginal_shift_metrics(P, M)
        >>> print(f"L2 shift: {metrics['l2_shift']:.3f}")
    """
    P = np.asarray(P, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    N, _J = P.shape

    empirical = P.sum(axis=0)
    diff = M - empirical

    l1_shift = float(np.sum(np.abs(diff)))
    l2_shift = float(np.linalg.norm(diff))
    max_shift = float(np.max(np.abs(diff)))

    # Per-column relative shift
    per_col_relative = np.abs(diff) / (np.abs(empirical) + 1e-15)

    return {
        "empirical_marginals": empirical.tolist(),
        "target_marginals": M.tolist(),
        "marginal_diff": diff.tolist(),
        "l1_shift": l1_shift,
        "l2_shift": l2_shift,
        "max_shift": max_shift,
        "relative_l2_shift": l2_shift / np.sqrt(N),
        "per_column_relative_shift": per_col_relative.tolist(),
        "mean_relative_shift": float(np.mean(per_col_relative)),
    }


def flatness_bound(P: np.ndarray, M: np.ndarray) -> dict[str, Any]:
    """Compute theoretical bound on expected solution flatness.

    This function estimates how flat the calibrated solution is likely to be
    based on the distribution shift magnitude. When the shift is large relative
    to the variance in P, we expect Q to be flat.

    The bound is based on the observation that calibration seeks Q closest to P
    subject to constraints. When constraints force Q far from P, the solution
    tends toward the uniform point Q_ij = M_j/N which satisfies all column
    constraints by construction.

    Args:
        P: Original probability matrix of shape (N, J).
        M: Target column sums of shape (J,).

    Returns:
        Dictionary containing:
            - "shift_magnitude": Normalized L2 shift
            - "variance_p": Total variance of P
            - "expected_variance_ratio": Estimated Var(Q)/Var(P)
            - "flatness_risk": Score from 0 to 1 indicating risk of flat solution
            - "recommendation": String with practical advice

    Examples:
        >>> import numpy as np
        >>> from rank_preserving_calibration import flatness_bound
        >>> P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> M = np.array([0.5, 0.8, 0.7])  # Large shift
        >>> bound = flatness_bound(P, M)
        >>> print(f"Flatness risk: {bound['flatness_risk']:.2f}")
    """
    P = np.asarray(P, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    N, J = P.shape

    # Compute shift magnitude
    empirical = P.sum(axis=0)
    diff = M - empirical
    shift_l2 = float(np.linalg.norm(diff))
    normalized_shift = shift_l2 / N  # Per-sample shift

    # Variance of P
    var_p = float(np.var(P))

    # Expected variance reduction heuristic:
    # When shift is large compared to natural variance, expect flat solution
    # variance_ratio ~ 1 / (1 + k * (shift / var_p))
    # where k is a scaling factor based on problem structure
    if var_p > 1e-15:
        shift_to_var_ratio = normalized_shift / var_p
        expected_var_ratio = 1.0 / (1.0 + shift_to_var_ratio)
    else:
        expected_var_ratio = 1.0

    # Flatness risk score (0 = low risk, 1 = high risk)
    # Based on relative shift magnitude
    mean_marginal = N / J
    relative_shift = shift_l2 / (mean_marginal * J + 1e-15)
    flatness_risk = min(1.0, relative_shift)

    # Generate recommendation
    if flatness_risk < 0.1:
        recommendation = "Low flatness risk. Standard calibration should work well."
    elif flatness_risk < 0.3:
        recommendation = (
            "Moderate flatness risk. Consider soft calibration with tuned penalties."
        )
    elif flatness_risk < 0.6:
        recommendation = (
            "High flatness risk. Recommend soft calibration or two-stage IPF approach."
        )
    else:
        recommendation = (
            "Very high flatness risk. Solution will likely be nearly uniform. "
            "Consider relaxing target marginals or using soft constraints."
        )

    return {
        "shift_magnitude": normalized_shift,
        "variance_p": var_p,
        "expected_variance_ratio": expected_var_ratio,
        "flatness_risk": flatness_risk,
        "recommendation": recommendation,
    }


def compare_calibration_methods(
    P: np.ndarray,
    M: np.ndarray,
    results: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compare multiple calibration results on flatness and constraint satisfaction.

    Useful for deciding which calibration approach is best for a given problem
    by comparing solutions from different methods (Dykstra, ADMM, soft, two-stage).

    Args:
        P: Original probability matrix.
        M: Target column sums.
        results: Dictionary mapping method names to calibrated Q matrices.

    Returns:
        Dictionary with per-method metrics and rankings.

    Examples:
        >>> import numpy as np
        >>> from rank_preserving_calibration import (
        ...     calibrate_dykstra, calibrate_soft, compare_calibration_methods
        ... )
        >>> P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> M = np.array([0.9, 0.8, 0.3])
        >>> Q_dykstra = calibrate_dykstra(P, M).Q
        >>> Q_soft = calibrate_soft(P, M).Q
        >>> comparison = compare_calibration_methods(P, M, {
        ...     "dykstra": Q_dykstra, "soft": Q_soft
        ... })
    """
    P = np.asarray(P, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    comparison: dict[str, Any] = {"methods": {}}

    for name, Q in results.items():
        Q = np.asarray(Q, dtype=np.float64)

        # Constraint satisfaction
        row_sums = Q.sum(axis=1)
        col_sums = Q.sum(axis=0)
        max_row_error = float(np.max(np.abs(row_sums - 1.0)))
        max_col_error = float(np.max(np.abs(col_sums - M)))

        # Rank violations
        max_rank_violation = 0.0
        for j in range(Q.shape[1]):
            idx = np.argsort(P[:, j])
            q_sorted = Q[idx, j]
            if q_sorted.size > 1:
                diffs = np.diff(q_sorted)
                violation = float(np.max(np.maximum(0.0, -diffs)))
                max_rank_violation = max(max_rank_violation, violation)

        # Flatness metrics
        flat = flatness_metrics(Q, P, M)

        # Distance from P
        distance_from_p = float(np.linalg.norm(Q - P))

        comparison["methods"][name] = {
            "max_row_error": max_row_error,
            "max_col_error": max_col_error,
            "max_rank_violation": max_rank_violation,
            "variance_ratio": flat.get("variance_ratio", float("nan")),
            "mean_column_variance": flat["mean_column_variance"],
            "distance_from_p": distance_from_p,
            "distance_to_uniform": flat.get("distance_to_uniform", float("nan")),
        }

    # Rank methods by variance ratio (higher is better = less flat)
    var_ratios = {
        name: m["variance_ratio"] for name, m in comparison["methods"].items()
    }
    sorted_by_variance = sorted(
        var_ratios.items(), key=lambda x: x[1] if np.isfinite(x[1]) else 0, reverse=True
    )
    comparison["ranking_by_informativeness"] = [name for name, _ in sorted_by_variance]

    # Rank by constraint satisfaction (lower error is better)
    errors = {
        name: m["max_row_error"] + m["max_col_error"]
        for name, m in comparison["methods"].items()
    }
    sorted_by_error = sorted(errors.items(), key=lambda x: x[1])
    comparison["ranking_by_feasibility"] = [name for name, _ in sorted_by_error]

    return comparison
