#!/usr/bin/env python3
"""
Synthetic "No Free Lunch" experiment for rank-preserving calibration.

Constructs a 3-class Gaussian mixture where the base model outputs the exact
source posterior. Target domain differs only by class priors, so classical
logit-shift yields the exact target posterior. This setup directly tests the
fundamental tradeoff between probabilistic fidelity and rank preservation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent directory to path to import the package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rank_preserving_calibration import (
    CalibrationError,
    calibrate_dykstra,
    isotonic_metrics,
)


def compute_source_posterior(
    X: np.ndarray,
    means: list[np.ndarray],
    covs: list[np.ndarray],
    source_priors: np.ndarray,
) -> np.ndarray:
    """Compute exact Bayes posterior under source priors."""
    N = X.shape[0]
    J = len(means)
    likelihoods = np.zeros((N, J))

    for j in range(J):
        diff = X - means[j]
        cov_inv = np.linalg.inv(covs[j])
        det = np.linalg.det(covs[j])
        exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
        likelihoods[:, j] = np.exp(exponent) / np.sqrt((2 * np.pi) ** X.shape[1] * det)

    joint = likelihoods * source_priors
    posterior = joint / joint.sum(axis=1, keepdims=True)
    return posterior


def compute_target_posterior(
    X: np.ndarray,
    means: list[np.ndarray],
    covs: list[np.ndarray],
    target_priors: np.ndarray,
) -> np.ndarray:
    """Compute exact Bayes posterior under target priors."""
    return compute_source_posterior(X, means, covs, target_priors)


def logit_shift(
    P: np.ndarray, source_priors: np.ndarray, target_priors: np.ndarray
) -> np.ndarray:
    """Classical multiplicative reweighting (label-shift correction)."""
    weights = target_priors / source_priors
    Q = P * weights
    Q = Q / Q.sum(axis=1, keepdims=True)
    return Q


def compute_rank_preservation_rate(Q: np.ndarray, P: np.ndarray) -> float:
    """Fraction of pairwise orderings preserved within each class."""
    N, J = Q.shape
    total_pairs = 0
    preserved = 0

    for j in range(J):
        p_col = P[:, j]
        q_col = Q[:, j]
        for i1 in range(N):
            for i2 in range(i1 + 1, N):
                total_pairs += 1
                p_order = np.sign(p_col[i1] - p_col[i2])
                q_order = np.sign(q_col[i1] - q_col[i2])
                if p_order == q_order or p_order == 0 or q_order == 0:
                    preserved += 1

    return preserved / total_pairs if total_pairs > 0 else 1.0


def run_synthetic(output_dir: str = "output", seed: int = 42) -> dict:
    """Run the synthetic No Free Lunch experiment."""
    np.random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Setup: 3-class Gaussian mixture in 2D
    J = 3
    N = 300

    means = [np.array([0.0, 0.0]), np.array([3.0, 0.0]), np.array([1.5, 2.5])]
    covs = [np.eye(2) * 0.8 for _ in range(J)]

    # Source priors (balanced)
    source_priors = np.array([1 / 3, 1 / 3, 1 / 3])
    # Target priors (shifted)
    target_priors = np.array([0.6, 0.25, 0.15])

    # Generate data from source distribution
    class_assignments = np.random.choice(J, size=N, p=source_priors)
    X = np.zeros((N, 2))
    for i in range(N):
        X[i] = np.random.multivariate_normal(
            means[class_assignments[i]], covs[class_assignments[i]]
        )

    # Compute exact posteriors
    P = compute_source_posterior(X, means, covs, source_priors)
    Q_target_true = compute_target_posterior(X, means, covs, target_priors)

    # Target marginals (expected counts under target distribution)
    M = target_priors * N

    results = []

    # Method 1: Original (no adjustment)
    rmse_orig = np.sqrt(np.mean((P - Q_target_true) ** 2))
    rank_rate_orig = compute_rank_preservation_rate(P, P)
    max_viol_orig = isotonic_metrics(P, P)["max_rank_violation"]
    results.append(
        {
            "method": "original",
            "rmse_to_target": rmse_orig,
            "rank_rate": rank_rate_orig,
            "max_violation": max_viol_orig,
        }
    )

    # Method 2: Logit shift (classical)
    Q_logit = logit_shift(P, source_priors, target_priors)
    rmse_logit = np.sqrt(np.mean((Q_logit - Q_target_true) ** 2))
    rank_rate_logit = compute_rank_preservation_rate(Q_logit, P)
    max_viol_logit = isotonic_metrics(Q_logit, P)["max_rank_violation"]
    results.append(
        {
            "method": "logit_shift",
            "rmse_to_target": rmse_logit,
            "rank_rate": rank_rate_logit,
            "max_violation": max_viol_logit,
        }
    )

    # Method 3: RPC strict
    try:
        res_strict = calibrate_dykstra(P, M, max_iters=5000, tol=1e-8)
        Q_strict = res_strict.Q
        rmse_strict = np.sqrt(np.mean((Q_strict - Q_target_true) ** 2))
        rank_rate_strict = compute_rank_preservation_rate(Q_strict, P)
        max_viol_strict = isotonic_metrics(Q_strict, P)["max_rank_violation"]
    except CalibrationError:
        Q_strict = None
        rmse_strict = np.nan
        rank_rate_strict = np.nan
        max_viol_strict = np.nan
    results.append(
        {
            "method": "rpc_strict",
            "rmse_to_target": rmse_strict,
            "rank_rate": rank_rate_strict,
            "max_violation": max_viol_strict,
        }
    )

    # Methods 4-5: RPC with epsilon-slack
    for eps in [0.01, 0.05]:
        method_name = f"rpc_eps_{eps}"
        try:
            res_eps = calibrate_dykstra(
                P, M, max_iters=10000, tol=1e-6, nearly={"mode": "epsilon", "eps": eps}
            )
            Q_eps = res_eps.Q
            rmse_eps = np.sqrt(np.mean((Q_eps - Q_target_true) ** 2))
            rank_rate_eps = compute_rank_preservation_rate(Q_eps, P)
            max_viol_eps = isotonic_metrics(Q_eps, P)["max_rank_violation"]
        except CalibrationError:
            rmse_eps = np.nan
            rank_rate_eps = np.nan
            max_viol_eps = np.nan
        results.append(
            {
                "method": method_name,
                "rmse_to_target": rmse_eps,
                "rank_rate": rank_rate_eps,
                "max_violation": max_viol_eps,
            }
        )

    # Create results DataFrame
    df = pd.DataFrame(results)
    df.to_csv(output_path / "synthetic_results.csv", index=False)

    # Generate LaTeX table
    latex_table = generate_latex_table(df)
    with open(output_path / "synthetic_table.tex", "w") as f:
        f.write(latex_table)

    # Generate tradeoff plot
    generate_tradeoff_plot(df, output_path / "synthetic_tradeoff.pdf")

    print(f"Synthetic experiment complete. Results saved to {output_path}")
    return {"df": df, "results": results}


def generate_latex_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table from results DataFrame."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Synthetic experiment: tradeoff between RMSE to true target posterior and rank preservation. "
        r"Logit-shift achieves near-zero RMSE but breaks ranks; strict RPC preserves ranks exactly but deviates from the optimal posterior.}",
        r"\label{tab:synthetic}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & RMSE to Target & Rank Rate & Max Violation \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        method = str(row["method"]).replace("_", r"\_")
        rmse = (
            f"{row['rmse_to_target']:.4f}"
            if not np.isnan(row["rmse_to_target"])
            else "---"
        )
        rank = f"{row['rank_rate']:.3f}" if not np.isnan(row["rank_rate"]) else "---"
        viol = (
            f"{row['max_violation']:.4f}"
            if not np.isnan(row["max_violation"])
            else "---"
        )
        lines.append(f"{method} & {rmse} & {rank} & {viol} \\\\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def generate_tradeoff_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Generate tradeoff frontier plot."""
    _, ax = plt.subplots(figsize=(6, 4))

    markers = {
        "original": "o",
        "logit_shift": "s",
        "rpc_strict": "^",
        "rpc_eps_0.01": "d",
        "rpc_eps_0.05": "p",
    }
    colors = {
        "original": "#1f77b4",
        "logit_shift": "#ff7f0e",
        "rpc_strict": "#2ca02c",
        "rpc_eps_0.01": "#d62728",
        "rpc_eps_0.05": "#9467bd",
    }

    for _, row in df.iterrows():
        if np.isnan(row["rmse_to_target"]) or np.isnan(row["rank_rate"]):
            continue
        method = str(row["method"])
        ax.scatter(
            row["rmse_to_target"],
            row["rank_rate"],
            marker=markers.get(method, "o"),
            color=colors.get(method, "gray"),
            s=100,
            label=method.replace("_", " "),
            zorder=5,
        )

    ax.set_xlabel("RMSE to True Target Posterior")
    ax.set_ylabel("Rank Preservation Rate")
    ax.set_title("Tradeoff: Probabilistic Fidelity vs. Rank Preservation")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_synthetic()
