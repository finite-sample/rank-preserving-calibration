#!/usr/bin/env python3
"""
Real-data shifted deployment experiment for rank-preserving calibration.

Trains Random Forest classifiers on UCI datasets (digits, wine), creates
shifted holdout sets by resampling with altered class composition, and
compares calibration methods.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rank_preserving_calibration import (
    CalibrationError,
    brier,
    calibrate_dykstra,
    isotonic_metrics,
    nll,
    top_label_ece,
)


def create_shifted_test_set(
    X: np.ndarray, y: np.ndarray, target_priors: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Resample test set to match target class priors."""
    classes = np.unique(y)
    J = len(classes)

    target_counts = (target_priors * len(y)).astype(int)
    target_counts[-1] = len(y) - target_counts[:-1].sum()

    X_new = []
    y_new = []

    for j, cls in enumerate(classes):
        mask = y == cls
        indices = np.where(mask)[0]
        n_samples = min(target_counts[j], len(indices))
        if n_samples > 0:
            sampled = rng.choice(indices, size=n_samples, replace=True)
            X_new.append(X[sampled])
            y_new.append(y[sampled])

    return np.vstack(X_new), np.concatenate(y_new)


def logit_shift(
    P: np.ndarray, source_priors: np.ndarray, target_priors: np.ndarray
) -> np.ndarray:
    """Classical multiplicative reweighting."""
    weights = target_priors / (source_priors + 1e-10)
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
        idx = np.argsort(p_col)
        p_sorted = p_col[idx]
        q_sorted = q_col[idx]

        for i1 in range(N):
            for i2 in range(i1 + 1, N):
                total_pairs += 1
                p_order = np.sign(p_sorted[i1] - p_sorted[i2])
                q_order = np.sign(q_sorted[i1] - q_sorted[i2])
                if p_order == q_order or p_order == 0 or q_order == 0:
                    preserved += 1

    return preserved / total_pairs if total_pairs > 0 else 1.0


def compute_col_total_error(Q: np.ndarray, M: np.ndarray) -> float:
    """Maximum absolute column total deviation."""
    return float(np.max(np.abs(Q.sum(axis=0) - M)))


def run_single_dataset(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    target_shift_factor: float,
    n_seeds: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Run experiment on a single dataset with multiple seeds."""
    results = []
    classes = np.unique(y)
    J = len(classes)

    for seed in range(n_seeds):
        seed_rng = np.random.default_rng(seed + 1000)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=seed, stratify=y
        )

        clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        clf.fit(X_train, y_train)

        source_priors = np.bincount(y_train, minlength=J) / len(y_train)

        shift = seed_rng.dirichlet(np.ones(J) * target_shift_factor)
        target_priors = shift / shift.sum()

        X_shifted, y_shifted = create_shifted_test_set(
            X_test, y_test, target_priors, seed_rng
        )

        P = clf.predict_proba(X_shifted)
        N = P.shape[0]
        M = target_priors * N

        actual_counts = np.bincount(y_shifted, minlength=J)

        # Original
        results.append(
            {
                "dataset": name,
                "seed": seed,
                "method": "original",
                "accuracy": float(np.mean(P.argmax(axis=1) == y_shifted)),
                "brier": brier(y_shifted, P),
                "nll": nll(y_shifted, P),
                "ece": top_label_ece(y_shifted, P)["ece"],
                "rank_rate": 1.0,
                "max_rank_violation": 0.0,
                "col_total_error": compute_col_total_error(P, M),
            }
        )

        # Logit shift
        Q_logit = logit_shift(P, source_priors, target_priors)
        iso_logit = isotonic_metrics(Q_logit, P)
        results.append(
            {
                "dataset": name,
                "seed": seed,
                "method": "logit_shift",
                "accuracy": float(np.mean(Q_logit.argmax(axis=1) == y_shifted)),
                "brier": brier(y_shifted, Q_logit),
                "nll": nll(y_shifted, Q_logit),
                "ece": top_label_ece(y_shifted, Q_logit)["ece"],
                "rank_rate": compute_rank_preservation_rate(Q_logit, P),
                "max_rank_violation": iso_logit["max_rank_violation"],
                "col_total_error": compute_col_total_error(Q_logit, M),
            }
        )

        # RPC strict
        try:
            res_strict = calibrate_dykstra(P, M, max_iters=10000, tol=1e-6)
            Q_strict = res_strict.Q
            iso_strict = isotonic_metrics(Q_strict, P)
            results.append(
                {
                    "dataset": name,
                    "seed": seed,
                    "method": "rpc_strict",
                    "accuracy": float(np.mean(Q_strict.argmax(axis=1) == y_shifted)),
                    "brier": brier(y_shifted, Q_strict),
                    "nll": nll(y_shifted, Q_strict),
                    "ece": top_label_ece(y_shifted, Q_strict)["ece"],
                    "rank_rate": compute_rank_preservation_rate(Q_strict, P),
                    "max_rank_violation": iso_strict["max_rank_violation"],
                    "col_total_error": compute_col_total_error(Q_strict, M),
                }
            )
        except CalibrationError:
            results.append(
                {
                    "dataset": name,
                    "seed": seed,
                    "method": "rpc_strict",
                    "accuracy": np.nan,
                    "brier": np.nan,
                    "nll": np.nan,
                    "ece": np.nan,
                    "rank_rate": np.nan,
                    "max_rank_violation": np.nan,
                    "col_total_error": np.nan,
                }
            )

        # RPC epsilon
        try:
            res_eps = calibrate_dykstra(
                P, M, max_iters=10000, tol=1e-6, nearly={"mode": "epsilon", "eps": 0.02}
            )
            Q_eps = res_eps.Q
            iso_eps = isotonic_metrics(Q_eps, P)
            results.append(
                {
                    "dataset": name,
                    "seed": seed,
                    "method": "rpc_eps_0.02",
                    "accuracy": float(np.mean(Q_eps.argmax(axis=1) == y_shifted)),
                    "brier": brier(y_shifted, Q_eps),
                    "nll": nll(y_shifted, Q_eps),
                    "ece": top_label_ece(y_shifted, Q_eps)["ece"],
                    "rank_rate": compute_rank_preservation_rate(Q_eps, P),
                    "max_rank_violation": iso_eps["max_rank_violation"],
                    "col_total_error": compute_col_total_error(Q_eps, M),
                }
            )
        except CalibrationError:
            results.append(
                {
                    "dataset": name,
                    "seed": seed,
                    "method": "rpc_eps_0.02",
                    "accuracy": np.nan,
                    "brier": np.nan,
                    "nll": np.nan,
                    "ece": np.nan,
                    "rank_rate": np.nan,
                    "max_rank_violation": np.nan,
                    "col_total_error": np.nan,
                }
            )

    return results


def run_real_data(output_dir: str = "output", n_seeds: int = 5, seed: int = 42) -> dict:
    """Run the real-data shifted deployment experiment."""
    rng = np.random.default_rng(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Load datasets
    digits = load_digits()
    wine = load_wine()

    datasets = [
        ("digits", digits.data, digits.target, 0.5),
        ("wine", wine.data, wine.target, 0.3),
    ]

    for name, X, y, shift_factor in datasets:
        print(f"Running {name}...")
        results = run_single_dataset(name, X, y, shift_factor, n_seeds, rng)
        all_results.extend(results)

    df = pd.DataFrame(all_results)
    df.to_csv(output_path / "real_data_results.csv", index=False)

    agg_df = (
        df.groupby(["dataset", "method"])
        .agg(
            {
                "accuracy": ["mean", "std"],
                "brier": ["mean", "std"],
                "nll": ["mean", "std"],
                "ece": ["mean", "std"],
                "rank_rate": ["mean", "std"],
                "col_total_error": ["mean", "std"],
            }
        )
        .reset_index()
    )

    latex_table = generate_latex_table(agg_df)
    with open(output_path / "real_data_table.tex", "w") as f:
        f.write(latex_table)

    generate_brier_vs_rank_plot(df, output_path / "brier_vs_rank.pdf")

    print(f"Real-data experiment complete. Results saved to {output_path}")
    return {"df": df, "agg_df": agg_df}


def generate_latex_table(agg_df: pd.DataFrame) -> str:
    """Generate LaTeX table from aggregated results."""
    agg_df = agg_df.copy()
    agg_df.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in agg_df.columns
    ]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Real-data experiments with shifted class distributions. Results averaged over 5 random seeds. "
        r"RPC-strict achieves exact column totals and rank preservation where it converges; on \texttt{digits} (10 classes with extreme shift), strict convergence fails and practitioners should use the nearly-isotonic variant.}",
        r"\label{tab:realdata}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Dataset & Method & Brier $\downarrow$ & ECE $\downarrow$ & Rank Rate $\uparrow$ & Col Err $\downarrow$ \\",
        r"\midrule",
    ]

    datasets = agg_df["dataset"].unique()
    for dataset in datasets:
        subset = agg_df[agg_df["dataset"] == dataset]
        for _, row in subset.iterrows():
            method = str(row["method"]).replace("_", r"\_")
            brier_mean = row["brier_mean"]
            brier_std = row["brier_std"]
            ece_mean = row["ece_mean"]
            ece_std = row["ece_std"]
            rank_mean = row["rank_rate_mean"]
            rank_std = row["rank_rate_std"]
            col_mean = row["col_total_error_mean"]
            col_std = row["col_total_error_std"]

            if np.isnan(brier_mean):
                brier_str = "---"
            else:
                brier_str = f"{brier_mean:.3f}$\\pm${brier_std:.3f}"

            if np.isnan(ece_mean):
                ece_str = "---"
            else:
                ece_str = f"{ece_mean:.3f}$\\pm${ece_std:.3f}"

            if np.isnan(rank_mean):
                rank_str = "---"
            else:
                rank_str = f"{rank_mean:.3f}$\\pm${rank_std:.3f}"

            if np.isnan(col_mean):
                col_str = "---"
            else:
                col_str = f"{col_mean:.2f}$\\pm${col_std:.2f}"

            lines.append(
                f"{dataset} & {method} & {brier_str} & {ece_str} & {rank_str} & {col_str} \\\\"
            )

        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.extend(
        [
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def generate_brier_vs_rank_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Generate Brier score vs rank preservation rate scatter plot."""
    _, ax = plt.subplots(figsize=(6, 4))

    markers = {
        "original": "o",
        "logit_shift": "s",
        "rpc_strict": "^",
        "rpc_eps_0.02": "d",
    }
    colors = {
        "original": "#1f77b4",
        "logit_shift": "#ff7f0e",
        "rpc_strict": "#2ca02c",
        "rpc_eps_0.02": "#d62728",
    }

    agg = df.groupby("method").agg({"brier": "mean", "rank_rate": "mean"}).reset_index()

    for _, row in agg.iterrows():
        method = str(row["method"])
        if np.isnan(row["brier"]) or np.isnan(row["rank_rate"]):
            continue
        ax.scatter(
            row["brier"],
            row["rank_rate"],
            marker=markers.get(method, "o"),
            color=colors.get(method, "gray"),
            s=150,
            label=method.replace("_", " "),
            zorder=5,
        )

    ax.set_xlabel("Brier Score")
    ax.set_ylabel("Rank Preservation Rate")
    ax.set_title("Real Data: Brier Score vs. Rank Preservation")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_real_data()
