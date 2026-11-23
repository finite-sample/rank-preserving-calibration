#!/usr/bin/env python3
"""Generate all manuscript tables, figures, and LaTeX macros from benchmark CSVs.

Reads committed CSVs in ``ms/experiments/output/`` and writes:
- ``ms/generated/numbers.tex``: every computed statistic quoted in prose
- ``ms/generated/*.tex``: booktabs tables via ``pandas.DataFrame.style.to_latex``
- ``ms/generated/*.pdf``: publication figures following a single visual grammar
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent / "output"
GEN_DIR = Path(__file__).resolve().parents[1] / "generated"

METHOD_LABELS = {
    "orig": "Original",
    "ipf": "IPF",
    "temp_ipf": "Temp + IPF",
    "euc_strict": "Euclidean RPC",
    "kl_weak": "KL weak (strict)",
    "kl_soft": "KL soft ($\\lambda=2$)",
    "temp_kl_soft": "Temp + KL soft",
}
METHOD_LABELS_TABLE = {
    "orig": "Original",
    "ipf": "IPF",
    "temp_ipf": "Temp + IPF",
    "euc_strict": "Euclidean RPC",
    "kl_weak": "KL weak (strict)",
    "kl_soft": "KL soft",
    "temp_kl_soft": "Temp + KL soft",
}
METHOD_LABELS_PLAIN = {
    "orig": "Original",
    "ipf": "IPF",
    "temp_ipf": "Temp + IPF",
    "euc_strict": "Euclidean RPC",
    "kl_weak": "KL weak (strict)",
    "kl_soft": "KL soft (λ=2)",
    "temp_kl_soft": "Temp + KL soft",
}
METHOD_ORDER = [
    "orig",
    "ipf",
    "temp_ipf",
    "euc_strict",
    "kl_weak",
    "kl_soft",
    "temp_kl_soft",
]
MODEL_ORDER = ["LogReg", "RF", "HGB", "MLP"]
MODEL_COLORS = {
    "LogReg": "#0072B2",
    "RF": "#009E73",
    "HGB": "#D55E00",
    "MLP": "#CC79A7",
}
MODEL_MARKERS = {
    "LogReg": "o",
    "RF": "s",
    "HGB": "^",
    "MLP": "D",
}
DATASET_LABELS = {
    "wine": "Wine (J=3)",
    "cal_housing": "CA Housing (J=4)",
    "adult_census": "US Census CPS (J=4)",
    "20news_pol": "20 Newsgroups (J=4)",
    "covtype": "Forest Cover (J=7)",
    "digits": "Digits (J=10)",
}
REGIME_LABELS = {
    "label_shift": "Label shift",
    "covariate_shift": "Covariate shift",
}
SCENARIO_LABELS = {
    "pure_label_shift": "Label shift (calib.)",
    "overconfident_label_shift": "Label shift (overconf.)",
    "covariate_shift": "Covariate shift",
}

THEME = {
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
}


def paired_bootstrap_ci(
    diffs: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> tuple[float, float, float]:
    vals = diffs[np.isfinite(diffs)]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(vals))
    if len(vals) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    boot_means = vals[idx].mean(axis=1)
    lo, hi = np.quantile(boot_means, [0.025, 0.975])
    return mean, float(lo), float(hi)


def style_to_latex(
    frame: pd.DataFrame,
    *,
    column_format: str,
    escape_cols: bool = True,
    **kwargs,
) -> str:
    styler = frame.style.hide(axis="index", names=True).format_index(
        escape="latex", axis="index"
    )
    if escape_cols:
        styler = styler.format_index(escape="latex", axis="columns")
    return styler.to_latex(
        hrules=True, multirow_align="naive", column_format=column_format, **kwargs
    )


def build_synthetic_table(df_syn: pd.DataFrame) -> None:
    order_scen = ["pure_label_shift", "overconfident_label_shift", "covariate_shift"]
    order_meth = ["orig", "ipf", "euc_strict", "kl_weak", "kl_soft"]
    records = []
    idx_tuples = []
    for scen in order_scen:
        sub_s = df_syn[df_syn["scenario"] == scen]
        for meth in order_meth:
            sub = sub_s[sub_s["method"] == meth]
            conv = sub["converged"].mean() * 100.0
            ok = sub[sub["converged"]]
            records.append(
                {
                    "RMSE to truth": f"{ok['rmse_true'].mean():.4f}"
                    if len(ok)
                    else "--",
                    "Brier": f"{ok['brier'].mean():.4f}" if len(ok) else "--",
                    "NLL": f"{ok['nll'].mean():.4f}" if len(ok) else "--",
                    "Rank rate": f"{ok['rank_rate'].mean():.4f}" if len(ok) else "--",
                    "Max viol.": f"{ok['max_viol'].mean():.3f}" if len(ok) else "--",
                    "Col err": (
                        f"{ok['col_err'].mean():.2f}"
                        if len(ok) and ok["col_err"].mean() >= 0.01
                        else ("$<0.01$" if len(ok) else "--")
                    ),
                    "Conv. (%)": f"{conv:.0f}",
                }
            )
            idx_tuples.append((SCENARIO_LABELS[scen], METHOD_LABELS_TABLE[meth]))
    frame = pd.DataFrame(
        records,
        index=pd.MultiIndex.from_tuples(idx_tuples, names=["Scenario", "Method"]),
    )
    (GEN_DIR / "synthetic_table.tex").write_text(
        style_to_latex(frame, column_format="llrrrrrrr")
    )


def build_real_summary_table(df_real: pd.DataFrame) -> None:
    """Regime-level matched-pair summary vs. IPF across all real datasets, models, and seeds."""
    records = []
    idx_tuples = []
    for regime in ("label_shift", "covariate_shift"):
        sub_r = df_real[df_real["regime"] == regime]
        ipf_ok = sub_r[(sub_r["method"] == "ipf") & sub_r["converged"]].set_index(
            ["dataset", "model", "seed"]
        )
        for meth in METHOD_ORDER:
            sub = sub_r[sub_r["method"] == meth]
            conv = sub["converged"].mean() * 100.0
            ok = sub[sub["converged"]].set_index(["dataset", "model", "seed"])
            common = ipf_ok.index.intersection(ok.index)
            if meth == "ipf":
                db_str = "0.0 (ref)"
                dn_str = "0.0 (ref)"
            elif len(common):
                db_m, db_lo, db_hi = paired_bootstrap_ci(
                    (ipf_ok.loc[common, "brier"] - ok.loc[common, "brier"]).to_numpy()
                    * 1000.0
                )
                dn_m, dn_lo, dn_hi = paired_bootstrap_ci(
                    (ipf_ok.loc[common, "nll"] - ok.loc[common, "nll"]).to_numpy()
                    * 1000.0
                )
                db_str = f"{db_m:+.1f} [{db_lo:+.1f}, {db_hi:+.1f}]"
                dn_str = f"{dn_m:+.1f} [{dn_lo:+.1f}, {dn_hi:+.1f}]"
            else:
                db_str, dn_str = "--", "--"
            records.append(
                {
                    "Conv. (\\%)": f"{conv:.1f}",
                    "$\\Delta$Brier ($\\times 10^3$)": db_str,
                    "$\\Delta$NLL ($\\times 10^3$)": dn_str,
                    "AUROC": f"{ok['auroc'].mean():.4f}" if len(ok) else "--",
                    "Rank (\\%)": f"{ok['rank_rate'].mean() * 100:.1f}"
                    if len(ok)
                    else "--",
                    "Max viol.": f"{ok['max_viol'].mean():.3f}" if len(ok) else "--",
                    "Col err": (
                        f"{ok['col_err'].mean():.2f}"
                        if len(ok) and ok["col_err"].mean() >= 0.01
                        else ("$<0.01$" if len(ok) else "--")
                    ),
                }
            )
            idx_tuples.append((REGIME_LABELS[regime], METHOD_LABELS_TABLE[meth]))
    frame = pd.DataFrame(
        records,
        index=pd.MultiIndex.from_tuples(idx_tuples, names=["Regime", "Method"]),
    )
    (GEN_DIR / "real_summary_table.tex").write_text(
        style_to_latex(frame, column_format="llrrrrrrr", escape_cols=False)
    )


def build_real_by_dataset_table(df_real: pd.DataFrame) -> None:
    """Dataset x regime detail comparing key methods (IPF, Temp+IPF, Euc, KL weak, KL soft, Temp+KL soft)."""
    shown = ["ipf", "temp_ipf", "euc_strict", "kl_weak", "kl_soft", "temp_kl_soft"]
    records = []
    idx_tuples = []
    for regime in ("label_shift", "covariate_shift"):
        sub_r = df_real[df_real["regime"] == regime]
        for ds in DATASET_LABELS:
            sub_d = sub_r[sub_r["dataset"] == ds]
            if sub_d.empty:
                continue
            for meth in shown:
                sub = sub_d[sub_d["method"] == meth]
                conv = sub["converged"].mean() * 100.0
                ok = sub[sub["converged"]]
                records.append(
                    {
                        "Brier": f"{ok['brier'].mean():.4f}" if len(ok) else "--",
                        "NLL": f"{ok['nll'].mean():.4f}" if len(ok) else "--",
                        "AUROC": f"{ok['auroc'].mean():.4f}" if len(ok) else "--",
                        "Rank rate": f"{ok['rank_rate'].mean():.4f}"
                        if len(ok)
                        else "--",
                        "Max viol.": f"{ok['max_viol'].mean():.3f}"
                        if len(ok)
                        else "--",
                        "Conv. (%)": f"{conv:.0f}",
                    }
                )
                idx_tuples.append(
                    (
                        f"{DATASET_LABELS[ds]} ({'Label' if regime == 'label_shift' else 'Cov.'})",
                        METHOD_LABELS_TABLE[meth],
                    )
                )
    frame = pd.DataFrame(
        records,
        index=pd.MultiIndex.from_tuples(
            idx_tuples, names=["Dataset (Regime)", "Method"]
        ),
    )
    (GEN_DIR / "real_by_dataset_table.tex").write_text(
        style_to_latex(
            frame,
            environment="longtable",
            column_format="llrrrrrr",
            caption=(
                "Mean real-data benchmark performance by dataset and shift regime "
                "(averaged across four model families and ten seeds; converged solves only; "
                r"KL soft uses $\lambda=2$). ``--'' indicates zero converged solves."
            ),
            label="tab:real_by_dataset",
        )
    )


def build_census_table(df_real: pd.DataFrame) -> None:
    """Dedicated US Census CPS (Male -> Female) table by model family."""
    sub = df_real[
        (df_real["dataset"] == "adult_census")
        & (df_real["regime"] == "covariate_shift")
    ]
    shown = [
        "orig",
        "ipf",
        "temp_ipf",
        "euc_strict",
        "kl_weak",
        "kl_soft",
        "temp_kl_soft",
    ]
    records = []
    idx_tuples = []
    for model in MODEL_ORDER:
        sub_m = sub[sub["model"] == model]
        for meth in shown:
            s = sub_m[sub_m["method"] == meth]
            ok = s[s["converged"]]
            records.append(
                {
                    "Brier": f"{ok['brier'].mean():.4f}" if len(ok) else "--",
                    "NLL": f"{ok['nll'].mean():.4f}" if len(ok) else "--",
                    "ECE": f"{ok['ece'].mean():.4f}" if len(ok) else "--",
                    "AUROC": f"{ok['auroc'].mean():.4f}" if len(ok) else "--",
                    "Rank rate": f"{ok['rank_rate'].mean():.4f}" if len(ok) else "--",
                    "Max viol.": f"{ok['max_viol'].mean():.3f}" if len(ok) else "--",
                    "Col err": (
                        f"{ok['col_err'].mean():.2f}"
                        if len(ok) and ok["col_err"].mean() >= 0.01
                        else ("$<0.01$" if len(ok) else "--")
                    ),
                }
            )
            idx_tuples.append((model, METHOD_LABELS_TABLE[meth]))
    frame = pd.DataFrame(
        records,
        index=pd.MultiIndex.from_tuples(idx_tuples, names=["Model", "Method"]),
    )
    (GEN_DIR / "census_table.tex").write_text(
        style_to_latex(frame, column_format="llrrrrrrr")
    )


def build_figures(
    df_syn: pd.DataFrame,
    df_syn_par: pd.DataFrame,
    df_real: pd.DataFrame,
    df_real_par: pd.DataFrame,
) -> None:
    with mpl.rc_context(THEME):
        # Figure 1: Synthetic lambda-path (2 rows x 3 scenarios: RMSE to truth + Max weak-order violation)
        fig, axes = plt.subplots(
            2, 3, figsize=(7.2, 4.0), sharex=True, layout="constrained"
        )
        lam_ticks = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, float("inf")]
        x_pos = np.arange(len(lam_ticks))
        tick_idx = [0, 2, 5, 7, 10]
        tick_lbl = ["IPF\n(0)", "0.1", "1", "5", "Strict\n(∞)"]
        for c_idx, scen in enumerate(
            ["pure_label_shift", "overconfident_label_shift", "covariate_shift"]
        ):
            sp_df = df_syn_par[df_syn_par["scenario"] == scen]
            agg = (
                sp_df.groupby("lam", as_index=False)[["rmse_true", "max_viol"]]
                .mean()
                .sort_values("lam")
            )
            agg_map = {float(r["lam"]): r for _, r in agg.iterrows()}
            rmse_vals = [agg_map[lv]["rmse_true"] for lv in lam_ticks]
            viol_vals = [agg_map[lv]["max_viol"] for lv in lam_ticks]

            ax0 = axes[0, c_idx]
            orig_rmse = df_syn[
                (df_syn["scenario"] == scen) & (df_syn["method"] == "orig")
            ]["rmse_true"].mean()
            ax0.axhline(orig_rmse, color="0.55", ls="--", lw=0.9)
            ax0.text(
                x_pos[-1],
                orig_rmse + 0.004,
                "Original",
                fontsize=7.5,
                color="0.4",
                ha="right",
                va="bottom",
            )
            ax0.plot(x_pos, rmse_vals, "-o", color="#0072B2", lw=1.4, ms=3.5)
            ax0.set_title(SCENARIO_LABELS[scen], fontsize=9, fontweight="bold")
            ax0.set_ylim(-0.01, orig_rmse * 1.18)
            ax0.spines[["top", "right"]].set_visible(False)

            ax1 = axes[1, c_idx]
            ax1.axhline(0.0, color="0.65", ls=":", lw=0.8)
            ax1.plot(x_pos, viol_vals, "-o", color="#0072B2", lw=1.4, ms=3.5)
            ax1.set_xticks(tick_idx, tick_lbl)
            ax1.set_xlabel(r"Rank penalty $\lambda$")
            ax1.set_ylim(-0.04, 0.72)
            ax1.spines[["top", "right"]].set_visible(False)

        axes[0, 0].set_ylabel("RMSE to true posterior")
        axes[1, 0].set_ylabel("Max rank violation")
        fig.savefig(
            GEN_DIR / "fig_synthetic_pareto.pdf",
            metadata={"CreationDate": None, "ModDate": None},
        )
        plt.close(fig)

        # Figure 2: Real-data paired Brier improvement over IPF (single monochrome dot style)
        fig, axes = plt.subplots(
            2, 4, figsize=(7.2, 4.4), sharey=True, sharex="row", layout="constrained"
        )
        comp_methods = ["euc_strict", "kl_weak", "kl_soft", "temp_ipf", "temp_kl_soft"]
        y_pos = np.arange(len(comp_methods))
        for r_idx, regime in enumerate(["label_shift", "covariate_shift"]):
            sub_r = df_real[df_real["regime"] == regime]
            for m_idx, model in enumerate(MODEL_ORDER):
                ax = axes[r_idx, m_idx]
                sub_m = sub_r[sub_r["model"] == model]
                ipf_m = sub_m[sub_m["method"] == "ipf"].set_index(["dataset", "seed"])[
                    "brier"
                ]
                for k, meth in enumerate(comp_methods):
                    mm = sub_m[
                        (sub_m["method"] == meth) & sub_m["converged"]
                    ].set_index(["dataset", "seed"])["brier"]
                    common = ipf_m.index.intersection(mm.index)
                    if len(common) == 0:
                        continue
                    diffs = (ipf_m.loc[common] - mm.loc[common]).to_numpy() * 1000.0
                    mean, lo, hi = paired_bootstrap_ci(diffs)
                    ax.plot([lo, hi], [k, k], color="#555555", lw=1.2)
                    ax.plot(mean, k, "o", color="#1F2937", ms=3.8)
                ax.axvline(0.0, color="0.45", ls=":", lw=0.8)
                ax.spines[["top", "right"]].set_visible(False)
                if r_idx == 0:
                    ax.set_title(model, fontweight="bold")
                if m_idx == 0:
                    ax.set_yticks(y_pos, [METHOD_LABELS_PLAIN[m] for m in comp_methods])
                    ax.set_ylabel(REGIME_LABELS[regime], fontsize=8.5)
        axes[0, 0].invert_yaxis()
        fig.supxlabel(
            r"Paired Brier improvement over IPF ($\times 1{,}000$; right favors named method)",
            fontsize=8.5,
        )
        fig.savefig(
            GEN_DIR / "fig_real_brier_delta.pdf",
            metadata={"CreationDate": None, "ModDate": None},
        )
        plt.close(fig)

        # Figure 3: Real-data lambda-Pareto frontier (US Census + CA Housing label vs cov shift)
        fig, axes = plt.subplots(
            1, 3, figsize=(7.2, 2.8), sharex=True, layout="constrained"
        )
        panels = [
            ("adult_census", "covariate_shift", "US Census CPS (Male → Female)"),
            ("cal_housing", "covariate_shift", "CA Housing (South → North)"),
            ("cal_housing", "label_shift", "CA Housing (Pure label shift)"),
        ]
        for ax, (ds, reg, title) in zip(axes, panels, strict=True):
            sub = df_real_par[
                (df_real_par["dataset"] == ds) & (df_real_par["regime"] == reg)
            ]
            for model in MODEL_ORDER:
                sm = sub[sub["model"] == model]
                agg = (
                    sm.groupby("lam", as_index=False)[["rank_rate", "brier"]]
                    .mean()
                    .sort_values("lam")
                )
                ax.plot(
                    agg["rank_rate"],
                    agg["brier"],
                    "-",
                    color=MODEL_COLORS[model],
                    lw=1.3,
                    label=model,
                )
                # Circle at IPF (lam=0), filled square at strict KL (lam=inf)
                p0 = agg[agg["lam"] == 0.0].iloc[0]
                pinf = agg[np.isinf(agg["lam"])].iloc[0]
                ax.scatter(
                    [p0["rank_rate"]],
                    [p0["brier"]],
                    facecolors="white",
                    edgecolors=MODEL_COLORS[model],
                    s=32,
                    zorder=4,
                )
                ax.scatter(
                    [pinf["rank_rate"]],
                    [pinf["brier"]],
                    color=MODEL_COLORS[model],
                    marker="s",
                    s=30,
                    zorder=5,
                )
            ax.set_title(title, fontsize=8.5)
            ax.set_xlabel("Weak-order rank rate")
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Held-out Brier score (lower is better)")
        axes[0].set_xlim(0.68, 1.015)
        fig.legend(
            *axes[0].get_legend_handles_labels(),
            loc="outside lower center",
            ncol=4,
            frameon=False,
            fontsize=8,
        )
        fig.savefig(
            GEN_DIR / "fig_real_pareto.pdf",
            metadata={"CreationDate": None, "ModDate": None},
        )
        plt.close(fig)


def write_macros(df_syn: pd.DataFrame, df_real: pd.DataFrame) -> None:
    """Emit every empirical statistic referenced in prose as a LaTeX macro."""
    macros: dict[str, str] = {}

    # Counts
    macros["NumDatasets"] = str(df_real["dataset"].nunique())
    macros["NumModels"] = str(df_real["model"].nunique())
    macros["NumRealSeeds"] = str(df_real["seed"].nunique())
    macros["NumSynthSeeds"] = str(df_syn["seed"].nunique())
    macros["NumTotalRealRuns"] = f"{len(df_real):,}"
    macros["SoftLam"] = "2"

    # Synthetic headline numbers
    for scen, tag in (
        ("pure_label_shift", "SynPure"),
        ("overconfident_label_shift", "SynOver"),
        ("covariate_shift", "SynCov"),
    ):
        s = df_syn[df_syn["scenario"] == scen]
        for meth, mtag in (
            ("orig", "Orig"),
            ("ipf", "Ipf"),
            ("euc_strict", "Euc"),
            ("kl_weak", "KlWeak"),
            ("kl_soft", "KlSoft"),
        ):
            ok = s[(s["method"] == meth) & s["converged"]]
            macros[f"{tag}{mtag}Rmse"] = (
                f"{ok['rmse_true'].mean():.4f}" if len(ok) else "--"
            )
            macros[f"{tag}{mtag}Rank"] = (
                f"{ok['rank_rate'].mean() * 100:.1f}" if len(ok) else "--"
            )
            macros[f"{tag}{mtag}Brier"] = (
                f"{ok['brier'].mean():.4f}" if len(ok) else "--"
            )

    # Tie-handling improvement in synthetic quantized test
    tie_rows = df_syn[df_syn["method"] == "kl_weak"].dropna(
        subset=["kl_weak_tie_val", "kl_stable_tie_val"]
    )
    if len(tie_rows):
        rel_drop = (
            tie_rows["kl_stable_tie_val"] - tie_rows["kl_weak_tie_val"]
        ) / np.maximum(tie_rows["kl_stable_tie_val"], 1e-12)
        macros["TieKlDropPct"] = f"{rel_drop.mean() * 100:.1f}"
        macros["TieStableConvPct"] = (
            f"{df_syn[df_syn['method'] == 'kl_weak']['kl_stable_conv'].mean() * 100:.0f}"
        )
    else:
        macros["TieKlDropPct"] = "0.0"
        macros["TieStableConvPct"] = "0"

    # Real benchmark convergence rates
    for meth, mtag in (
        ("euc_strict", "Euc"),
        ("kl_weak", "KlWeak"),
        ("kl_soft", "KlSoft"),
        ("temp_kl_soft", "TempKlSoft"),
    ):
        sub = df_real[df_real["method"] == meth]
        macros[f"Real{mtag}ConvPct"] = f"{sub['converged'].mean() * 100:.1f}"
        # Digits specifically
        sub_d = sub[sub["dataset"] == "digits"]
        macros[f"Digits{mtag}ConvPct"] = f"{sub_d['converged'].mean() * 100:.0f}"

    # Regime-level means and matched-pair deltas vs IPF
    for regime, rtag in (("label_shift", "Lab"), ("covariate_shift", "Cov")):
        sr = df_real[df_real["regime"] == regime]
        ipf_ok = sr[(sr["method"] == "ipf") & sr["converged"]].set_index(
            ["dataset", "model", "seed"]
        )
        for meth, mtag in (
            ("orig", "Orig"),
            ("ipf", "Ipf"),
            ("temp_ipf", "TempIpf"),
            ("euc_strict", "Euc"),
            ("kl_weak", "KlWeak"),
            ("kl_soft", "KlSoft"),
            ("temp_kl_soft", "TempKlSoft"),
        ):
            sub_m = sr[sr["method"] == meth]
            macros[f"{rtag}{mtag}Conv"] = f"{sub_m['converged'].mean() * 100:.1f}"
            ok = sub_m[sub_m["converged"]].set_index(["dataset", "model", "seed"])
            macros[f"{rtag}{mtag}Brier"] = (
                f"{ok['brier'].mean():.4f}" if len(ok) else "--"
            )
            macros[f"{rtag}{mtag}Nll"] = f"{ok['nll'].mean():.4f}" if len(ok) else "--"
            macros[f"{rtag}{mtag}Rank"] = (
                f"{ok['rank_rate'].mean() * 100:.1f}" if len(ok) else "--"
            )
            macros[f"{rtag}{mtag}MaxViol"] = (
                f"{ok['max_viol'].mean():.3f}" if len(ok) else "--"
            )
            macros[f"{rtag}{mtag}Auroc"] = (
                f"{ok['auroc'].mean():.4f}" if len(ok) else "--"
            )
            common = ipf_ok.index.intersection(ok.index)
            if len(common):
                db, dblo, dbhi = paired_bootstrap_ci(
                    (ipf_ok.loc[common, "brier"] - ok.loc[common, "brier"]).to_numpy()
                    * 1000.0
                )
                dn, dnlo, dnhi = paired_bootstrap_ci(
                    (ipf_ok.loc[common, "nll"] - ok.loc[common, "nll"]).to_numpy()
                    * 1000.0
                )
                dr = float(
                    (
                        ok.loc[common, "rank_rate"] - ipf_ok.loc[common, "rank_rate"]
                    ).mean()
                    * 100.0
                )
                macros[f"{rtag}{mtag}DBrier"] = f"{db:+.1f}"
                macros[f"{rtag}{mtag}DBrierLo"] = f"{dblo:+.1f}"
                macros[f"{rtag}{mtag}DBrierHi"] = f"{dbhi:+.1f}"
                macros[f"{rtag}{mtag}DNll"] = f"{dn:+.1f}"
                macros[f"{rtag}{mtag}DRank"] = f"{dr:+.2f}"

    # US Census CPS (Male -> Female) headline numbers (overall + MLP + LogReg)
    cen = df_real[
        (df_real["dataset"] == "adult_census")
        & (df_real["regime"] == "covariate_shift")
    ]
    for meth, mtag in (
        ("orig", "Orig"),
        ("ipf", "Ipf"),
        ("temp_ipf", "TempIpf"),
        ("kl_weak", "KlWeak"),
        ("kl_soft", "KlSoft"),
        ("temp_kl_soft", "TempKlSoft"),
    ):
        ok = cen[(cen["method"] == meth) & cen["converged"]]
        macros[f"Cen{mtag}Brier"] = f"{ok['brier'].mean():.4f}" if len(ok) else "--"
        macros[f"Cen{mtag}Nll"] = f"{ok['nll'].mean():.4f}" if len(ok) else "--"
        macros[f"Cen{mtag}Rank"] = (
            f"{ok['rank_rate'].mean() * 100:.1f}" if len(ok) else "--"
        )
        macros[f"Cen{mtag}MaxViol"] = (
            f"{ok['max_viol'].mean():.3f}" if len(ok) else "--"
        )
        for mod, modtag in (("MLP", "Mlp"), ("HGB", "Hgb"), ("LogReg", "Lr")):
            okm = ok[ok["model"] == mod]
            macros[f"Cen{modtag}{mtag}Brier"] = (
                f"{okm['brier'].mean():.4f}" if len(okm) else "--"
            )
            macros[f"Cen{modtag}{mtag}Nll"] = (
                f"{okm['nll'].mean():.4f}" if len(okm) else "--"
            )
            macros[f"Cen{modtag}{mtag}Rank"] = (
                f"{okm['rank_rate'].mean() * 100:.1f}" if len(okm) else "--"
            )

    # Overconfident models (HGB + MLP) under covariate shift: KL soft / Temp+KL soft vs IPF / Temp+IPF
    oc = df_real[
        (df_real["regime"] == "covariate_shift")
        & (df_real["model"].isin(["HGB", "MLP"]))
    ]
    for meth, mtag in (
        ("ipf", "Ipf"),
        ("temp_ipf", "TempIpf"),
        ("kl_weak", "KlWeak"),
        ("kl_soft", "KlSoft"),
        ("temp_kl_soft", "TempKlSoft"),
    ):
        ok = oc[(oc["method"] == meth) & oc["converged"]]
        macros[f"CovOver{mtag}Brier"] = f"{ok['brier'].mean():.4f}" if len(ok) else "--"
        macros[f"CovOver{mtag}Nll"] = f"{ok['nll'].mean():.4f}" if len(ok) else "--"
        macros[f"CovOver{mtag}Rank"] = (
            f"{ok['rank_rate'].mean() * 100:.1f}" if len(ok) else "--"
        )

    (GEN_DIR / "numbers.tex").write_text(
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items()))
    )


def main() -> int:
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    df_syn = pd.read_csv(OUT_DIR / "synthetic_results.csv")
    df_syn_par = pd.read_csv(OUT_DIR / "synthetic_pareto.csv")
    df_real = pd.read_csv(OUT_DIR / "real_benchmark.csv")
    df_real_par = pd.read_csv(OUT_DIR / "real_pareto.csv")

    build_synthetic_table(df_syn)
    build_real_summary_table(df_real)
    build_real_by_dataset_table(df_real)
    build_census_table(df_real)
    build_figures(df_syn, df_syn_par, df_real, df_real_par)
    write_macros(df_syn, df_real)
    print(f"Wrote tables, figures, and macros to {GEN_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
