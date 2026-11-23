#!/usr/bin/env python3
"""Comprehensive, reproducible benchmark for rank-preserving multiclass calibration.

Runs three experiments and writes raw CSVs under ``ms/experiments/output/``:
1. ``synthetic_results.csv`` & ``synthetic_pareto.csv``: 3-class Gaussian mixture
   with known exact Bayes posteriors under pure label shift (calibrated vs.
   overconfident source model) and covariate-mean shift, plus quantized-tie
   comparison of ``ties="weak"`` vs. ``ties="stable"``.
2. ``real_benchmark.csv``: 6 real multiclass datasets (J = 3..10), 4 model
   families (LogReg, RF, HGB, MLP), 2 shift regimes (pure label shift vs.
   covariate / subpopulation shift), 10 seeds each, comparing 7 methods:
   orig, ipf, temp_ipf, euc_strict, kl_weak, kl_soft, temp_kl_soft.
3. ``real_pareto.csv``: Full lambda-Pareto frontier on US Census (adult_census,
   Male -> Female subpopulation shift) and California Housing (label shift vs.
   covariate shift).
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import multivariate_normal
from sklearn.datasets import (
    fetch_20newsgroups,
    fetch_california_housing,
    fetch_covtype,
    fetch_openml,
    load_digits,
    load_wine,
)
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rank_preserving_calibration import (  # noqa: E402
    CalibrationError,
    calibrate_dykstra,
    calibrate_ipf,
    calibrate_kl,
    calibrate_kl_soft,
    kl_divergence,
    top_label_ece,
)
from rank_preserving_calibration.kl_calibration import (  # noqa: E402
    _compute_weak_rank_violation,
)

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parent / "output"
N_SEEDS_SYNTH = 20
N_SEEDS_REAL = 10
N_SEEDS_PARETO = 5
SOFT_LAM = 2.0


def weak_order_metrics(
    Q: np.ndarray, A: np.ndarray, tol: float = 1e-5
) -> tuple[float, float]:
    """Return (weak-order pairwise preservation rate, max strict weak-order violation)."""
    max_viol = _compute_weak_rank_violation(Q, A)
    tot = 0
    ok = 0
    for j in range(Q.shape[1]):
        idx = np.argsort(A[:, j], kind="mergesort")
        a_s = A[idx, j]
        q_s = Q[idx, j]
        diff_a = a_s[:, None] < a_s[None, :]
        ok_q = q_s[:, None] <= q_s[None, :] + tol
        tot += int(diff_a.sum())
        ok += int((diff_a & ok_q).sum())
    return (ok / tot if tot else 1.0), max_viol


def safe_probs(Q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    Qc = np.clip(np.asarray(Q, dtype=float), eps, 1.0)
    Qc /= Qc.sum(axis=1, keepdims=True)
    return Qc


def macro_ovr_auroc(y_true: np.ndarray, Q: np.ndarray, J: int) -> float:
    """Macro one-vs-rest AUROC across classes present in y_true."""
    Qc = safe_probs(Q)
    scores = []
    for j in range(J):
        y_bin = (y_true == j).astype(int)
        if 0 < y_bin.sum() < len(y_bin):
            scores.append(float(roc_auc_score(y_bin, Qc[:, j])))
    return float(np.mean(scores)) if scores else float("nan")


def fit_temperature(P_val: np.ndarray, y_val: np.ndarray, J: int) -> float:
    logits = np.log(np.clip(P_val, 1e-12, 1.0))

    def obj(tau: float) -> float:
        z = logits / tau
        z -= z.max(axis=1, keepdims=True)
        q = np.exp(z)
        q /= q.sum(axis=1, keepdims=True)
        return float(log_loss(y_val, q, labels=np.arange(J)))

    res = minimize_scalar(obj, bounds=(0.15, 10.0), method="bounded")
    return float(res.x)


def apply_temperature(P: np.ndarray, tau: float) -> np.ndarray:
    z = np.log(np.clip(P, 1e-12, 1.0)) / tau
    z -= z.max(axis=1, keepdims=True)
    q = np.exp(z)
    q /= q.sum(axis=1, keepdims=True)
    return q


def evaluate_matrix(
    Q: np.ndarray,
    P_anchor: np.ndarray,
    y_true: np.ndarray,
    T: np.ndarray,
    J: int,
) -> dict[str, float]:
    Qc = safe_probs(Q)
    rr, mv = weak_order_metrics(Q, P_anchor)
    return {
        "brier": float(
            brier_score_loss(y_true, Qc, labels=np.arange(J), scale_by_half=False)
        ),
        "nll": float(log_loss(y_true, Qc, labels=np.arange(J))),
        "ece": float(top_label_ece(y_true, Qc)["ece"]),
        "auroc": macro_ovr_auroc(y_true, Qc, J),
        "acc": float(np.mean(Qc.argmax(axis=1) == y_true)),
        "rank_rate": rr,
        "max_viol": mv,
        "col_err": float(np.max(np.abs(Q.sum(axis=0) - T))),
        "row_err": float(np.max(np.abs(Q.sum(axis=1) - 1.0))),
    }


def make_models(seed: int) -> dict[str, object]:
    return {
        "LogReg": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)
        ),
        "RF": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        "HGB": HistGradientBoostingClassifier(random_state=seed),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed),
        ),
    }


# ---------------------------------------------------------------------
# Experiment 1: Exact-posterior synthetic benchmark
# ---------------------------------------------------------------------


def _gmm_posterior(
    X: np.ndarray, means: np.ndarray, cov: np.ndarray, priors: np.ndarray
) -> np.ndarray:
    J = len(priors)
    dens = np.column_stack(
        [
            multivariate_normal.pdf(X, mean=means[j], cov=cov) * priors[j]
            for j in range(J)
        ]
    )
    dens = np.clip(dens, 1e-300, None)
    return dens / dens.sum(axis=1, keepdims=True)


def run_synthetic() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("=== Experiment 1: Exact-posterior synthetic benchmark ===")
    means_src = np.array([[0.0, 0.0], [1.6, 0.4], [0.5, 1.7]])
    cov = np.eye(2) * 1.1
    priors_src = np.array([0.45, 0.35, 0.20])
    priors_tgt = np.array([0.15, 0.60, 0.25])
    J = 3
    N = 250

    rows = []
    pareto_rows = []
    lam_grid = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0]

    for scenario in (
        "pure_label_shift",
        "overconfident_label_shift",
        "covariate_shift",
    ):
        for seed in range(N_SEEDS_SYNTH):
            rng = np.random.default_rng(100 + seed)
            y = rng.choice(J, size=N, p=priors_tgt)
            # Guarantee every class appears
            for j in range(J):
                if (y == j).sum() == 0:
                    y[j] = j
            if scenario == "covariate_shift":
                means_tgt = means_src + np.array(
                    [[0.0, 0.0], [-0.45, 0.35], [0.35, -0.4]]
                )
            else:
                means_tgt = means_src
            X = np.vstack([rng.multivariate_normal(means_tgt[c], cov) for c in y])

            P_true = _gmm_posterior(X, means_tgt, cov, priors_tgt)
            P_src = _gmm_posterior(X, means_src, cov, priors_src)
            if scenario == "overconfident_label_shift":
                P_in = safe_probs(P_src**2.5, eps=1e-6)
            else:
                P_in = safe_probs(P_src, eps=1e-6)
            T = P_true.sum(axis=0)

            Q_ipf = calibrate_ipf(P_in, T, max_iters=5000, tol=1e-10).Q

            try:
                t0 = time.time()
                r_euc = calibrate_dykstra(P_in, T, max_iters=1500, tol=1e-6)
                Q_euc, conv_euc, dt_euc = r_euc.Q, r_euc.converged, time.time() - t0
            except CalibrationError:
                Q_euc, conv_euc, dt_euc = None, False, float("nan")

            t0 = time.time()
            r_kl = calibrate_kl(P_in, T, tol=1e-8, ties="weak")
            dt_kl = time.time() - t0

            t0 = time.time()
            r_kls = calibrate_kl_soft(P_in, T, lam=SOFT_LAM, tol=1e-8, ties="weak")
            dt_kls = time.time() - t0

            # Also test quantized ties (rounded to 0.05) comparing weak vs stable
            P_q = np.round(P_in * 20.0) / 20.0
            P_q = safe_probs(P_q, eps=1e-6)
            r_q_weak = calibrate_kl(P_q, T, tol=1e-8, ties="weak")
            try:
                r_q_stable = calibrate_kl(P_q, T, tol=1e-8, ties="stable")
                kl_stable_val = r_q_stable.kl_divergence
                conv_stable = True
            except CalibrationError:
                kl_stable_val = float("nan")
                conv_stable = False

            for name, Q, conv, dt in (
                ("orig", P_in, True, 0.0),
                ("ipf", Q_ipf, True, 0.0),
                ("euc_strict", Q_euc, conv_euc, dt_euc),
                ("kl_weak", r_kl.Q, r_kl.converged, dt_kl),
                ("kl_soft", r_kls.Q, r_kls.converged, dt_kls),
            ):
                if Q is None:
                    rows.append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "method": name,
                            "converged": False,
                            "seconds": dt,
                            "rmse_true": float("nan"),
                            "kl_from_in": float("nan"),
                            "kl_weak_tie_val": r_q_weak.kl_divergence,
                            "kl_stable_tie_val": kl_stable_val,
                            "kl_stable_conv": conv_stable,
                            **{
                                k: float("nan")
                                for k in (
                                    "brier",
                                    "nll",
                                    "ece",
                                    "auroc",
                                    "acc",
                                    "rank_rate",
                                    "max_viol",
                                    "col_err",
                                    "row_err",
                                )
                            },
                        }
                    )
                    continue
                m = evaluate_matrix(Q, P_in, y, T, J)
                rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "method": name,
                        "converged": conv,
                        "seconds": dt,
                        "rmse_true": float(
                            np.sqrt(np.mean((safe_probs(Q) - P_true) ** 2))
                        ),
                        "kl_from_in": float(kl_divergence(safe_probs(Q), P_in)),
                        "kl_weak_tie_val": r_q_weak.kl_divergence,
                        "kl_stable_tie_val": kl_stable_val,
                        "kl_stable_conv": conv_stable,
                        **m,
                    }
                )

            if seed < 10:
                for lam in lam_grid:
                    r = (
                        calibrate_ipf(P_in, T, max_iters=5000, tol=1e-10)
                        if lam == 0.0
                        else calibrate_kl_soft(P_in, T, lam=lam, tol=1e-8, ties="weak")
                    )
                    m = evaluate_matrix(r.Q, P_in, y, T, J)
                    pareto_rows.append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "lam": lam,
                            "rmse_true": float(
                                np.sqrt(np.mean((safe_probs(r.Q) - P_true) ** 2))
                            ),
                            "kl_from_in": float(kl_divergence(safe_probs(r.Q), P_in)),
                            **m,
                        }
                    )
                # Append strict endpoint as lam = inf
                m = evaluate_matrix(r_kl.Q, P_in, y, T, J)
                pareto_rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "lam": float("inf"),
                        "rmse_true": float(
                            np.sqrt(np.mean((safe_probs(r_kl.Q) - P_true) ** 2))
                        ),
                        "kl_from_in": float(kl_divergence(safe_probs(r_kl.Q), P_in)),
                        **m,
                    }
                )

    df_syn = pd.DataFrame(rows)
    df_par = pd.DataFrame(pareto_rows)
    df_syn.to_csv(OUT_DIR / "synthetic_results.csv", index=False)
    df_par.to_csv(OUT_DIR / "synthetic_pareto.csv", index=False)
    print(
        df_syn.groupby(["scenario", "method"])[
            ["rmse_true", "brier", "nll", "rank_rate", "max_viol", "col_err"]
        ]
        .mean()
        .round(4)
    )
    return df_syn, df_par


# ---------------------------------------------------------------------
# Real datasets loader
# ---------------------------------------------------------------------


def load_real_datasets() -> dict[str, dict]:
    """Load 6 diverse real multiclass datasets with natural covariate splits."""
    datasets: dict[str, dict] = {}

    # 1. wine (J=3)
    X, y = load_wine(return_X_y=True)
    datasets["wine"] = {"X": X, "y": y, "J": 3, "cov_split": None, "n_eval": 71}

    # 2. cal_housing (J=4, price quartiles; natural north/south latitude shift!)
    X, y_cont = fetch_california_housing(return_X_y=True)
    rng = np.random.default_rng(42)
    sub = rng.choice(len(y_cont), size=4000, replace=False)
    X_c = X[sub]
    qs = np.quantile(y_cont[sub], [0.25, 0.5, 0.75])
    y_c = np.digitize(y_cont[sub], qs)
    # Column 6 is Latitude: train South California (Lat <= median), test North
    lat_med = float(np.median(X_c[:, 6]))
    datasets["cal_housing"] = {
        "X": X_c,
        "y": y_c,
        "J": 4,
        "cov_split": X_c[:, 6] <= lat_med,
        "cov_desc": "South -> North CA",
        "n_eval": 250,
    }

    # 3. adult_census (J=4, US Census CPS marital status; natural Male -> Female shift!)
    df = fetch_openml(data_id=1590, as_frame=True, parser="auto").frame.dropna()
    rng_a = np.random.default_rng(42)
    df = df.sample(n=6000, random_state=42).reset_index(drop=True)
    marital_map = {
        "Married-civ-spouse": 0,
        "Married-AF-spouse": 0,
        "Never-married": 1,
        "Divorced": 2,
        "Separated": 3,
        "Widowed": 3,
        "Married-spouse-absent": 3,
    }
    y_a = df["marital-status"].astype(str).map(marital_map).to_numpy(dtype=int)
    is_male = (df["sex"].astype(str) == "Male").to_numpy()
    num_cols = [
        "age",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    cat_cols = ["workclass", "occupation", "race", "class"]
    X_num = df[num_cols].to_numpy(dtype=float)
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = enc.fit_transform(df[cat_cols].astype(str))
    X_a = np.hstack([X_num, X_cat])
    datasets["adult_census"] = {
        "X": X_a,
        "y": y_a,
        "J": 4,
        "cov_split": is_male,
        "cov_desc": "Male -> Female (CPS)",
        "n_eval": 250,
    }

    # 4. 20news_pol (J=4, political/religion text; TF-IDF + SVD(100))
    ng = fetch_20newsgroups(
        subset="all",
        categories=[
            "talk.politics.guns",
            "talk.politics.mideast",
            "talk.politics.misc",
            "soc.religion.christian",
        ],
        remove=("headers", "footers", "quotes"),
    )
    vec = TfidfVectorizer(max_features=8000, stop_words="english", min_df=3)
    X_tfidf = vec.fit_transform(ng.data)
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_ng = svd.fit_transform(X_tfidf)
    y_ng = np.asarray(ng.target, dtype=int)
    # Split by document length median (short posts -> long posts)
    doc_lens = np.array([len(t) for t in ng.data])
    len_med = float(np.median(doc_lens))
    datasets["20news_pol"] = {
        "X": X_ng,
        "y": y_ng,
        "J": 4,
        "cov_split": doc_lens <= len_med,
        "cov_desc": "Short -> Long posts",
        "n_eval": 250,
    }

    # 5. covtype (J=7, forest cover types; natural Elevation split!)
    X_cov, y_cov = fetch_covtype(return_X_y=True)
    y_cov = y_cov - 1
    # Balanced stratified-ish sample ensuring rare classes have enough rows
    rng_cv = np.random.default_rng(42)
    idxs = []
    for j in range(7):
        j_idx = np.where(y_cov == j)[0]
        take = min(800, len(j_idx))
        idxs.append(rng_cv.choice(j_idx, size=take, replace=False))
    sel = np.concatenate(idxs)
    X_cv, y_cv = X_cov[sel], y_cov[sel]
    # Split by Slope/Aspect/Horizontal_Distance_To_Roadways (col 5) so all 7 classes appear on both sides
    col5_med = float(np.median(X_cv[:, 5]))
    cov_mask = X_cv[:, 5] <= col5_med
    if len(np.unique(y_cv[cov_mask])) < 7 or len(np.unique(y_cv[~cov_mask])) < 7:
        cov_mask = None
    datasets["covtype"] = {
        "X": X_cv,
        "y": y_cv,
        "J": 7,
        "cov_split": cov_mask,
        "cov_desc": "Near -> Far road",
        "n_eval": 220,
    }

    # 6. digits (J=10, handwritten digits; PC1 split)
    X_d, y_d = load_digits(return_X_y=True)
    pc1 = np.linalg.svd(StandardScaler().fit_transform(X_d), full_matrices=False)[0][
        :, 0
    ]
    datasets["digits"] = {
        "X": X_d,
        "y": y_d,
        "J": 10,
        "cov_split": pc1 <= float(np.median(pc1)),
        "cov_desc": "PC1 <= 0 -> PC1 > 0",
        "n_eval": 200,
    }

    return datasets


# ---------------------------------------------------------------------
# Experiment 2: Broad multi-dataset, multi-model, multi-regime benchmark
# ---------------------------------------------------------------------


def run_real_benchmark(datasets: dict[str, dict]) -> pd.DataFrame:
    print("\n=== Experiment 2: Broad real-data benchmark ===")
    all_rows = []
    out_path = OUT_DIR / "real_benchmark.csv"

    for dname, info in datasets.items():
        X, y, J, n_eval = info["X"], info["y"], info["J"], info["n_eval"]
        regimes = ["label_shift"]
        if info["cov_split"] is not None:
            regimes.append("covariate_shift")

        for regime in regimes:
            t_reg = time.time()
            for seed in range(N_SEEDS_REAL):
                rng = np.random.default_rng(1000 + seed * 17)
                if regime == "label_shift":
                    X_tr_all, X_te, y_tr_all, y_te = train_test_split(
                        X, y, test_size=0.4, random_state=seed, stratify=y
                    )
                    # Draw Dirichlet-shifted target class counts
                    w = rng.dirichlet(np.ones(J) * 0.7)
                    counts = np.maximum((w * n_eval).astype(int), 2)
                    Xs, ys = [], []
                    for j in range(J):
                        idx = np.where(y_te == j)[0]
                        s = rng.choice(idx, size=counts[j], replace=True)
                        Xs.append(X_te[s])
                        ys.append(y_te[s])
                    X_eval, y_eval = np.vstack(Xs), np.concatenate(ys)
                else:
                    mask = info["cov_split"]
                    X_tr_all, y_tr_all = X[mask], y[mask]
                    X_te, y_te = X[~mask], y[~mask]
                    # Subsample evaluation batch of size n_eval (ensuring all classes present)
                    if len(y_te) > n_eval:
                        sub = rng.choice(len(y_te), size=n_eval, replace=False)
                        X_eval, y_eval = X_te[sub], y_te[sub]
                    else:
                        X_eval, y_eval = X_te, y_te

                # Split source training data into model-fit (75%) and temp-val (25%)
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_tr_all,
                    y_tr_all,
                    test_size=0.25,
                    random_state=seed,
                    stratify=y_tr_all,
                )
                T = np.bincount(y_eval, minlength=J).astype(float)
                # Ensure strictly positive targets for IPF/KL stability when a rare class is absent in a subsample
                if np.any(T == 0):
                    continue

                for mname, clf in make_models(seed).items():
                    clf.fit(X_tr, y_tr)
                    P_val = safe_probs(clf.predict_proba(X_val), eps=1e-6)
                    tau = fit_temperature(P_val, y_val, J)
                    P = safe_probs(clf.predict_proba(X_eval), eps=1e-6)
                    P_temp = apply_temperature(P, tau)

                    # 1. orig
                    methods_out = [("orig", P, True, 0.0)]

                    # 2. ipf
                    t0 = time.time()
                    r_ipf = calibrate_ipf(P, T, max_iters=5000, tol=1e-10)
                    methods_out.append(
                        ("ipf", r_ipf.Q, r_ipf.converged, time.time() - t0)
                    )

                    # 3. temp_ipf
                    t0 = time.time()
                    r_tipf = calibrate_ipf(P_temp, T, max_iters=5000, tol=1e-10)
                    methods_out.append(
                        ("temp_ipf", r_tipf.Q, r_tipf.converged, time.time() - t0)
                    )

                    # 4. euc_strict (Dykstra)
                    t0 = time.time()
                    try:
                        r_euc = calibrate_dykstra(P, T, max_iters=1500, tol=1e-6)
                        methods_out.append(
                            ("euc_strict", r_euc.Q, r_euc.converged, time.time() - t0)
                        )
                    except CalibrationError:
                        methods_out.append(
                            ("euc_strict", None, False, time.time() - t0)
                        )

                    # 5. kl_weak (exact weak-order KL)
                    t0 = time.time()
                    try:
                        r_kl = calibrate_kl(P, T, tol=1e-8, ties="weak")
                        methods_out.append(
                            ("kl_weak", r_kl.Q, r_kl.converged, time.time() - t0)
                        )
                    except CalibrationError:
                        methods_out.append(("kl_weak", None, False, time.time() - t0))

                    # 6. kl_soft (lam = 2.0)
                    t0 = time.time()
                    try:
                        r_kls = calibrate_kl_soft(
                            P, T, lam=SOFT_LAM, tol=1e-8, ties="weak"
                        )
                        methods_out.append(
                            ("kl_soft", r_kls.Q, r_kls.converged, time.time() - t0)
                        )
                    except CalibrationError:
                        methods_out.append(("kl_soft", None, False, time.time() - t0))

                    # 7. temp_kl_soft (anchor-reference decoupled: R=P_temp, A=P)
                    t0 = time.time()
                    try:
                        r_tkls = calibrate_kl_soft(
                            P, T, lam=SOFT_LAM, R=P_temp, A=P, tol=1e-8, ties="weak"
                        )
                        methods_out.append(
                            (
                                "temp_kl_soft",
                                r_tkls.Q,
                                r_tkls.converged,
                                time.time() - t0,
                            )
                        )
                    except CalibrationError:
                        methods_out.append(
                            ("temp_kl_soft", None, False, time.time() - t0)
                        )

                    for meth, Q, conv, dt in methods_out:
                        if Q is None:
                            all_rows.append(
                                {
                                    "dataset": dname,
                                    "J": J,
                                    "regime": regime,
                                    "model": mname,
                                    "seed": seed,
                                    "N": len(y_eval),
                                    "method": meth,
                                    "converged": False,
                                    "seconds": dt,
                                    **{
                                        k: float("nan")
                                        for k in (
                                            "brier",
                                            "nll",
                                            "ece",
                                            "auroc",
                                            "acc",
                                            "rank_rate",
                                            "max_viol",
                                            "col_err",
                                            "row_err",
                                        )
                                    },
                                }
                            )
                        else:
                            m = evaluate_matrix(Q, P, y_eval, T, J)
                            all_rows.append(
                                {
                                    "dataset": dname,
                                    "J": J,
                                    "regime": regime,
                                    "model": mname,
                                    "seed": seed,
                                    "N": len(y_eval),
                                    "method": meth,
                                    "converged": conv,
                                    "seconds": dt,
                                    **m,
                                }
                            )

            # Checkpoint after every (dataset, regime)
            pd.DataFrame(all_rows).to_csv(out_path, index=False)
            print(
                f"  finished {dname:13s} | {regime:15s} in {time.time() - t_reg:5.1f}s "
                f"({len(all_rows)} rows total)",
                flush=True,
            )

    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False)
    return df


# ---------------------------------------------------------------------
# Experiment 3: Real-data lambda-Pareto frontier
# ---------------------------------------------------------------------


def run_real_pareto(datasets: dict[str, dict]) -> pd.DataFrame:
    print("\n=== Experiment 3: Real-data lambda-Pareto frontier ===")
    lam_grid = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, float("inf")]
    rows = []

    configs = [
        ("adult_census", "covariate_shift"),
        ("cal_housing", "label_shift"),
        ("cal_housing", "covariate_shift"),
    ]
    for dname, regime in configs:
        info = datasets[dname]
        X, y, J, n_eval = info["X"], info["y"], info["J"], info["n_eval"]
        for seed in range(N_SEEDS_PARETO):
            rng = np.random.default_rng(2000 + seed * 13)
            if regime == "label_shift":
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.4, random_state=seed, stratify=y
                )
                w = rng.dirichlet(np.ones(J) * 0.7)
                counts = np.maximum((w * n_eval).astype(int), 2)
                Xs, ys = [], []
                for j in range(J):
                    idx = np.where(y_te == j)[0]
                    s = rng.choice(idx, size=counts[j], replace=True)
                    Xs.append(X_te[s])
                    ys.append(y_te[s])
                X_eval, y_eval = np.vstack(Xs), np.concatenate(ys)
            else:
                mask = info["cov_split"]
                X_tr, y_tr = X[mask], y[mask]
                X_te, y_te = X[~mask], y[~mask]
                sub = rng.choice(len(y_te), size=n_eval, replace=False)
                X_eval, y_eval = X_te[sub], y_te[sub]

            T = np.bincount(y_eval, minlength=J).astype(float)
            for mname, clf in make_models(seed).items():
                clf.fit(X_tr, y_tr)
                P = safe_probs(clf.predict_proba(X_eval), eps=1e-6)
                for lam in lam_grid:
                    if lam == 0.0:
                        r = calibrate_ipf(P, T, max_iters=5000, tol=1e-10)
                    elif np.isinf(lam):
                        r = calibrate_kl(P, T, tol=1e-8, ties="weak")
                    else:
                        r = calibrate_kl_soft(P, T, lam=lam, tol=1e-8, ties="weak")
                    m = evaluate_matrix(r.Q, P, y_eval, T, J)
                    rows.append(
                        {
                            "dataset": dname,
                            "regime": regime,
                            "model": mname,
                            "seed": seed,
                            "lam": lam,
                            "kl_from_in": float(kl_divergence(safe_probs(r.Q), P)),
                            **m,
                        }
                    )

    df_p = pd.DataFrame(rows)
    df_p.to_csv(OUT_DIR / "real_pareto.csv", index=False)
    print(f"  wrote real_pareto.csv ({len(df_p)} rows)")
    return df_p


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    run_synthetic()
    datasets = load_real_datasets()
    run_real_benchmark(datasets)
    run_real_pareto(datasets)
    meta = {
        "n_seeds_synth": N_SEEDS_SYNTH,
        "n_seeds_real": N_SEEDS_REAL,
        "n_seeds_pareto": N_SEEDS_PARETO,
        "soft_lam": SOFT_LAM,
        "elapsed_seconds": round(time.time() - t_all, 1),
    }
    (OUT_DIR / "benchmark_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nALL BENCHMARKS COMPLETE in {meta['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
