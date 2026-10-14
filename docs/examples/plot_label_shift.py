"""
Matching known class shares after deployment
============================================

A classifier trained where every digit is equally common is deployed where
0, 1 and 2 are four times as common as the rest, and the new shares are known.
Calibration makes each class's predicted total equal its known share while
keeping the order of rows within each class.

The standard fix for this kind of shift rescales each class probability by
new share / training share and renormalizes each row (Saerens et al., 2002).
It gives the better log loss here, for both an accurate and an unsure model.
It does not force the predicted totals to equal the known ones (the unsure
model's land at 266 against 240), and because each row is renormalized
separately it reverses the order of many rows within a class.
Rank-preserving calibration does both exactly, at some cost in log loss.
Reach for it when the totals must match, such as when probabilities feed
an allocation or quota, and the ranking within each class is what decisions
use.
"""

# %%
# Two models, one deployment
# --------------------------
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from rank_preserving_calibration import calibrate, nll

X, y = load_digits(return_X_y=True)
X_train, X_pool, y_train, y_pool = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=0
)
rng = np.random.default_rng(0)
counts = np.where(np.arange(10) <= 2, 80, 20)
deploy = np.concatenate(
    [
        rng.choice(np.flatnonzero(y_pool == k), n, replace=False)
        for k, n in enumerate(counts)
    ]
)
X_test, y_test = X_pool[deploy], y_pool[deploy]

shares = counts / counts.sum()
M = shares * len(y_test)  # expected count per class; must sum to the row count

models = {
    "accurate (C=1)": LogisticRegression(C=1.0, max_iter=5000),
    "unsure (C=1e-4)": LogisticRegression(C=1e-4, max_iter=5000),
}

# %%
# Log loss, totals and order
# --------------------------
# ``reorders`` counts adjacent pairs, within each class, whose order the
# prior-ratio fix reverses; rank-preserving calibration reverses none.


def reorders(before, after):
    total = 0
    for k in range(before.shape[1]):
        column = after[np.argsort(before[:, k], kind="stable"), k]
        total += int((np.diff(column) < -1e-12).sum())
    return total


print(
    f"{'model':<16} {'raw':>6} {'ratio':>6} {'calib':>6} {'reorders':>9}"
    f"   total for 0-2 (actual {counts[:3].sum()})"
)
calibrated = {}
for name, model in models.items():
    P = model.fit(X_train, y_train).predict_proba(X_test)
    R = P * shares / np.bincount(y_train, minlength=10) * len(y_train)
    R /= R.sum(axis=1, keepdims=True)
    result = calibrate(P, M)
    calibrated[name] = (P, result.Q)
    print(
        f"{name:<16} {nll(y_test, P):6.3f} {nll(y_test, R):6.3f} "
        f"{nll(y_test, result.Q):6.3f} {reorders(P, R):9d}   "
        f"{P[:, :3].sum():.0f} raw, {R[:, :3].sum():.0f} ratio, "
        f"{result.Q[:, :3].sum():.0f} calibrated"
    )

# %%
# Predicted against actual class counts
# -------------------------------------
actual = np.bincount(y_test, minlength=10)
digits = np.arange(10)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
for ax, (name, (P, Q)) in zip(axes, calibrated.items(), strict=True):
    ax.bar(digits - 0.2, P.sum(axis=0), width=0.4, label="model")
    ax.bar(digits + 0.2, Q.sum(axis=0), width=0.4, label="calibrated")
    ax.scatter(digits, actual, color="black", zorder=3, label="actual")
    ax.set_title(name)
    ax.set_xticks(digits)
    ax.set_xlabel("digit")
axes[0].set_ylabel(f"count among {len(y_test)} deployed rows")
axes[0].legend()
fig.tight_layout()
plt.show()
