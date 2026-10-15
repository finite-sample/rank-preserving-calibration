"""Metrics checked against hand-computed values or an independent implementation.

Each expected value comes from somewhere other than the function under test:
arithmetic done in the test, scipy, or the published definition of the metric.
"""

import numpy as np
import pytest
from scipy.special import rel_entr
from scipy.stats import entropy, mannwhitneyu

from rank_preserving_calibration import (
    auc_deltas,
    brier,
    classwise_ece,
    column_variance,
    distance_metrics,
    feasibility_metrics,
    informativeness_ratio,
    isotonic_metrics,
    kl_divergence,
    nll,
    reverse_kl_divergence,
    sharpness_metrics,
    symmetrized_kl,
    tie_group_variance,
    top_label_ece,
)

RNG = np.random.default_rng(0)
P = RNG.dirichlet(np.ones(3), size=50)
Q = RNG.dirichlet(np.ones(3), size=50)


class TestKL:
    def test_matches_scipy_relative_entropy(self):
        assert kl_divergence(Q, P) == pytest.approx(rel_entr(Q, P).sum())

    def test_reverse_swaps_arguments(self):
        assert reverse_kl_divergence(Q, P) == pytest.approx(rel_entr(P, Q).sum())

    def test_symmetrized_is_mean_of_both_directions(self):
        both = 0.5 * (rel_entr(Q, P).sum() + rel_entr(P, Q).sum())
        assert symmetrized_kl(Q, P) == pytest.approx(both)

    def test_zero_entries_of_first_argument_contribute_nothing(self):
        q = np.array([[0.0, 1.0]])
        r = np.array([[0.5, 0.5]])
        assert kl_divergence(q, r) == pytest.approx(np.log(2))


class TestFeasibility:
    def test_row_and_column_errors(self):
        q = np.array([[0.6, 0.5], [0.3, 0.7]])
        out = feasibility_metrics(q, M=np.array([1.0, 1.0]))
        assert out["row"]["max_abs_error"] == pytest.approx(0.1)
        assert out["row"]["mean_abs_error"] == pytest.approx(0.05)
        assert out["row"]["min_value"] == pytest.approx(0.3)
        # column sums are 0.9 and 1.2 against targets of 1
        assert out["col"]["max_abs_error"] == pytest.approx(0.2)
        assert out["col"]["l1_error"] == pytest.approx(0.3)
        assert out["col"]["l2_error"] == pytest.approx(np.hypot(0.1, 0.2))
        assert out["col"]["rel_max_abs_error"] == pytest.approx(0.2)

    def test_without_targets_reports_rows_only(self):
        assert "col" not in feasibility_metrics(P)


class TestIsotonic:
    def test_violation_is_measured_in_the_order_of_the_original_scores(self):
        p = np.array([[0.3, 0.7], [0.1, 0.9], [0.2, 0.8]])
        q = np.array([[0.4, 0.6], [0.5, 0.5], [0.3, 0.7]])
        # class 0 in the order of p: q = 0.5, 0.3, 0.4 -> one drop of 0.2
        # class 1 in the order of p: q = 0.6, 0.7, 0.5 -> one drop of 0.2
        out = isotonic_metrics(q, p)
        assert out["max_rank_violation"] == pytest.approx(0.2)
        assert out["total_violation_mass"] == pytest.approx(0.4)
        assert out["mean_flat_fraction"] == 0.0

    def test_flat_runs_are_counted(self):
        p = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])
        q = np.array([[0.2, 0.8], [0.2, 0.8], [0.6, 0.4]])
        out = isotonic_metrics(q, p)
        assert out["max_rank_violation"] == 0.0
        assert out["per_class"]["flat_fraction"] == [0.5, 0.5]

    def test_tie_group_variance_is_the_variance_within_equal_scores(self):
        p = np.array([[0.5, 0.5], [0.5, 0.5], [0.2, 0.8]])
        q = np.array([[0.4, 0.6], [0.6, 0.4], [0.1, 0.9]])
        per_class = tie_group_variance(q, p)["per_class"]
        assert per_class[0]["group_count"] == 1
        assert per_class[0]["mean_within_group_var"] == pytest.approx(0.01)
        assert per_class[1]["mean_within_group_var"] == pytest.approx(0.01)


def test_distance_metrics():
    p = np.array([[0.5, 0.5], [0.2, 0.8]])
    q = np.array([[0.6, 0.4], [0.2, 0.8]])
    out = distance_metrics(q, p)
    assert out["frobenius_sq"] == pytest.approx(0.02)
    assert out["frobenius"] == pytest.approx(np.sqrt(0.02))
    assert out["mean_abs"] == pytest.approx(0.05)
    assert out["max_abs"] == pytest.approx(0.1)


class TestScoringRules:
    y = np.array([0, 1])
    probs = np.array([[0.8, 0.2], [0.4, 0.6]])

    def test_nll(self):
        assert nll(self.y, self.probs) == pytest.approx(-np.log([0.8, 0.6]).mean())

    def test_brier_sums_squared_error_over_classes(self):
        # (0.2^2 + 0.2^2 + 0.4^2 + 0.4^2) / 2 rows
        assert brier(self.y, self.probs) == pytest.approx(0.2)


class TestECE:
    def test_top_label_ece(self):
        y = np.array([0, 0, 1, 1])
        probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
        # confidences 0.9, 0.8, 0.7, 0.6 all fall in the upper of two bins;
        # mean confidence 0.75, accuracy 3/4
        out = top_label_ece(y, probs, n_bins=2)
        assert out["ece"] == pytest.approx(0.0)
        assert out["bins"][1] == (4, pytest.approx(0.75), pytest.approx(0.75))

        out = top_label_ece(y, probs, n_bins=10)
        # four singleton bins with gaps 0.1, 0.2, 0.3, 0.6
        assert out["ece"] == pytest.approx((0.1 + 0.2 + 0.3 + 0.6) / 4)
        assert out["mce"] == pytest.approx(0.6)

    def test_classwise_ece_is_the_mean_over_classes(self):
        """Kull et al. (2019), eq. 4: (1/k) sum_j sum_i |B_ij|/n |y_j - p_j|."""
        y = np.array([0, 1, 2, 2])
        probs = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.2, 0.7, 0.1],
                [0.2, 0.2, 0.6],
                [0.6, 0.2, 0.2],
            ]
        )
        # Two bins per class, split at 0.5.
        # class 0: low bin {0.2, 0.2} mean 0.2, freq 0 -> 2/4 * 0.2 = 0.1
        #          high bin {0.7, 0.6} mean 0.65, freq 1/2 -> 2/4 * 0.15 = 0.075
        # class 1: low bin {0.2, 0.2, 0.2} mean 0.2, freq 0 -> 3/4 * 0.2 = 0.15
        #          high bin {0.7} mean 0.7, freq 1 -> 1/4 * 0.3 = 0.075
        # class 2: low bin {0.1, 0.1, 0.2} mean 0.1333, freq 1/3 -> 3/4 * 0.2 = 0.15
        #          high bin {0.6} mean 0.6, freq 1 -> 1/4 * 0.4 = 0.1
        class_eces = [0.175, 0.225, 0.25]
        out = classwise_ece(y, probs, n_bins=2)
        assert [c["ece"] for c in out["per_class"]] == pytest.approx(class_eces)
        assert out["ece"] == pytest.approx(np.mean(class_eces))
        assert out["mce"] == pytest.approx(0.4)


def test_sharpness_matches_scipy_entropy():
    out = sharpness_metrics(P)
    top_two = np.sort(P, axis=1)[:, -2:]
    assert out["mean_entropy"] == pytest.approx(entropy(P, axis=1).mean())
    assert out["mean_max_prob"] == pytest.approx(P.max(axis=1).mean())
    assert out["mean_margin"] == pytest.approx((top_two[:, 1] - top_two[:, 0]).mean())


class TestVariance:
    def test_column_variance(self):
        q = np.array([[0.2, 0.8], [0.4, 0.6]])
        out = column_variance(q)
        assert out["per_column"] == pytest.approx([0.01, 0.01])
        assert [out["mean"], out["min"], out["max"]] == pytest.approx([0.01] * 3)

    def test_informativeness_ratio_compares_variances(self):
        p = np.array([[0.1, 0.9], [0.9, 0.1]])
        q = np.array([[0.3, 0.7], [0.7, 0.3]])
        out = informativeness_ratio(q, p)
        # every entry is 0.5 +/- 0.2 against 0.5 +/- 0.4
        assert out["total_ratio"] == pytest.approx(0.25)
        assert out["per_column_ratio"] == pytest.approx([0.25, 0.25])
        assert out["interpretation"].startswith("Low")

    def test_informativeness_ratio_with_a_flat_original(self):
        flat = np.full((2, 2), 0.5)
        assert informativeness_ratio(flat, flat)["total_ratio"] == 1.0
        out = informativeness_ratio(P[:2, :2], flat)
        assert out["total_ratio"] == np.inf
        assert np.isnan(out["mean_column_ratio"])


class TestAUC:
    def test_ties_count_half(self):
        """A calibrated column often has flat runs; their order is arbitrary."""
        scores = np.array([0.5, 0.5, 0.5, 0.9])
        y = np.array([1, 0, 0, 1])
        p = np.column_stack([1 - scores, scores])
        before = auc_deltas(y, p, p)["per_class"]["before"][1]
        u = mannwhitneyu(scores[y == 1], scores[y == 0]).statistic
        assert before == pytest.approx(u / (2 * 2))
        assert before == pytest.approx(0.75)

    def test_does_not_depend_on_row_order_within_ties(self):
        scores = np.array([0.5, 0.5, 0.5, 0.9])
        y = np.array([1, 0, 0, 1])
        order = [1, 2, 0, 3]
        p = np.column_stack([1 - scores, scores])
        a = auc_deltas(y, p, p)["per_class"]["before"]
        b = auc_deltas(y[order], p[order], p[order])["per_class"]["before"]
        assert a == pytest.approx(b)

    def test_matches_mann_whitney_on_random_scores(self):
        y = RNG.integers(0, 3, size=50)
        out = auc_deltas(y, P, Q)
        for j in range(3):
            pos, neg = y == j, y != j
            u_p = mannwhitneyu(P[pos, j], P[neg, j]).statistic
            u_q = mannwhitneyu(Q[pos, j], Q[neg, j]).statistic
            n = pos.sum() * neg.sum()
            assert out["per_class"]["before"][j] == pytest.approx(u_p / n)
            assert out["per_class"]["after"][j] == pytest.approx(u_q / n)
            assert out["per_class"]["delta"][j] == pytest.approx((u_q - u_p) / n)

    def test_class_absent_from_labels_is_nan(self):
        y = np.zeros(50, dtype=int)
        out = auc_deltas(y, P, Q)
        assert np.isnan(out["per_class"]["before"]).all()
        assert np.isnan(out["mean_delta_auc"])
