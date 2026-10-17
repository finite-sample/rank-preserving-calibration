"""Degenerate inputs and argument checks for the helpers, with exact expectations."""

import numpy as np
import pytest
from scipy.optimize import isotonic_regression

from rank_preserving_calibration import (
    CalibrationError,
    calibrate_ipf,
    calibrate_kl,
    calibrate_kl_pareto,
    calibrate_soft,
    calibrate_two_stage,
    compare_calibration_methods,
    flatness_bound,
    flatness_metrics,
    informativeness_ratio,
    isotonic_metrics,
    project_near_isotonic_euclidean,
    prox_near_isotonic,
    prox_near_isotonic_with_sum,
)
from rank_preserving_calibration._isotonic import (
    _isotonic_regression as _pav_increasing,
)
from rank_preserving_calibration.calibration import (
    _project_column_isotonic_sum,
    _run_lengths_of_equals,
)
from rank_preserving_calibration.kl_nearly import (
    _kl_pav_increasing,
    project_near_kl_isotonic,
    prox_kl_near_isotonic,
)
from rank_preserving_calibration.metrics import classwise_ece
from rank_preserving_calibration.soft_calibration import _compute_rank_penalty

Y = np.array([3.0, 1.0, 1.0, 2.0, 0.5, 0.5, 4.0, 3.5])
W = np.array([0.5, 1.5, 2.25, 1.0, 0.1, 3.0, 0.7, 1.2])


def _problem(N=12, J=3, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(J), size=N)
    M = P.sum(axis=0) * np.linspace(1.2, 0.8, J)
    return P, M * N / M.sum()


class TestPAVVariants:
    def test_nearly_weighted_pav_matches_scipy(self):
        expected = isotonic_regression(Y, weights=W).x
        np.testing.assert_allclose(_pav_increasing(Y, W), expected, atol=1e-12)

    @pytest.mark.parametrize(
        ("w", "message"), [(np.ones(3), "same shape"), (-W, "positive")]
    )
    def test_reject_bad_weights(self, w, message):
        with pytest.raises(ValueError, match=message):
            _pav_increasing(Y, w)

    def test_kl_pav_is_geometric_pooling(self):
        # Constant z = c on a block minimises sum(z log(z / y) - z + y) at
        # log c = weighted mean of log y, so the KL fit is PAV in log space.
        expected = np.exp(isotonic_regression(np.log(Y), weights=W).x)
        np.testing.assert_allclose(_kl_pav_increasing(Y, W), expected)

    @pytest.mark.parametrize(
        "fit",
        [
            _kl_pav_increasing,
            lambda y: prox_near_isotonic(y, 1.0),
            lambda y: prox_kl_near_isotonic(y, 1.0),
        ],
    )
    def test_single_element_is_returned_unchanged(self, fit):
        np.testing.assert_array_equal(fit(np.array([0.4])), [0.4])

    def test_run_lengths_of_empty_array(self):
        assert _run_lengths_of_equals(np.array([])).size == 0


class TestNearlyIsotonic:
    def test_single_element_takes_the_sum_target(self):
        np.testing.assert_allclose(
            project_near_isotonic_euclidean(np.array([0.4]), 0.1, sum_target=2.0),
            [2.0],
        )
        np.testing.assert_allclose(
            project_near_kl_isotonic(np.array([0.4]), 0.1, sum_target=2.0), [2.0]
        )

    @pytest.mark.parametrize(
        ("call", "message"),
        [
            (lambda: project_near_isotonic_euclidean(Y, -0.1), "eps"),
            (lambda: project_near_kl_isotonic(Y, 1.0), "eps_slack"),
            (lambda: prox_near_isotonic(Y, -1.0), "lam"),
            (lambda: prox_kl_near_isotonic(Y, -1.0), "lam"),
        ],
    )
    def test_rejects_bad_parameters(self, call, message):
        with pytest.raises(ValueError, match=message):
            call()

    @pytest.mark.parametrize("prox", [prox_near_isotonic, prox_kl_near_isotonic])
    def test_zero_penalty_is_the_identity(self, prox):
        np.testing.assert_allclose(prox(Y, 0.0), Y)

    def test_kl_prox_reports_its_iterations(self):
        _, info = prox_kl_near_isotonic(Y, 0.0, return_info=True)
        assert info["iterations"] == 0
        _, info = prox_kl_near_isotonic(np.array([0.4]), 1.0, return_info=True)
        assert info == {"iterations": 0, "converged": True}
        _, info = prox_kl_near_isotonic(Y, 1.0, max_iters=1, return_info=True)
        assert info == {"iterations": 1, "converged": False}

    def test_prox_with_sum_is_a_translation(self):
        z = prox_near_isotonic(Y, 1.0)
        shifted = prox_near_isotonic_with_sum(Y, 1.0, 10.0)
        np.testing.assert_allclose(shifted, z + (10.0 - z.sum()) / Y.size)
        assert np.sum(shifted) == pytest.approx(10.0)

    def test_kl_slack_projection_with_weights_and_no_slack_is_kl_pav(self):
        expected = np.exp(isotonic_regression(np.log(Y), weights=W).x)
        np.testing.assert_allclose(
            project_near_kl_isotonic(Y, 0.0, weights=W), expected
        )


class TestColumnProjections:
    def test_empty_columns(self):
        empty = np.array([])
        order = np.array([], dtype=np.int64)
        assert _project_column_isotonic_sum(empty, order, 0.0).size == 0


class TestSolverEdges:
    def test_calibrate_kl_raises_on_infeasible_targets(self):
        P, M = _problem()
        with (
            pytest.warns(UserWarning, match="must equal N"),
            pytest.raises(CalibrationError, match="status"),
        ):
            calibrate_kl(P, M * 1.1)

    def test_pareto_uses_the_given_grid_in_order(self):
        P, M = _problem()
        result = calibrate_kl_pareto(P, M, lambda_grid=[10.0, 0.1])
        assert result.lambda_path == [0.1, 10.0]

    def test_soft_warns_when_targets_are_far_from_n(self):
        P, M = _problem()
        with pytest.warns(UserWarning, match="differs significantly"):
            calibrate_soft(P, M * 1.5)

    def test_ipf_and_two_stage_report_an_unconverged_run(self):
        P, M = _problem()
        assert not calibrate_ipf(P, M, max_iters=1).converged
        result = calibrate_two_stage(P, M, ipf_max_iters=1, proj_max_iters=1)
        assert not result.converged
        assert result.projection_iterations == 1

    def test_rank_penalty_ignores_single_row_columns(self):
        assert _compute_rank_penalty(np.ones((1, 2)), [np.zeros(1, int)] * 2) == 0.0


class TestMetricEdges:
    def test_isotonic_metrics_on_one_row(self):
        out = isotonic_metrics(np.array([[0.4, 0.6]]), np.array([[0.5, 0.5]]))
        assert out["max_rank_violation"] == 0.0

    def test_classwise_ece_records_empty_bins(self):
        y = np.array([0, 1])
        probs = np.array([[0.9, 0.1], [0.2, 0.8]])
        bins = classwise_ece(y, probs, n_bins=4)["per_class"][0]["bins"]
        assert bins[1][0] == 0
        assert np.isnan(bins[1][1])

    def test_informativeness_moderate_band(self):
        P = np.array([[0.9, 0.1], [0.1, 0.9]])
        # Scaling the spread around 0.5 by sqrt(0.6) keeps 0.6 of the
        # variance, inside the [0.5, 0.8) "moderate" band.
        Q = 0.5 + (P - 0.5) * np.sqrt(0.6)
        result = informativeness_ratio(Q, P)
        assert result["total_ratio"] == pytest.approx(0.6)
        assert result["interpretation"].startswith("Moderate")

    def test_flatness_against_a_flat_original(self):
        P = np.full((4, 2), 0.5)
        flat = flatness_metrics(P, P)
        assert flat["variance_ratio"] == 1.0
        assert flat["column_variance_ratio"] == [1.0, 1.0]
        Q = np.array([[0.4, 0.6], [0.5, 0.5], [0.5, 0.5], [0.6, 0.4]])
        assert flatness_metrics(Q, P)["variance_ratio"] == float("inf")

    def test_flatness_bound_extremes(self):
        P = np.full((4, 2), 0.5)
        bound = flatness_bound(P, np.array([4.0, 0.0]))
        assert bound["expected_variance_ratio"] == 1.0
        assert bound["recommendation"].startswith("Very high")

    def test_compare_methods_on_one_row(self):
        P = np.array([[0.4, 0.6]])
        out = compare_calibration_methods(P, np.array([0.4, 0.6]), {"same": P})
        assert out["methods"]["same"]["max_rank_violation"] == 0.0
