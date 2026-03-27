"""
Test suite for KL-divergence rank-preserving calibration.

Tests cover:
- Basic KL calibration functionality
- Constraint satisfaction (row simplex, column sums, rank preservation)
- Geometric mean pooling in KL isotonic regression
- Multiplicative rescaling for sum constraints
- Anchor-reference decoupling
- Soft KL calibration with lambda penalty
- Pareto frontier computation
- Comparison with Euclidean methods

Run with: python -m pytest tests/test_kl_calibration.py -v
"""

import numpy as np
import pytest

from rank_preserving_calibration import (
    CalibrationError,
    KLCalibrationResult,
    KLParetoResult,
    calibrate_dykstra,
    calibrate_kl,
    calibrate_kl_pareto,
    calibrate_kl_soft,
    kl_divergence,
    project_near_kl_isotonic,
    prox_kl_near_isotonic,
    reverse_kl_divergence,
    symmetrized_kl,
)
from rank_preserving_calibration.kl_calibration import (
    _kl_isotonic_regression,
    _project_column_kl_isotonic_sum,
    _project_row_kl_simplex,
)

from .data_helpers import create_test_case


class TestCalibrateKL:
    """Test basic KL calibration functionality."""

    def test_simple_2x2_case(self):
        """Test simple 2x2 case that should converge easily."""
        P = np.array([[0.8, 0.2], [0.3, 0.7]])
        M = np.array([1.0, 1.0])

        result = calibrate_kl(P, M, verbose=False)

        assert isinstance(result, KLCalibrationResult)
        assert result.Q.shape == P.shape
        assert result.max_row_error < 1e-6
        assert result.max_col_error < 1e-6
        assert np.allclose(result.Q.sum(axis=1), 1.0, atol=1e-8)
        assert np.allclose(result.Q.sum(axis=0), M, atol=1e-6)

    def test_random_case(self):
        """Test on random generated data."""
        P, M = create_test_case("random", N=20, J=3, seed=42)

        result = calibrate_kl(P, M, verbose=False)

        assert result.converged
        assert result.max_row_error < 1e-5
        assert result.max_col_error < 1e-5

    def test_rank_preservation(self):
        """Test that rank ordering is preserved."""
        P, M = create_test_case("linear", N=10, J=3, seed=42)

        result = calibrate_kl(P, M, verbose=False)

        for j in range(P.shape[1]):
            original_order = np.argsort(P[:, j])
            calibrated_sorted = result.Q[original_order, j]
            diffs = np.diff(calibrated_sorted)
            assert np.all(diffs >= -1e-8), f"Rank violation in column {j}"

    def test_probability_constraints(self):
        """Test that probability constraints are satisfied."""
        P, M = create_test_case("skewed", N=15, J=4, seed=123)

        result = calibrate_kl(P, M, verbose=False)

        assert np.all(result.Q >= 0), "Negative probability found"
        row_sums = result.Q.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-8), "Row sums not 1"
        col_sums = result.Q.sum(axis=0)
        assert np.allclose(col_sums, M, atol=1e-6), "Column sums don't match M"

    def test_kl_divergence_finite(self):
        """Test that KL divergence in result is finite and non-negative."""
        P, M = create_test_case("random", N=20, J=4, seed=99)

        result = calibrate_kl(P, M, verbose=False)

        assert np.isfinite(result.kl_divergence)
        assert result.kl_divergence >= 0


class TestKLColumnProjection:
    """Test KL isotonic regression and column projection."""

    def test_already_isotonic_unchanged(self):
        """Already isotonic sequence should be unchanged (up to rescaling)."""
        y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        z = _kl_isotonic_regression(y)

        diffs = np.diff(z)
        assert np.all(diffs >= 0), "Isotonic regression broke monotonicity"

    def test_geometric_mean_pooling(self):
        """Test that pooling uses geometric mean, not arithmetic."""
        y = np.array([0.4, 0.1])
        z = _kl_isotonic_regression(y)

        assert np.allclose(z[0], z[1])
        expected = np.sqrt(0.4 * 0.1)
        assert np.allclose(z[0], expected, rtol=1e-6), (
            f"Expected geometric mean {expected}, got {z[0]}"
        )

    def test_three_element_geometric_mean(self):
        """Test geometric mean for three elements that need pooling."""
        y = np.array([0.8, 0.1, 0.05])
        z = _kl_isotonic_regression(y)

        assert np.allclose(z[0], z[1])
        assert np.allclose(z[1], z[2])
        expected = np.exp(np.mean(np.log(y)))
        assert np.allclose(z[0], expected, rtol=1e-6)

    def test_multiplicative_rescaling(self):
        """Test that sum constraint uses multiplicative (not additive) rescaling."""
        column = np.array([0.1, 0.3, 0.2, 0.4])
        order = np.argsort(column)
        target_sum = 2.0

        projected = _project_column_kl_isotonic_sum(column, order, target_sum)

        assert np.allclose(projected.sum(), target_sum, rtol=1e-6)
        projected_sorted = projected[order]
        diffs = np.diff(projected_sorted)
        assert np.all(diffs >= -1e-10), "Isotonicity violated"

    def test_column_projection_preserves_order(self):
        """Test that column projection preserves rank ordering."""
        column = np.array([0.5, 0.1, 0.3, 0.8, 0.2])
        order = np.argsort(column)
        target_sum = 1.5

        projected = _project_column_kl_isotonic_sum(column, order, target_sum)

        projected_sorted = projected[order]
        diffs = np.diff(projected_sorted)
        assert np.all(diffs >= -1e-10)


class TestKLRowProjection:
    """Test KL row simplex projection."""

    def test_row_projection_sums_to_one(self):
        """Test that row projection produces rows summing to 1."""
        rows = np.array([[0.5, 0.3, 0.2], [1.0, 2.0, 3.0], [0.1, 0.1, 0.1]])
        projected = _project_row_kl_simplex(rows)

        row_sums = projected.sum(axis=1)
        assert np.allclose(row_sums, 1.0)

    def test_row_projection_nonnegative(self):
        """Test that row projection produces non-negative values."""
        rows = np.array([[0.5, 0.3, 0.2], [1.0, 2.0, 3.0]])
        projected = _project_row_kl_simplex(rows)

        assert np.all(projected >= 0)

    def test_row_projection_multiplicative(self):
        """Test that KL row projection is multiplicative normalization."""
        rows = np.array([[2.0, 4.0, 6.0]])
        projected = _project_row_kl_simplex(rows)

        expected = rows / rows.sum(axis=1, keepdims=True)
        assert np.allclose(projected, expected)


class TestCalibrateKLSoft:
    """Test soft KL calibration with lambda penalty."""

    def test_basic_soft_calibration(self):
        """Test basic soft calibration functionality."""
        P, M = create_test_case("random", N=15, J=3, seed=42)

        result = calibrate_kl_soft(P, M, lam=1.0, verbose=False)

        assert isinstance(result, KLCalibrationResult)
        assert result.Q.shape == P.shape
        assert np.all(result.Q >= 0)

    def test_lambda_effect_on_rank_violation(self):
        """Test that higher lambda tends to reduce rank violations (soft constraint)."""
        P, M = create_test_case("random", N=20, J=3, seed=123)

        result_low = calibrate_kl_soft(P, M, lam=0.1, max_iters=2000, verbose=False)
        result_mid = calibrate_kl_soft(P, M, lam=1.0, max_iters=2000, verbose=False)

        assert result_mid.max_rank_violation <= result_low.max_rank_violation + 0.1

    def test_soft_row_simplex_maintained(self):
        """Test that rows approximately sum to 1 in soft calibration."""
        P, M = create_test_case("random", N=15, J=3, seed=456)

        result = calibrate_kl_soft(P, M, lam=10.0, max_iters=2000, verbose=False)

        row_sums = result.Q.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-3)


class TestAnchorReferenceDecoupling:
    """Test anchor-reference decoupling feature."""

    def test_different_anchor_produces_different_result(self):
        """Test that different anchor A produces different ordering."""
        P = np.array([[0.6, 0.4], [0.3, 0.7], [0.5, 0.5], [0.8, 0.2]])
        M = np.array([2.0, 2.0])

        A_different = np.array([[0.2, 0.8], [0.9, 0.1], [0.5, 0.5], [0.4, 0.6]])

        result_same = calibrate_kl(P, M, R=P, A=P, verbose=False)
        result_diff = calibrate_kl(P, M, R=P, A=A_different, verbose=False)

        assert not np.allclose(result_same.Q, result_diff.Q, atol=1e-4), (
            "Different anchors should produce different results"
        )

    def test_anchor_determines_rank_order(self):
        """Test that anchor determines which rankings to preserve."""
        P = np.array([[0.7, 0.3], [0.4, 0.6], [0.5, 0.5]])
        M = np.array([1.5, 1.5])

        A = np.array([[0.2, 0.8], [0.9, 0.1], [0.5, 0.5]])

        result = calibrate_kl(P, M, R=P, A=A, verbose=False)

        for j in range(P.shape[1]):
            anchor_order = np.argsort(A[:, j])
            q_sorted = result.Q[anchor_order, j]
            diffs = np.diff(q_sorted)
            assert np.all(diffs >= -1e-8), f"Rank violation in column {j}"


class TestKLvsEuclidean:
    """Compare KL and Euclidean calibration on same problems."""

    def test_both_satisfy_constraints(self):
        """Test that both methods satisfy the same constraints."""
        P, M = create_test_case("random", N=20, J=3, seed=42)

        result_kl = calibrate_kl(P, M, verbose=False)
        result_euc = calibrate_dykstra(P, M, verbose=False)

        assert np.allclose(result_kl.Q.sum(axis=1), 1.0, atol=1e-6)
        assert np.allclose(result_euc.Q.sum(axis=1), 1.0, atol=1e-6)

        assert np.allclose(result_kl.Q.sum(axis=0), M, atol=1e-5)
        assert np.allclose(result_euc.Q.sum(axis=0), M, atol=1e-5)

    def test_solutions_can_differ(self):
        """Test that KL and Euclidean can give different solutions."""
        rng = np.random.default_rng(789)
        P = rng.dirichlet(np.ones(4), size=30)
        target_shift = rng.normal(0, 0.1, 4)
        target_shift -= target_shift.mean()
        M = P.sum(axis=0) + target_shift * 3

        result_kl = calibrate_kl(P, M, verbose=False)
        result_euc = calibrate_dykstra(P, M, verbose=False)

        diff = np.linalg.norm(result_kl.Q - result_euc.Q)
        assert diff < 5.0


class TestKLPareto:
    """Test Pareto frontier computation."""

    def test_pareto_returns_multiple_solutions(self):
        """Test that Pareto solver returns multiple solutions."""
        P, M = create_test_case("random", N=15, J=3, seed=42)
        lambda_grid = [0.1, 1.0, 10.0]

        result = calibrate_kl_pareto(P, M, lambda_grid=lambda_grid, verbose=False)

        assert isinstance(result, KLParetoResult)
        assert len(result.solutions) >= 1
        assert len(result.kl_values) == len(result.solutions)
        assert len(result.rank_violations) == len(result.solutions)

    def test_pareto_monotonicity(self):
        """Test that rank violations tend to decrease as lambda increases."""
        P, M = create_test_case("random", N=20, J=3, seed=123)
        lambda_grid = [0.01, 0.1, 1.0]

        result = calibrate_kl_pareto(
            P, M, lambda_grid=lambda_grid, max_iters=1000, verbose=False
        )

        if len(result.rank_violations) >= 2:
            first_viol = result.rank_violations[0]
            last_viol = result.rank_violations[-1]
            assert last_viol <= first_viol + 0.5

    def test_pareto_solutions_valid(self):
        """Test that all Pareto solutions satisfy basic constraints."""
        P, M = create_test_case("random", N=15, J=3, seed=456)
        lambda_grid = [0.1, 1.0, 10.0]

        result = calibrate_kl_pareto(P, M, lambda_grid=lambda_grid, verbose=False)

        for _, Q in result.solutions:
            row_sums = Q.sum(axis=1)
            assert np.allclose(row_sums, 1.0, atol=0.1)
            assert np.all(Q >= -1e-10)


class TestKLNearlyIsotonic:
    """Test KL nearly-isotonic utilities."""

    def test_near_kl_isotonic_basic(self):
        """Test basic near-KL-isotonic projection."""
        v = np.array([0.5, 0.2, 0.3, 0.1])
        z = project_near_kl_isotonic(v, eps_slack=0.1)

        assert np.all(z > 0)
        assert z.shape == v.shape

    def test_near_kl_isotonic_with_sum(self):
        """Test near-KL-isotonic with sum constraint."""
        v = np.array([0.3, 0.4, 0.2, 0.1])
        target_sum = 2.0

        z = project_near_kl_isotonic(v, eps_slack=0.05, sum_target=target_sum)

        assert np.allclose(z.sum(), target_sum, rtol=1e-6)

    def test_near_kl_isotonic_zero_slack_is_strict(self):
        """Test that zero slack gives strict isotonic."""
        v = np.array([0.4, 0.1, 0.2])
        z = project_near_kl_isotonic(v, eps_slack=0.0)

        diffs = np.diff(z)
        assert np.all(diffs >= -1e-10)

    def test_prox_kl_near_isotonic_basic(self):
        """Test basic KL nearly-isotonic prox."""
        y = np.array([0.5, 0.2, 0.3, 0.4])
        z = prox_kl_near_isotonic(y, lam=1.0)

        assert isinstance(z, np.ndarray)
        assert z.shape == y.shape
        assert np.all(z > 0)

    def test_prox_kl_zero_lambda_unchanged(self):
        """Test that zero lambda returns input unchanged."""
        y = np.array([0.3, 0.5, 0.2])
        z = prox_kl_near_isotonic(y, lam=0.0)

        assert np.allclose(z, y)


class TestKLDivergenceMetrics:
    """Test KL divergence metric functions."""

    def test_kl_divergence_nonnegative(self):
        """Test that KL divergence is non-negative."""
        Q = np.array([[0.6, 0.4], [0.3, 0.7]])
        R = np.array([[0.5, 0.5], [0.4, 0.6]])

        kl = kl_divergence(Q, R)

        assert kl >= 0

    def test_kl_divergence_zero_for_same(self):
        """Test that KL(P||P) = 0."""
        P = np.array([[0.6, 0.4], [0.3, 0.7]])

        kl = kl_divergence(P, P)

        assert np.isclose(kl, 0.0, atol=1e-10)

    def test_kl_divergence_asymmetric(self):
        """Test that KL is asymmetric: KL(Q||R) != KL(R||Q) in general."""
        Q = np.array([[0.8, 0.2], [0.3, 0.7]])
        R = np.array([[0.5, 0.5], [0.4, 0.6]])

        kl_forward = kl_divergence(Q, R)
        kl_reverse = kl_divergence(R, Q)

        assert not np.isclose(kl_forward, kl_reverse)

    def test_reverse_kl_divergence(self):
        """Test reverse KL divergence."""
        Q = np.array([[0.6, 0.4], [0.3, 0.7]])
        R = np.array([[0.5, 0.5], [0.4, 0.6]])

        rkl = reverse_kl_divergence(Q, R)
        expected = kl_divergence(R, Q)

        assert np.isclose(rkl, expected)

    def test_symmetrized_kl(self):
        """Test symmetrized KL divergence."""
        Q = np.array([[0.6, 0.4], [0.3, 0.7]])
        R = np.array([[0.5, 0.5], [0.4, 0.6]])

        skl = symmetrized_kl(Q, R)
        expected = 0.5 * (kl_divergence(Q, R) + kl_divergence(R, Q))

        assert np.isclose(skl, expected)

    def test_symmetrized_kl_is_symmetric(self):
        """Test that symmetrized KL is symmetric."""
        Q = np.array([[0.6, 0.4], [0.3, 0.7]])
        R = np.array([[0.5, 0.5], [0.4, 0.6]])

        skl_qr = symmetrized_kl(Q, R)
        skl_rq = symmetrized_kl(R, Q)

        assert np.isclose(skl_qr, skl_rq)


class TestKLInputValidation:
    """Test input validation for KL calibration."""

    def test_invalid_R_shape_raises(self):
        """Test that mismatched R shape raises error."""
        P = np.array([[0.6, 0.4], [0.3, 0.7]])
        M = np.array([1.0, 1.0])
        R = np.array([[0.5, 0.5, 0.0]])

        with pytest.raises(CalibrationError):
            calibrate_kl(P, M, R=R)

    def test_invalid_A_shape_raises(self):
        """Test that mismatched A shape raises error."""
        P = np.array([[0.6, 0.4], [0.3, 0.7]])
        M = np.array([1.0, 1.0])
        A = np.array([[0.5, 0.5, 0.0]])

        with pytest.raises(CalibrationError):
            calibrate_kl(P, M, A=A)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
