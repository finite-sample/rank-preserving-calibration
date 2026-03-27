"""
Tests for analysis utilities (flatness, shift metrics).

Run with: python -m pytest tests/test_analysis.py -v
"""

import numpy as np
import pytest

from rank_preserving_calibration import (
    column_variance,
    compare_calibration_methods,
    flatness_bound,
    flatness_metrics,
    informativeness_ratio,
    marginal_shift_metrics,
)

from .data_helpers import create_test_case


class TestFlatnessMetrics:
    """Test flatness_metrics function."""

    def test_basic_functionality(self):
        """Test basic flatness metrics computation."""
        Q = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.3, 0.4, 0.3]])

        result = flatness_metrics(Q)

        assert "column_variance" in result
        assert "mean_column_variance" in result
        assert "total_variance" in result
        assert "entropy_per_row" in result
        assert "mean_max_prob" in result

        assert len(result["column_variance"]) == 3
        assert result["mean_column_variance"] >= 0
        assert result["total_variance"] >= 0

    def test_with_original_P(self):
        """Test flatness metrics with original P for variance ratio."""
        P = np.array([[0.8, 0.15, 0.05], [0.1, 0.7, 0.2]])
        Q = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])  # Flatter

        result = flatness_metrics(Q, P)

        assert "variance_ratio" in result
        assert "column_variance_ratio" in result
        # Q is flatter, so variance ratio should be < 1
        assert result["variance_ratio"] < 1.0

    def test_with_target_M(self):
        """Test flatness metrics with target M for uniform distance."""
        Q = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
        M = np.array([0.9, 0.7, 0.4])

        result = flatness_metrics(Q, M=M)

        assert "distance_to_uniform" in result
        assert "relative_distance_to_uniform" in result
        assert result["distance_to_uniform"] >= 0

    def test_uniform_solution(self):
        """Test metrics for a uniform solution."""
        N, J = 4, 3
        M = np.array([N / J] * J)
        Q_uniform = np.outer(np.ones(N), M / N)

        result = flatness_metrics(Q_uniform, M=M)

        # Uniform solution should have ~0 distance to uniform
        assert result["distance_to_uniform"] < 1e-10
        # Very low column variance
        assert result["mean_column_variance"] < 1e-10

    def test_high_variance_solution(self):
        """Test metrics for high-variance solution."""
        Q = np.array([[0.99, 0.005, 0.005], [0.01, 0.98, 0.01], [0.02, 0.02, 0.96]])
        P = np.array([[0.5, 0.3, 0.2], [0.3, 0.5, 0.2], [0.4, 0.3, 0.3]])

        result = flatness_metrics(Q, P)

        # Q has higher variance than P
        assert result["variance_ratio"] > 1.0


class TestMarginalShiftMetrics:
    """Test marginal_shift_metrics function."""

    def test_basic_functionality(self):
        """Test basic shift metrics computation."""
        P = np.array([[0.6, 0.3, 0.1], [0.4, 0.4, 0.2]])
        M = np.array([0.8, 0.8, 0.4])

        result = marginal_shift_metrics(P, M)

        assert "empirical_marginals" in result
        assert "target_marginals" in result
        assert "marginal_diff" in result
        assert "l1_shift" in result
        assert "l2_shift" in result
        assert "max_shift" in result
        assert "relative_l2_shift" in result
        assert "per_column_relative_shift" in result

    def test_no_shift(self):
        """Test when there's no shift (M equals empirical)."""
        P = np.array([[0.5, 0.5], [0.5, 0.5]])
        M = P.sum(axis=0)

        result = marginal_shift_metrics(P, M)

        assert result["l1_shift"] < 1e-10
        assert result["l2_shift"] < 1e-10
        assert result["max_shift"] < 1e-10

    def test_large_shift(self):
        """Test with large distribution shift."""
        P = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
        M = np.array([0.5, 2.5])  # Very different from empirical

        result = marginal_shift_metrics(P, M)

        assert result["l2_shift"] > 1.0
        assert result["max_shift"] > 0.5

    def test_shift_values(self):
        """Test that shift values are computed correctly."""
        P = np.array([[0.6, 0.4], [0.4, 0.6]])
        M = np.array([1.2, 0.8])  # Empirical is [1.0, 1.0]

        result = marginal_shift_metrics(P, M)

        # diff = M - empirical = [0.2, -0.2]
        assert np.allclose(result["marginal_diff"], [0.2, -0.2])
        assert np.isclose(result["l1_shift"], 0.4)
        assert np.isclose(result["l2_shift"], np.sqrt(0.08))


class TestFlatnessBound:
    """Test flatness_bound function."""

    def test_basic_functionality(self):
        """Test basic flatness bound computation."""
        P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        M = np.array([0.8, 0.8, 0.4])

        result = flatness_bound(P, M)

        assert "shift_magnitude" in result
        assert "variance_p" in result
        assert "expected_variance_ratio" in result
        assert "flatness_risk" in result
        assert "recommendation" in result

        assert 0 <= result["flatness_risk"] <= 1
        assert result["variance_p"] >= 0

    def test_low_risk(self):
        """Test with low flatness risk (small shift)."""
        P = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.3, 0.4, 0.3]])
        M = P.sum(axis=0) * 1.01  # Very small shift

        result = flatness_bound(P, M)

        assert result["flatness_risk"] < 0.2
        assert (
            "Low" in result["recommendation"] or "Moderate" in result["recommendation"]
        )

    def test_high_risk(self):
        """Test with high flatness risk (large shift)."""
        P = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
        M = np.array([0.3, 0.7, 1.0])  # Large shift

        result = flatness_bound(P, M)

        assert result["flatness_risk"] > 0.3
        assert (
            "High" in result["recommendation"]
            or "Moderate" in result["recommendation"]
            or "Very" in result["recommendation"]
        )

    def test_expected_variance_ratio(self):
        """Test expected variance ratio bounds."""
        P, M = create_test_case("random", N=20, J=4, seed=42)

        result = flatness_bound(P, M)

        # Expected variance ratio should be between 0 and 1
        assert 0 <= result["expected_variance_ratio"] <= 1.0


class TestColumnVariance:
    """Test column_variance function."""

    def test_basic_functionality(self):
        """Test basic column variance computation."""
        Q = np.array([[0.6, 0.3, 0.1], [0.4, 0.4, 0.2], [0.3, 0.5, 0.2]])

        result = column_variance(Q)

        assert "per_column" in result
        assert "mean" in result
        assert "min" in result
        assert "max" in result

        assert len(result["per_column"]) == 3
        assert result["mean"] >= 0
        assert result["min"] <= result["mean"] <= result["max"]

    def test_uniform_columns(self):
        """Test with uniform column values."""
        Q = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [0.5, 0.3, 0.2]])

        result = column_variance(Q)

        # Each column is constant, so variance should be 0
        assert np.allclose(result["per_column"], [0, 0, 0])
        assert result["mean"] < 1e-10

    def test_high_variance_column(self):
        """Test with one high-variance column."""
        Q = np.array([[0.9, 0.5, 0.2], [0.1, 0.5, 0.2], [0.5, 0.5, 0.2]])

        result = column_variance(Q)

        # First column has high variance, others have low/zero
        assert result["per_column"][0] > result["per_column"][1]
        assert result["per_column"][0] > result["per_column"][2]


class TestInformativenessRatio:
    """Test informativeness_ratio function."""

    def test_basic_functionality(self):
        """Test basic informativeness ratio computation."""
        P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        Q = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])

        result = informativeness_ratio(Q, P)

        assert "total_ratio" in result
        assert "per_column_ratio" in result
        assert "mean_column_ratio" in result
        assert "interpretation" in result

    def test_preserved_variance(self):
        """Test when variance is preserved."""
        P = np.array([[0.6, 0.4], [0.4, 0.6]])
        Q = P.copy()  # Same as P

        result = informativeness_ratio(Q, P)

        assert np.isclose(result["total_ratio"], 1.0)
        assert "High" in result["interpretation"]

    def test_flattened_variance(self):
        """Test when variance is reduced (flattening)."""
        P = np.array([[0.9, 0.1], [0.1, 0.9]])
        Q = np.array([[0.5, 0.5], [0.5, 0.5]])  # Uniform/flat

        result = informativeness_ratio(Q, P)

        assert result["total_ratio"] < 0.1
        assert "low" in result["interpretation"].lower()

    def test_per_column_ratios(self):
        """Test per-column ratio computation."""
        P = np.array([[0.8, 0.5], [0.2, 0.5]])  # Col 1: var>0, Col 2: var=0
        Q = np.array([[0.6, 0.5], [0.4, 0.5]])

        result = informativeness_ratio(Q, P)

        # Should have 2 column ratios
        assert len(result["per_column_ratio"]) == 2


class TestCompareCalibrationMethods:
    """Test compare_calibration_methods function."""

    def test_basic_functionality(self):
        """Test basic comparison functionality."""
        P = np.array([[0.6, 0.3, 0.1], [0.3, 0.5, 0.2], [0.2, 0.4, 0.4]])
        M = np.array([1.0, 1.2, 0.8])

        Q1 = np.array([[0.55, 0.35, 0.1], [0.35, 0.45, 0.2], [0.25, 0.35, 0.4]])
        Q2 = np.array([[0.5, 0.4, 0.1], [0.3, 0.5, 0.2], [0.2, 0.4, 0.4]])

        result = compare_calibration_methods(P, M, {"method1": Q1, "method2": Q2})

        assert "methods" in result
        assert "method1" in result["methods"]
        assert "method2" in result["methods"]
        assert "ranking_by_informativeness" in result
        assert "ranking_by_feasibility" in result

    def test_method_metrics(self):
        """Test that each method has expected metrics."""
        P = np.array([[0.6, 0.4], [0.4, 0.6]])
        M = np.array([1.0, 1.0])
        Q = np.array([[0.5, 0.5], [0.5, 0.5]])

        result = compare_calibration_methods(P, M, {"test": Q})

        method_result = result["methods"]["test"]
        assert "max_row_error" in method_result
        assert "max_col_error" in method_result
        assert "max_rank_violation" in method_result
        assert "variance_ratio" in method_result
        assert "distance_from_p" in method_result

    def test_rankings(self):
        """Test that rankings are computed correctly."""
        P = np.array([[0.7, 0.3], [0.3, 0.7]])
        M = np.array([1.0, 1.0])

        Q_flat = np.array([[0.5, 0.5], [0.5, 0.5]])  # Low variance
        Q_varied = np.array([[0.7, 0.3], [0.3, 0.7]])  # High variance (same as P)

        result = compare_calibration_methods(P, M, {"flat": Q_flat, "varied": Q_varied})

        # Varied should rank higher in informativeness
        rankings = result["ranking_by_informativeness"]
        assert rankings[0] == "varied"


class TestIntegration:
    """Integration tests combining multiple analysis functions."""

    def test_analysis_workflow(self):
        """Test typical analysis workflow."""
        P, M = create_test_case("random", N=30, J=4, seed=42)

        # Check shift
        shift = marginal_shift_metrics(P, M)
        assert shift["l2_shift"] >= 0

        # Get flatness bound
        bound = flatness_bound(P, M)
        assert bound["flatness_risk"] >= 0

        # Create a simple calibrated solution
        Q = P.copy()
        row_sums = Q.sum(axis=1, keepdims=True)
        Q = Q / row_sums

        # Analyze flatness
        flat = flatness_metrics(Q, P, M)
        assert flat["mean_column_variance"] >= 0

        # Check informativeness
        info = informativeness_ratio(Q, P)
        assert info["total_ratio"] >= 0

    def test_consistent_metrics(self):
        """Test that related metrics are consistent."""
        P, _M = create_test_case("linear", N=20, J=3, seed=123)
        Q = P * 0.5 + 0.5 / 3  # Shrink toward uniform
        Q = Q / Q.sum(axis=1, keepdims=True)

        flat = flatness_metrics(Q, P)
        info = informativeness_ratio(Q, P)
        col_var = column_variance(Q)

        # Mean column variance should match
        assert np.isclose(flat["mean_column_variance"], col_var["mean"], rtol=1e-6)

        # Variance ratio should match
        assert np.isclose(flat["variance_ratio"], info["total_ratio"], rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
