"""
Utilities test suite for rank_preserving_calibration package.

Tests data generation utilities, entropy calculations, random number handling,
and other utility functions. Consolidates test_entropy.py and test_rng.py.

Run with: python -m pytest tests/test_utilities.py
"""

from types import SimpleNamespace

import numpy as np
import pytest

from .data_helpers import (
    analyze_calibration_result,
    create_realistic_classifier_case,
    create_survey_reweighting_case,
    create_test_case,
)


class TestDataGeneration:
    """Test synthetic data generation utilities."""

    def test_create_test_case_types(self):
        """Test different test case types."""
        case_types = ["random", "linear", "skewed", "challenging"]

        for case_type in case_types:
            P, M = create_test_case(case_type, N=10, J=3, seed=42)

            # Basic validity checks
            assert P.shape == (10, 3)
            assert M.shape == (3,)
            assert np.allclose(
                P.sum(axis=1), 1.0
            ), f"Rows don't sum to 1 for {case_type}"
            assert np.all(P >= 0), f"Negative probabilities in {case_type}"
            assert np.all(M > 0), f"Non-positive marginals in {case_type}"

    def test_create_test_case_parameters(self):
        """Test test case generation with different parameters."""
        # Test different sizes
        for N in [5, 10, 20]:
            for J in [2, 3, 5]:
                P, M = create_test_case("random", N=N, J=J, seed=123)
                assert P.shape == (N, J)
                assert M.shape == (J,)

    def test_invalid_case_type(self):
        """Test that invalid case types raise errors."""
        with pytest.raises(ValueError, match="Unknown case type"):
            create_test_case("invalid_type", N=10, J=3)

    def test_create_realistic_classifier_case(self):
        """Test realistic classifier case generation."""
        P, M, info = create_realistic_classifier_case(N=50, J=4, seed=456)

        assert P.shape == (50, 4)
        assert M.shape == (4,)
        assert isinstance(info, dict)
        assert "miscalibration_type" in info
        assert "true_class_distribution" in info

        # Should have reasonable properties
        assert np.allclose(P.sum(axis=1), 1.0)
        assert np.all(P >= 0)
        assert np.all(M > 0)

    def test_create_survey_reweighting_case(self):
        """Test survey reweighting case generation."""
        P, M, info = create_survey_reweighting_case(N=100, seed=789)

        # Survey case has fixed structure
        assert P.shape[0] == 100
        assert P.shape[1] > 1  # Multiple categories
        assert M.shape == (P.shape[1],)
        assert isinstance(info, dict)

        assert np.allclose(P.sum(axis=1), 1.0)
        assert np.all(P >= 0)
        assert np.all(M > 0)


class TestRandomNumberHandling:
    """Test random number generation and state management."""

    def _assert_state_unchanged(self, before):
        """Assert that random state is unchanged."""
        after = np.random.get_state()
        assert before[0] == after[0]
        assert np.array_equal(before[1], after[1])
        assert before[2] == after[2]
        assert before[3] == after[3]
        assert before[4] == after[4]

    def _assert_dict_equal(self, d1, d2):
        """Assert that two dictionaries are equal."""
        assert d1.keys() == d2.keys()
        for k in d1:
            v1, v2 = d1[k], d2[k]
            if isinstance(v1, dict):
                self._assert_dict_equal(v1, v2)
            elif isinstance(v1, np.ndarray):
                assert np.allclose(v1, v2)
            else:
                assert v1 == v2

    def test_create_test_case_deterministic_and_state_isolated(self):
        """Test that create_test_case is deterministic and doesn't affect global state."""
        np.random.seed(123)
        state_before = np.random.get_state()

        P1, M1 = create_test_case("random", N=5, J=2, seed=42)
        self._assert_state_unchanged(state_before)

        P2, M2 = create_test_case("random", N=5, J=2, seed=42)
        assert np.allclose(P1, P2), "create_test_case not deterministic"
        assert np.allclose(M1, M2), "create_test_case marginals not deterministic"

    def test_create_realistic_classifier_case_deterministic_and_state_isolated(self):
        """Test realistic classifier case determinism and state isolation."""
        np.random.seed(123)
        state_before = np.random.get_state()

        P1, M1, info1 = create_realistic_classifier_case(N=50, J=3, seed=42)
        self._assert_state_unchanged(state_before)

        P2, M2, info2 = create_realistic_classifier_case(N=50, J=3, seed=42)
        assert np.allclose(P1, P2), "Realistic classifier case not deterministic"
        assert np.allclose(M1, M2), "Realistic classifier marginals not deterministic"
        self._assert_dict_equal(info1, info2)

    def test_create_survey_reweighting_case_deterministic_and_state_isolated(self):
        """Test survey reweighting case determinism and state isolation."""
        np.random.seed(123)
        state_before = np.random.get_state()

        P1, M1, info1 = create_survey_reweighting_case(N=100, seed=42)
        self._assert_state_unchanged(state_before)

        P2, M2, info2 = create_survey_reweighting_case(N=100, seed=42)
        assert np.allclose(P1, P2), "Survey reweighting case not deterministic"
        assert np.allclose(M1, M2), "Survey reweighting marginals not deterministic"
        self._assert_dict_equal(info1, info2)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        P1, _M1 = create_test_case("random", N=10, J=3, seed=42)
        P2, _M2 = create_test_case("random", N=10, J=3, seed=123)

        # P should be different (stochastic generation)
        assert not np.allclose(P1, P2), "Different seeds produced same results"

        # M may be the same for 'random' case (designed to have equal marginals)
        # Test with a case that varies marginals instead
        P1_skewed, _M1_skewed = create_test_case("skewed", N=10, J=3, seed=42)
        P2_skewed, _M2_skewed = create_test_case("skewed", N=10, J=3, seed=123)
        assert not np.allclose(
            P1_skewed, P2_skewed
        ), "Different seeds produced same skewed results"


class TestEntropyCalculations:
    """Test entropy calculation functionality."""

    def test_entropy_regression(self):
        """Verify entropy calculations on a known small matrix."""
        P = np.array([[0.6, 0.4], [0.5, 0.5]])
        Q = np.array([[0.7, 0.3], [0.4, 0.6]])
        M = Q.sum(axis=0)

        # Create mock result object
        result = SimpleNamespace(
            Q=Q, converged=True, iterations=0, final_change=0.0, max_rank_violation=0.0
        )

        analysis = analyze_calibration_result(P, result, M)

        # Compute expected entropies manually
        expected_P_entropy = np.mean(-np.sum(P * np.log(P + 1e-10), axis=1))
        expected_Q_entropy = np.mean(-np.sum(Q * np.log(Q + 1e-10), axis=1))

        assert np.isclose(
            analysis["distribution_impact"]["original_entropy"], expected_P_entropy
        ), "Original entropy calculation incorrect"

        assert np.isclose(
            analysis["distribution_impact"]["calibrated_entropy"], expected_Q_entropy
        ), "Calibrated entropy calculation incorrect"

    def test_entropy_edge_cases(self):
        """Test entropy calculations on edge cases."""
        # Case with zeros (should handle gracefully)
        P_with_zeros = np.array([[1.0, 0.0], [0.0, 1.0]])
        Q_with_zeros = np.array([[0.9, 0.1], [0.1, 0.9]])
        M = Q_with_zeros.sum(axis=0)

        result = SimpleNamespace(
            Q=Q_with_zeros,
            converged=True,
            iterations=0,
            final_change=0.0,
            max_rank_violation=0.0,
        )

        analysis = analyze_calibration_result(P_with_zeros, result, M)

        # Should not produce NaN or infinite values
        assert np.isfinite(analysis["distribution_impact"]["original_entropy"])
        assert np.isfinite(analysis["distribution_impact"]["calibrated_entropy"])

    def test_entropy_comparison_properties(self):
        """Test that entropy comparisons have expected properties."""
        # Uniform distribution should have high entropy
        P_uniform = np.full((3, 4), 0.25)

        # Concentrated distribution should have low entropy
        P_concentrated = np.array(
            [[0.9, 0.05, 0.05, 0.0], [0.05, 0.9, 0.05, 0.0], [0.05, 0.05, 0.9, 0.0]]
        )

        M = np.array([1.0, 1.0, 1.0, 0.0])  # Adjusted for concentrated case

        result_uniform = SimpleNamespace(
            Q=P_uniform,
            converged=True,
            iterations=0,
            final_change=0.0,
            max_rank_violation=0.0,
        )

        result_concentrated = SimpleNamespace(
            Q=P_concentrated,
            converged=True,
            iterations=0,
            final_change=0.0,
            max_rank_violation=0.0,
        )

        analysis_uniform = analyze_calibration_result(P_uniform, result_uniform, M)
        analysis_concentrated = analyze_calibration_result(
            P_concentrated, result_concentrated, M
        )

        # Uniform should have higher entropy than concentrated
        entropy_uniform = analysis_uniform["distribution_impact"]["original_entropy"]
        entropy_concentrated = analysis_concentrated["distribution_impact"][
            "original_entropy"
        ]

        assert (
            entropy_uniform > entropy_concentrated
        ), f"Uniform entropy ({entropy_uniform}) should be higher than concentrated ({entropy_concentrated})"


class TestAnalysisUtilities:
    """Test result analysis utilities."""

    def test_analyze_calibration_result_structure(self):
        """Test that analysis result has expected structure."""
        P = np.array([[0.4, 0.6], [0.3, 0.7], [0.6, 0.4]])
        Q = np.array([[0.5, 0.5], [0.4, 0.6], [0.5, 0.5]])
        M = Q.sum(axis=0)

        result = SimpleNamespace(
            Q=Q,
            converged=True,
            iterations=5,
            final_change=1e-8,
            max_rank_violation=0.01,
        )

        analysis = analyze_calibration_result(P, result, M)

        # Check expected top-level keys
        expected_keys = [
            "convergence",
            "distribution_impact",
            "constraint_satisfaction",
        ]
        for key in expected_keys:
            assert key in analysis, f"Missing key: {key}"

        # Check convergence subkeys
        perf = analysis["convergence"]
        assert "converged" in perf
        assert "iterations" in perf
        assert "final_change" in perf

        # Check distribution impact subkeys
        impact = analysis["distribution_impact"]
        assert "original_entropy" in impact
        assert "calibrated_entropy" in impact
        assert "total_change" in impact

    def test_analysis_with_different_convergence_states(self):
        """Test analysis with different convergence outcomes."""
        P = np.array([[0.4, 0.6], [0.3, 0.7]])
        Q = np.array([[0.45, 0.55], [0.35, 0.65]])
        M = Q.sum(axis=0)

        # Converged case
        result_converged = SimpleNamespace(
            Q=Q,
            converged=True,
            iterations=10,
            final_change=1e-10,
            max_rank_violation=0.0,
        )

        analysis_conv = analyze_calibration_result(P, result_converged, M)

        # Should produce valid analysis for converged result
        assert analysis_conv["convergence"]["converged"]

        # Should have finite entropy values
        assert np.isfinite(analysis_conv["distribution_impact"]["original_entropy"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
