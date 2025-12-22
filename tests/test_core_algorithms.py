"""
Core algorithm test suite for rank_preserving_calibration package.

Tests basic functionality of Dykstra and ADMM algorithms, input validation,
and core algorithm behavior. This consolidates basic smoke tests and ensures
the fundamental algorithms work correctly.

Run with: python -m pytest tests/test_core_algorithms.py
"""

import numpy as np
import pytest

from rank_preserving_calibration import (
    ADMMResult,
    CalibrationError,
    CalibrationResult,
    calibrate_admm,
    calibrate_dykstra,
)

from .data_helpers import create_test_case


class TestDykstraCore:
    """Test core Dykstra algorithm functionality."""

    def test_simple_2x2_case(self):
        """Test simple 2x2 case that should converge easily."""
        P = np.array([[0.8, 0.2], [0.3, 0.7]])
        M = np.array([1.0, 1.0])

        result = calibrate_dykstra(P, M, verbose=False)

        assert isinstance(result, CalibrationResult)
        # converged is always True if we get a result (failure raises CalibrationError)
        assert result.Q.shape == P.shape
        assert result.max_row_error < 1e-6
        assert result.max_col_error < 1e-6
        assert np.allclose(result.Q.sum(axis=1), 1.0)
        assert np.allclose(result.Q.sum(axis=0), M)

    def test_random_case(self):
        """Test on random generated data."""
        P, M = create_test_case("random", N=20, J=3, seed=42)

        result = calibrate_dykstra(P, M, verbose=False)

        # converged is always True if we get a result (failure raises CalibrationError)
        assert result.max_row_error < 1e-6
        assert result.max_col_error < 1e-6
        assert result.max_rank_violation < 1e-6

    def test_rank_preservation(self):
        """Test that rank ordering is preserved."""
        P, M = create_test_case("linear", N=10, J=3, seed=42)

        result = calibrate_dykstra(P, M, verbose=False)

        # Check rank preservation in each column
        for j in range(P.shape[1]):
            original_order = np.argsort(P[:, j])
            calibrated_sorted = result.Q[original_order, j]

            # Should be non-decreasing
            diffs = np.diff(calibrated_sorted)
            assert np.all(diffs >= -1e-10), f"Rank violation in column {j}"

    def test_probability_constraints(self):
        """Test that probability constraints are satisfied."""
        P, M = create_test_case("skewed", N=15, J=4, seed=123)

        result = calibrate_dykstra(P, M, verbose=False)

        # All probabilities should be non-negative
        assert np.all(result.Q >= -1e-10)

        # All rows should sum to 1
        row_sums = result.Q.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-10)

        # Column sums should match targets
        col_sums = result.Q.sum(axis=0)
        assert np.allclose(col_sums, M, atol=1e-10)

    def test_ties_parameter_basic(self):
        """Test basic ties parameter functionality."""
        P = np.array([[0.3, 0.7], [0.3, 0.7], [0.6, 0.4]])  # First two rows tied
        M = np.array([1.2, 1.8])

        # Both tie modes should work
        result_stable = calibrate_dykstra(P, M, ties="stable", verbose=False)
        result_group = calibrate_dykstra(P, M, ties="group", verbose=False)

        for result in [result_stable, result_group]:
            assert result.max_row_error < 1e-7
            assert result.max_col_error < 1e-8


class TestADMMCore:
    """Test core ADMM algorithm functionality."""

    def test_admm_basic(self):
        """Test basic ADMM functionality."""
        P = np.array([[0.7, 0.3], [0.4, 0.6], [0.6, 0.4]])
        M = np.array([1.5, 1.5])

        result = calibrate_admm(P, M, verbose=False)

        assert isinstance(result, ADMMResult)
        assert result.Q.shape == P.shape
        assert len(result.objective_values) > 0
        assert len(result.primal_residuals) > 0
        assert len(result.dual_residuals) > 0
        assert result.max_row_error < 1e-8
        assert result.max_col_error < 1e-8

    def test_admm_vs_dykstra_consistency(self):
        """Test that ADMM and Dykstra give similar results on simple cases."""
        P = np.array([[0.6, 0.4], [0.3, 0.7], [0.5, 0.5]])
        M = np.array([1.2, 1.8])

        result_dykstra = calibrate_dykstra(P, M, verbose=False)
        result_admm = calibrate_admm(P, M, verbose=False)

        # Results should be close (not identical due to different algorithms)
        distance = np.linalg.norm(result_dykstra.Q - result_admm.Q, "fro")
        assert distance < 0.1, (
            f"Dykstra and ADMM results too different: distance = {distance}"
        )


class TestInputValidation:
    """Test input validation and error handling."""

    def test_invalid_P_shape(self):
        """Test that invalid P shapes raise errors."""
        with pytest.raises(CalibrationError, match="2D array"):
            calibrate_dykstra(np.array([1, 2, 3]), np.array([1, 1]))

    def test_invalid_P_values(self):
        """Test that negative P values raise errors."""
        P = np.array([[-0.1, 0.6], [0.3, 0.7]])
        M = np.array([1.0, 1.0])
        with pytest.raises(CalibrationError, match="non-negative"):
            calibrate_dykstra(P, M)

    def test_mismatched_dimensions(self):
        """Test that mismatched P and M dimensions raise errors."""
        P = np.array([[0.5, 0.5], [0.3, 0.7]])  # 2 classes
        M = np.array([1.0, 1.0, 1.0])  # 3 classes
        with pytest.raises(CalibrationError, match="length"):
            calibrate_dykstra(P, M)

    def test_empty_inputs(self):
        """Test behavior with empty inputs."""
        with pytest.raises(CalibrationError):
            calibrate_dykstra(np.array([]).reshape(0, 2), np.array([1.0, 1.0]))

    def test_invalid_ties_parameter(self):
        """Test that invalid ties parameters are rejected."""
        P = np.array([[0.5, 0.5], [0.3, 0.7]])
        M = np.array([0.8, 1.2])

        with pytest.raises(ValueError, match="ties must be 'stable' or 'group'"):
            calibrate_dykstra(P, M, ties="invalid")


class TestAlgorithmProperties:
    """Test fundamental algorithmic properties."""

    def test_deterministic_behavior(self):
        """Test that algorithms are deterministic."""
        P, M = create_test_case("random", N=10, J=3, seed=42)

        # Run multiple times
        results = []
        for _ in range(3):
            result = calibrate_dykstra(P, M, verbose=False)
            results.append(result.Q)

        # All results should be identical
        for i in range(1, len(results)):
            assert np.allclose(results[0], results[i], atol=1e-14), (
                f"Non-deterministic behavior: run 0 vs {i}"
            )

    def test_constraint_satisfaction(self):
        """Test that all constraints are satisfied to machine precision."""
        P, M = create_test_case("challenging", N=12, J=4, seed=789)

        result = calibrate_dykstra(P, M, verbose=False)

        # Row constraints (simplex)
        row_sums = result.Q.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), "Row simplex constraint violated"
        assert np.all(result.Q >= -1e-7), "Non-negativity constraint violated"

        # Column constraints
        col_sums = result.Q.sum(axis=0)
        assert np.allclose(col_sums, M, atol=1e-10), "Column sum constraint violated"

        # Isotonic constraints
        for j in range(P.shape[1]):
            order = np.argsort(P[:, j])
            calibrated_sorted = result.Q[order, j]
            diffs = np.diff(calibrated_sorted)
            assert np.all(diffs >= -1e-12), (
                f"Isotonic constraint violated in column {j}"
            )


class TestDataGeneration:
    """Test synthetic data generation utilities."""

    def test_random_case_generation(self):
        """Test random test case generation."""
        P, M = create_test_case("random", N=10, J=3, seed=42)

        assert P.shape == (10, 3)
        assert M.shape == (3,)
        assert np.allclose(P.sum(axis=1), 1.0)  # Rows sum to 1
        assert np.all(P >= 0)  # Non-negative
        assert np.all(M > 0)  # Positive marginals

    def test_linear_case_generation(self):
        """Test linear trend test case generation."""
        P, M = create_test_case("linear", N=15, J=3, seed=456, noise_level=0.05)

        assert P.shape == (15, 3)
        assert M.shape == (3,)
        assert np.allclose(P.sum(axis=1), 1.0)

    def test_invalid_case_type(self):
        """Test that invalid case types raise errors."""
        with pytest.raises(ValueError, match="Unknown case type"):
            create_test_case("invalid_type", N=10, J=3)

    def test_reproducible_generation(self):
        """Test that data generation is reproducible with same seed."""
        P1, M1 = create_test_case("random", N=8, J=3, seed=123)
        P2, M2 = create_test_case("random", N=8, J=3, seed=123)

        assert np.allclose(P1, P2), "Data generation not reproducible"
        assert np.allclose(M1, M2), "Marginal generation not reproducible"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
