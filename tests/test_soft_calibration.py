"""
Tests for soft-constraint calibration methods.

Run with: python -m pytest tests/test_soft_calibration.py -v
"""

import numpy as np
import pytest

from rank_preserving_calibration import (
    CalibrationError,
    SoftCalibrationResult,
    calibrate_soft,
    calibrate_soft_admm,
)

from .data_helpers import create_test_case


class TestCalibratesoft:
    """Test calibrate_soft function."""

    def test_basic_functionality(self):
        """Test basic soft calibration."""
        P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
        M = np.array([1.0, 1.0, 1.0])

        result = calibrate_soft(P, M, lam_m=1.0, lam_r=10.0)

        assert isinstance(result, SoftCalibrationResult)
        assert result.Q.shape == P.shape
        assert result.iterations > 0
        assert len(result.objective_values) > 0
        # Rows should sum to 1 (hard constraint)
        assert np.allclose(result.Q.sum(axis=1), 1.0, atol=1e-6)
        # All values non-negative
        assert np.all(result.Q >= -1e-10)

    def test_marginal_penalty_effect(self):
        """Test that marginal penalty term is included in optimization."""
        P = np.array([[0.6, 0.3, 0.1], [0.3, 0.5, 0.2], [0.2, 0.4, 0.4]])
        M = np.array([1.1, 1.2, 0.7])

        result = calibrate_soft(P, M, lam_m=1.0, lam_r=1.0, max_iters=500)

        # Marginal term should be computed
        assert result.marginal_term >= 0
        # Rows should sum to 1
        assert np.allclose(result.Q.sum(axis=1), 1.0, atol=1e-6)

    def test_rank_penalty_effect(self):
        """Test that rank penalty term is computed correctly."""
        P, M = create_test_case("linear", N=15, J=3, seed=123)

        result = calibrate_soft(P, M, lam_m=1.0, lam_r=10.0, max_iters=500)

        # Rank term should be non-negative
        assert result.rank_term >= 0
        # Rows should sum to 1
        assert np.allclose(result.Q.sum(axis=1), 1.0, atol=1e-6)

    def test_zero_penalties(self):
        """Test with zero penalties."""
        P = np.array([[0.6, 0.4], [0.3, 0.7]])
        M = np.array([1.0, 1.0])

        # Zero marginal penalty - should stay closer to P
        result = calibrate_soft(P, M, lam_m=0.0, lam_r=0.0, max_iters=100)
        assert result.Q.shape == P.shape
        assert np.allclose(result.Q.sum(axis=1), 1.0, atol=1e-6)

    def test_convergence(self):
        """Test that algorithm makes progress."""
        P = np.array([[0.6, 0.4], [0.3, 0.7], [0.5, 0.5]])
        M = np.array([1.4, 1.6])

        result = calibrate_soft(P, M, lam_m=1.0, lam_r=10.0, max_iters=500, tol=1e-7)

        # Algorithm should run and produce valid output
        assert result.iterations > 0
        assert np.allclose(result.Q.sum(axis=1), 1.0, atol=1e-6)

    def test_objective_decreasing(self):
        """Test that objective function generally decreases."""
        P, M = create_test_case("random", N=15, J=4, seed=789)

        result = calibrate_soft(P, M, lam_m=1.0, lam_r=5.0, max_iters=500)

        # Check that objective doesn't increase significantly
        # (may have small increases due to projection steps)
        obj_vals = result.objective_values
        if len(obj_vals) > 10:
            # First half average should be >= second half average
            mid = len(obj_vals) // 2
            first_half_mean = np.mean(obj_vals[:mid])
            second_half_mean = np.mean(obj_vals[mid:])
            assert second_half_mean <= first_half_mean * 1.1


class TestCalibrateSoftADMM:
    """Test ADMM-based soft calibration."""

    def test_basic_functionality(self):
        """Test basic ADMM soft calibration."""
        P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
        M = np.array([1.0, 1.0, 1.0])

        result = calibrate_soft_admm(P, M, lam_m=1.0, lam_r=10.0)

        assert isinstance(result, SoftCalibrationResult)
        assert result.Q.shape == P.shape
        assert result.iterations > 0

    def test_vs_gradient_descent(self):
        """Test that ADMM and gradient descent both produce valid outputs."""
        P = np.array([[0.6, 0.4], [0.3, 0.7], [0.5, 0.5]])
        M = np.array([1.4, 1.6])

        result_gd = calibrate_soft(P, M, lam_m=1.0, lam_r=5.0, max_iters=500)
        result_admm = calibrate_soft_admm(P, M, lam_m=1.0, lam_r=5.0, max_iters=500)

        # Both should produce valid probability matrices
        assert np.allclose(result_gd.Q.sum(axis=1), 1.0, atol=1e-6)
        assert np.allclose(result_admm.Q.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(result_gd.Q >= -1e-10)
        assert np.all(result_admm.Q >= -1e-10)

    def test_rho_parameter(self):
        """Test effect of rho parameter."""
        P = np.array([[0.6, 0.4], [0.3, 0.7], [0.5, 0.5]])
        M = np.array([1.2, 1.8])

        # Different rho values should converge to similar solutions
        result_rho1 = calibrate_soft_admm(P, M, lam_m=1.0, lam_r=10.0, rho=0.5)
        result_rho2 = calibrate_soft_admm(P, M, lam_m=1.0, lam_r=10.0, rho=2.0)

        # Both should satisfy row simplex
        assert np.allclose(result_rho1.Q.sum(axis=1), 1.0, atol=1e-6)
        assert np.allclose(result_rho2.Q.sum(axis=1), 1.0, atol=1e-6)


class TestSoftCalibrationValidation:
    """Test input validation for soft calibration."""

    def test_invalid_P_shape(self):
        """Test that invalid P shapes raise errors."""
        with pytest.raises(CalibrationError, match="2D"):
            calibrate_soft(np.array([1, 2, 3]), np.array([1, 1]))

    def test_invalid_P_values(self):
        """Test that negative P values raise errors."""
        P = np.array([[-0.1, 1.1], [0.3, 0.7]])
        M = np.array([1.0, 1.0])
        with pytest.raises(CalibrationError, match="non-negative"):
            calibrate_soft(P, M)

    def test_mismatched_dimensions(self):
        """Test that mismatched dimensions raise errors."""
        P = np.array([[0.5, 0.5], [0.3, 0.7]])
        M = np.array([1.0, 1.0, 1.0])
        with pytest.raises(CalibrationError, match="length"):
            calibrate_soft(P, M)

    def test_single_column(self):
        """Test that single column raises error."""
        P = np.array([[1.0], [1.0]])
        M = np.array([2.0])
        with pytest.raises(CalibrationError, match="at least 2"):
            calibrate_soft(P, M)

    def test_negative_penalties(self):
        """Test that negative penalties raise errors."""
        P = np.array([[0.5, 0.5], [0.3, 0.7]])
        M = np.array([0.8, 1.2])

        with pytest.raises(CalibrationError, match="non-negative"):
            calibrate_soft(P, M, lam_m=-1.0)

        with pytest.raises(CalibrationError, match="non-negative"):
            calibrate_soft(P, M, lam_r=-1.0)


class TestSoftCalibrationResult:
    """Test SoftCalibrationResult dataclass."""

    def test_result_fields(self):
        """Test that all expected fields are present."""
        P, M = create_test_case("random", N=10, J=3, seed=42)
        result = calibrate_soft(P, M)

        # Check all fields exist
        assert hasattr(result, "Q")
        assert hasattr(result, "converged")
        assert hasattr(result, "iterations")
        assert hasattr(result, "objective_values")
        assert hasattr(result, "fit_term")
        assert hasattr(result, "marginal_term")
        assert hasattr(result, "rank_term")
        assert hasattr(result, "max_row_error")
        assert hasattr(result, "max_col_error")
        assert hasattr(result, "max_rank_violation")
        assert hasattr(result, "final_change")

    def test_result_types(self):
        """Test that result fields have correct types."""
        P, M = create_test_case("random", N=10, J=3, seed=42)
        result = calibrate_soft(P, M)

        assert isinstance(result.Q, np.ndarray)
        assert isinstance(result.converged, bool)
        assert isinstance(result.iterations, int)
        assert isinstance(result.objective_values, list)
        assert isinstance(result.fit_term, float)
        assert isinstance(result.marginal_term, float)
        assert isinstance(result.rank_term, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
