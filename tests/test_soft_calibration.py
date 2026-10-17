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
        # Rows should sum to 1 (hard constraint)
        assert np.allclose(result.Q.sum(axis=1), 1.0, atol=1e-6)
        # All values non-negative
        assert np.all(result.Q >= -1e-10)

    def test_marginal_penalty_effect(self):
        """Test that marginal penalty term is included in optimization."""
        P = np.array([[0.6, 0.3, 0.1], [0.3, 0.5, 0.2], [0.2, 0.4, 0.4]])
        M = np.array([1.1, 1.2, 0.7])

        result = calibrate_soft(P, M, lam_m=1.0, lam_r=1.0)

        # Marginal term should be computed
        assert result.marginal_term >= 0
        # Rows should sum to 1
        assert np.allclose(result.Q.sum(axis=1), 1.0, atol=1e-6)

    def test_rank_penalty_effect(self):
        """Test that rank penalty term is computed correctly."""
        P, M = create_test_case("linear", N=15, J=3, seed=123)

        result = calibrate_soft(P, M, lam_m=1.0, lam_r=10.0)

        # Rank term should be non-negative
        assert result.rank_term >= 0
        # Rows should sum to 1
        assert np.allclose(result.Q.sum(axis=1), 1.0, atol=1e-6)

    def test_zero_penalties(self):
        """Test with zero penalties."""
        P = np.array([[0.6, 0.4], [0.3, 0.7]])
        M = np.array([1.0, 1.0])

        # Zero marginal penalty - should stay closer to P
        result = calibrate_soft(P, M, lam_m=0.0, lam_r=0.0)
        assert result.Q.shape == P.shape
        assert np.allclose(result.Q.sum(axis=1), 1.0, atol=1e-6)

    def test_convergence(self):
        """Test that algorithm makes progress."""
        P = np.array([[0.6, 0.4], [0.3, 0.7], [0.5, 0.5]])
        M = np.array([1.4, 1.6])

        result = calibrate_soft(P, M, lam_m=1.0, lam_r=10.0)

        # Algorithm should run and produce valid output
        assert result.iterations > 0
        assert np.allclose(result.Q.sum(axis=1), 1.0, atol=1e-6)


class TestSolverTolerance:
    def test_tighter_tolerance_moves_the_answer_by_little(self):
        """The optimum is unique, so tolerances only change precision."""
        P, M = create_test_case("random", N=12, J=3, seed=4)
        Q1 = calibrate_soft(P, M, lam_m=1.0, lam_r=0.5, tol=1e-8).Q
        Q2 = calibrate_soft(P, M, lam_m=1.0, lam_r=0.5, tol=1e-11).Q
        np.testing.assert_allclose(Q1, Q2, atol=1e-4)


def test_solver_options_are_keyword_only():
    # Solver options are keyword-only, so a positional call from an older
    # signature, such as (P, M, 1.0, 10.0, 1000, 1e-6), fails loudly instead of
    # being misread.
    P = np.array([[0.6, 0.4], [0.3, 0.7]])
    M = np.array([0.9, 1.1])
    with pytest.raises(TypeError):
        calibrate_soft(P, M, 1.0, 10.0, 1000, 1e-6)  # type: ignore[misc]


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
    def test_objective_is_its_terms_recombined(self):
        P, M = create_test_case("random", N=10, J=3, seed=42)
        r = calibrate_soft(P, M, lam_m=2.0, lam_r=0.5)
        assert r.objective == pytest.approx(
            0.5 * r.fit_term + 0.5 * 2.0 * r.marginal_term + 0.5 * r.rank_term
        )
        assert r.fit_term == pytest.approx(float(np.sum((r.Q - P) ** 2)))
        assert r.marginal_term == pytest.approx(
            float(np.sum((r.Q.sum(axis=0) - M) ** 2))
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
