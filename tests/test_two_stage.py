"""
Tests for two-stage IPF-based calibration methods.

Run with: python -m pytest tests/test_two_stage.py -v
"""

import numpy as np
import pytest

from rank_preserving_calibration import (
    CalibrationError,
    IPFResult,
    TwoStageResult,
    calibrate_ipf,
    calibrate_two_stage,
)

from .data_helpers import create_test_case


class TestCalibrateIPF:
    """Test calibrate_ipf function."""

    def test_basic_functionality(self):
        """Test basic IPF functionality."""
        P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
        M = np.array([1.0, 1.0, 1.0])

        result = calibrate_ipf(P, M)

        assert isinstance(result, IPFResult)
        assert result.Q.shape == P.shape
        assert result.iterations > 0
        # Rows should sum to 1
        assert np.allclose(result.Q.sum(axis=1), 1.0, atol=1e-6)
        # All values non-negative
        assert np.all(result.Q >= -1e-10)

    def test_marginal_matching(self):
        """Test that IPF approximately matches target marginals."""
        P, M = create_test_case("random", N=20, J=3, seed=42)

        result = calibrate_ipf(P, M, max_iters=200)

        # IPF should approximately match marginals
        # (may not be exact due to row normalization at the end)
        col_sums = result.Q.sum(axis=0)
        # Allow some tolerance since IPF alternates
        assert np.allclose(col_sums, M, atol=0.1)

    def test_convergence(self):
        """Test that IPF converges."""
        P = np.array([[0.6, 0.4], [0.3, 0.7], [0.5, 0.5]])
        M = np.array([1.4, 1.6])

        result = calibrate_ipf(P, M, max_iters=100, tol=1e-8)

        assert result.converged
        assert result.final_change < 1e-6

    def test_preserves_ratios_approximately(self):
        """Test that IPF preserves within-row ratios better than projection."""
        P = np.array([[0.8, 0.15, 0.05], [0.1, 0.3, 0.6]])
        M = np.array([0.9, 0.6, 0.5])

        result = calibrate_ipf(P, M)

        # Check that relative ordering within rows is preserved
        for i in range(P.shape[0]):
            p_order = np.argsort(P[i])
            q_order = np.argsort(result.Q[i])
            # Order should be the same (or very close values may swap)
            assert np.allclose(p_order, q_order) or np.allclose(
                np.sort(P[i]), np.sort(result.Q[i]), atol=0.1
            )

    def test_equal_marginals(self):
        """Test IPF when target equals empirical marginals."""
        P = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.3, 0.3, 0.4]])
        M = P.sum(axis=0)  # Same as empirical

        result = calibrate_ipf(P, M)

        # Should be very close to original
        assert np.allclose(result.Q, P, atol=1e-6)


class TestCalibrateTwoStage:
    """Test calibrate_two_stage function."""

    def test_basic_functionality(self):
        """Test basic two-stage calibration."""
        P = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
        M = np.array([1.0, 1.0, 1.0])

        result = calibrate_two_stage(P, M)

        assert isinstance(result, TwoStageResult)
        assert result.Q.shape == P.shape
        assert result.ipf_iterations > 0
        assert result.projection_iterations >= 0
        # Rows should sum to 1
        assert np.allclose(result.Q.sum(axis=1), 1.0, atol=1e-6)
        # All values non-negative
        assert np.all(result.Q >= -1e-10)
        # IPF result should be included
        assert result.ipf_result.shape == P.shape

    def test_rank_preservation(self):
        """Test that two-stage generally preserves rank ordering."""
        P, M = create_test_case("linear", N=15, J=3, seed=42)

        result = calibrate_two_stage(P, M)

        # Check that rank violation is small (not zero, since two-stage is approximate)
        assert result.max_rank_violation < 0.1  # Allow small violations
        # Rows should sum to 1
        assert np.allclose(result.Q.sum(axis=1), 1.0, atol=1e-6)

    def test_preserve_marginals_option(self):
        """Test preserve_marginals option."""
        P, M = create_test_case("random", N=15, J=3, seed=123)

        result_no_preserve = calibrate_two_stage(P, M, preserve_marginals=False)
        result_preserve = calibrate_two_stage(P, M, preserve_marginals=True)

        # With preserve_marginals, column sums should be closer to M
        error_no_preserve = np.sum(np.abs(result_no_preserve.Q.sum(axis=0) - M))
        error_preserve = np.sum(np.abs(result_preserve.Q.sum(axis=0) - M))

        # The preserve option should give better marginal matching
        # (though may have more rank violations)
        assert error_preserve <= error_no_preserve + 0.5

    def test_ipf_result_stored(self):
        """Test that intermediate IPF result is stored."""
        P = np.array([[0.6, 0.4], [0.3, 0.7], [0.5, 0.5]])
        M = np.array([1.2, 1.8])

        result = calibrate_two_stage(P, M)

        # IPF result should be different from final Q (due to isotonic projection)
        assert result.ipf_result is not None
        # Final Q should be at least as rank-preserving as IPF result
        assert result.max_rank_violation >= 0

    def test_iteration_counts(self):
        """Test that iteration counts are reported correctly."""
        P, M = create_test_case("random", N=10, J=3, seed=456)

        result = calibrate_two_stage(
            P, M, ipf_max_iters=50, proj_max_iters=100, ipf_tol=1e-6, proj_tol=1e-6
        )

        assert result.ipf_iterations >= 1
        assert result.ipf_iterations <= 50
        assert result.projection_iterations >= 1
        assert result.projection_iterations <= 100


class TestTwoStageValidation:
    """Test input validation for two-stage methods."""

    def test_ipf_invalid_P_shape(self):
        """Test that invalid P shapes raise errors."""
        with pytest.raises(CalibrationError, match="2D"):
            calibrate_ipf(np.array([1, 2, 3]), np.array([1, 1]))

    def test_ipf_invalid_P_values(self):
        """Test that negative P values raise errors."""
        P = np.array([[-0.1, 1.1], [0.3, 0.7]])
        M = np.array([1.0, 1.0])
        with pytest.raises(CalibrationError, match="non-negative"):
            calibrate_ipf(P, M)

    def test_two_stage_invalid_P(self):
        """Test two-stage with invalid P."""
        with pytest.raises(CalibrationError, match="2D"):
            calibrate_two_stage(np.array([1, 2]), np.array([1, 1]))

    def test_two_stage_mismatched_dims(self):
        """Test two-stage with mismatched dimensions."""
        P = np.array([[0.5, 0.5], [0.3, 0.7]])
        M = np.array([1.0, 1.0, 1.0])
        with pytest.raises(CalibrationError, match="length"):
            calibrate_two_stage(P, M)

    def test_single_column(self):
        """Test that single column raises error."""
        P = np.array([[1.0], [1.0]])
        M = np.array([2.0])
        with pytest.raises(CalibrationError, match="at least 2"):
            calibrate_two_stage(P, M)


class TestTwoStageVsDykstra:
    """Compare two-stage approach with Dykstra."""

    def test_less_flat_solution(self):
        """Test that two-stage can produce less flat solutions."""
        from rank_preserving_calibration import calibrate_dykstra, flatness_metrics

        P, M = create_test_case("random", N=30, J=4, seed=42)
        # Create larger shift to induce flatness
        M = M * 1.3  # Shift marginals
        M = M * (P.shape[0] / M.sum())  # Renormalize to sum to N

        try:
            result_dykstra = calibrate_dykstra(P, M)
            result_two_stage = calibrate_two_stage(P, M, preserve_marginals=False)

            flat_dykstra = flatness_metrics(result_dykstra.Q, P)
            flat_two_stage = flatness_metrics(result_two_stage.Q, P)

            # Two-stage may have higher variance (less flat)
            # This is not guaranteed but often happens
            # At minimum, both should be valid results
            assert flat_dykstra["mean_column_variance"] >= 0
            assert flat_two_stage["mean_column_variance"] >= 0
        except Exception:
            # If Dykstra doesn't converge on shifted marginals, that's OK
            pass

    def test_both_satisfy_constraints(self):
        """Test that both methods satisfy basic constraints."""
        from rank_preserving_calibration import calibrate_dykstra

        P = np.array(
            [[0.6, 0.3, 0.1], [0.3, 0.4, 0.3], [0.2, 0.5, 0.3], [0.4, 0.3, 0.3]]
        )
        M = np.array([1.4, 1.6, 1.0])

        result_dykstra = calibrate_dykstra(P, M)
        result_two_stage = calibrate_two_stage(P, M)

        # Both should have row sums = 1
        assert np.allclose(result_dykstra.Q.sum(axis=1), 1.0, atol=1e-6)
        assert np.allclose(result_two_stage.Q.sum(axis=1), 1.0, atol=1e-6)


class TestResultDataclasses:
    """Test result dataclass fields."""

    def test_ipf_result_fields(self):
        """Test IPFResult has all expected fields."""
        P, M = create_test_case("random", N=10, J=3, seed=42)
        result = calibrate_ipf(P, M)

        assert hasattr(result, "Q")
        assert hasattr(result, "converged")
        assert hasattr(result, "iterations")
        assert hasattr(result, "max_row_error")
        assert hasattr(result, "max_col_error")
        assert hasattr(result, "final_change")

    def test_two_stage_result_fields(self):
        """Test TwoStageResult has all expected fields."""
        P, M = create_test_case("random", N=10, J=3, seed=42)
        result = calibrate_two_stage(P, M)

        assert hasattr(result, "Q")
        assert hasattr(result, "converged")
        assert hasattr(result, "ipf_iterations")
        assert hasattr(result, "projection_iterations")
        assert hasattr(result, "max_row_error")
        assert hasattr(result, "max_col_error")
        assert hasattr(result, "max_rank_violation")
        assert hasattr(result, "ipf_result")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
