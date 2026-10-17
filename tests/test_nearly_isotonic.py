"""
Nearly isotonic functionality test suite for rank_preserving_calibration package.

Tests epsilon-slack projection and lambda-penalty functionality, integration
with main calibration algorithms, and edge cases specific to nearly isotonic
constraints. Deduplicated from original test_nearly_isotonic.py.

Run with: python -m pytest tests/test_nearly_isotonic_consolidated.py
"""

import numpy as np
import pytest

from rank_preserving_calibration import (
    CalibrationError,
    calibrate_dykstra,
    project_near_isotonic_euclidean,
    prox_near_isotonic,
    prox_near_isotonic_with_sum,
)

from .data_helpers import create_test_case


class TestEpsilonSlackIntegration:
    """Test epsilon-slack nearly isotonic integration with calibration algorithms."""

    def test_dykstra_with_epsilon_slack(self):
        """Test Dykstra algorithm with epsilon-slack constraints."""
        P, M = create_test_case("linear", N=15, J=3, seed=42)

        # Standard isotonic calibration
        result_strict = calibrate_dykstra(P, M, verbose=False)

        # Nearly isotonic with epsilon slack
        nearly_params = {"mode": "epsilon", "eps": 0.05}
        result_nearly = calibrate_dykstra(P, M, nearly=nearly_params, verbose=False)

        # Both should satisfy basic constraints
        for result in [result_strict, result_nearly]:
            assert np.allclose(result.Q.sum(axis=1), 1.0, atol=1e-10)
            assert np.allclose(result.Q.sum(axis=0), M, atol=1e-8)
            assert np.all(result.Q >= -1e-10)

        # Nearly isotonic should be more flexible
        distance_strict = np.linalg.norm(result_strict.Q - P, "fro")
        distance_nearly = np.linalg.norm(result_nearly.Q - P, "fro")

        # Nearly isotonic should achieve equal or better fit
        assert distance_nearly <= distance_strict + 1e-6, (
            f"Nearly isotonic worse fit: {distance_nearly} > {distance_strict}"
        )

    def test_epsilon_convergence_improvement(self):
        """Test that epsilon slack can improve convergence."""
        # Create challenging problem
        P, M = create_test_case("challenging", N=20, J=4, seed=789)

        # Test different epsilon values
        epsilons = [0.0, 0.01, 0.05]
        iterations = []

        for eps in epsilons:
            if eps == 0.0:
                result = calibrate_dykstra(P, M, max_iters=1000, verbose=False)
            else:
                nearly_params = {"mode": "epsilon", "eps": eps}
                result = calibrate_dykstra(
                    P, M, nearly=nearly_params, max_iters=1000, verbose=False
                )

            iterations.append(result.iterations)

        # Generally expect fewer iterations with epsilon slack (though not guaranteed)
        assert all(it < 1000 for it in iterations), (
            "Should converge within iteration limit"
        )

    def test_large_epsilon_flexibility(self):
        """Test that large epsilon provides more flexibility."""
        v = np.array([1.0, 0.2, 0.8, 0.1])
        target_sum = 2.5

        # Small epsilon (nearly strict)
        z_small = project_near_isotonic_euclidean(v, eps=0.01, sum_target=target_sum)

        # Large epsilon (very flexible)
        z_large = project_near_isotonic_euclidean(v, eps=0.5, sum_target=target_sum)

        # Both should satisfy sum constraint
        assert abs(z_small.sum() - target_sum) < 1e-12
        assert abs(z_large.sum() - target_sum) < 1e-12

        # Large epsilon should be closer to original (more flexible)
        dist_small = np.linalg.norm(z_small - v)
        dist_large = np.linalg.norm(z_large - v)

        assert dist_large <= dist_small + 1e-10, (
            f"Large epsilon not more flexible: {dist_large} > {dist_small}"
        )


class TestNearlyIsotonicUtilities:
    """Test nearly isotonic utility functions directly."""

    def test_prox_convergence_to_isotonic(self):
        """Test that large lambda makes prox approach isotonic regression."""
        v = np.array([1.0, 0.3, 0.8, 0.2])

        # Very large lambda should approach isotonic regression
        z_large_lambda = prox_near_isotonic(v, lam=1000.0)

        # Should be much more isotonic than original
        def isotonic_violation(x):
            return np.sum(np.maximum(0, x[:-1] - x[1:]))

        viol_original = isotonic_violation(v)
        viol_prox = isotonic_violation(z_large_lambda)

        assert viol_prox <= viol_original * 0.1, (
            "Large lambda prox should be much more isotonic"
        )

    def test_prox_with_sum_properties(self):
        """Test prox operator with sum constraint maintains properties."""
        v = np.array([0.5, 0.2, 0.4, 0.3])
        lam = 2.0
        target_sum = 1.8

        z = prox_near_isotonic_with_sum(v, lam, target_sum)

        # Should satisfy sum constraint exactly
        assert abs(z.sum() - target_sum) < 1e-12, "Sum constraint violated"

        # Should be finite and reasonable
        assert np.isfinite(z).all(), "Non-finite result"
        assert np.all(np.abs(z) < 100), "Unreasonably large values"


class TestNearlyIsotonicEdgeCases:
    """Test edge cases specific to nearly isotonic functionality."""

    def test_boundary_epsilon_values(self):
        """Test epsilon values at boundaries."""
        v = np.array([0.4, 0.1, 0.3, 0.2])

        # Very small epsilon (nearly strict)
        z_tiny = project_near_isotonic_euclidean(v, eps=1e-10)

        # Moderate epsilon
        z_moderate = project_near_isotonic_euclidean(v, eps=0.1)

        # Both should be valid
        assert np.isfinite(z_tiny).all()
        assert np.isfinite(z_moderate).all()

        # Tiny epsilon should be closer to strict isotonic
        from rank_preserving_calibration._isotonic import (
            _isotonic_regression as _pav_increasing,
        )

        z_isotonic = _pav_increasing(v)

        dist_tiny = np.linalg.norm(z_tiny - z_isotonic)
        dist_moderate = np.linalg.norm(z_moderate - z_isotonic)

        assert dist_tiny <= dist_moderate + 1e-10, (
            "Tiny epsilon should be closer to isotonic"
        )

    def test_extreme_lambda_values(self):
        """Test lambda values at extremes."""
        v = np.array([1.0, 0.3, 0.7, 0.2])

        # Very small lambda (nearly identity)
        z_tiny = prox_near_isotonic(v, lam=1e-10)
        assert np.allclose(z_tiny, v, atol=1e-8), "Tiny lambda should be near identity"

        # Very large lambda (nearly isotonic)
        z_large = prox_near_isotonic(v, lam=1e10)

        # Should be more isotonic than original
        def measure_isotonicity(x):
            return -np.sum(np.maximum(0, x[:-1] - x[1:]))  # Higher is more isotonic

        assert measure_isotonicity(z_large) >= measure_isotonicity(v) - 1e-10

    def test_invalid_parameters(self):
        """Test parameter handling behavior."""
        P = np.array([[0.5, 0.5], [0.3, 0.7]])
        M = np.array([0.8, 1.2])

        # An unknown mode is an error, not a silent fall-back to strict
        # isotonic: that is how {"mode": "lambda"} became a no-op.
        for mode in ("invalid", "lambda"):
            with pytest.raises(CalibrationError, match="nearly supports only"):
                calibrate_dykstra(P, M, nearly={"mode": mode})

        # Missing parameters use defaults
        result_eps_default = calibrate_dykstra(
            P, M, nearly={"mode": "epsilon"}
        )  # Uses default eps
        assert result_eps_default.Q.shape == P.shape

    def test_empty_and_single_element_cases(self):
        """Test empty and single element edge cases."""
        # Single element
        v_single = np.array([1.5])

        z_eps = project_near_isotonic_euclidean(v_single, eps=0.1)
        assert len(z_eps) == 1
        assert z_eps[0] == v_single[0]

        z_prox = prox_near_isotonic(v_single, lam=1.0)
        assert len(z_prox) == 1
        assert np.allclose(z_prox, v_single)

        # Empty arrays should be handled gracefully (tested in main algorithms)
        empty_P = np.empty((0, 2))
        empty_M = np.array([0.0, 0.0])

        from rank_preserving_calibration import CalibrationError

        with pytest.raises(CalibrationError):  # Should raise CalibrationError
            calibrate_dykstra(empty_P, empty_M)


class TestNearlyIsotonicConsistency:
    """Test consistency between different nearly isotonic approaches."""

    def test_convergence_to_strict_limits(self):
        """Test that parameters converge to strict isotonic in limits."""
        v = np.array([1.0, 0.4, 0.8, 0.3])
        target_sum = 2.0

        # Test epsilon -> 0
        epsilons = [0.1, 0.01, 0.001]
        eps_results = []
        for eps in epsilons:
            z = project_near_isotonic_euclidean(v, eps, sum_target=target_sum)
            eps_results.append(z)

        # Should converge as epsilon decreases
        dist_01_to_001 = np.linalg.norm(eps_results[0] - eps_results[2])
        dist_001_to_0001 = np.linalg.norm(eps_results[1] - eps_results[2])

        assert dist_001_to_0001 <= dist_01_to_001 + 1e-10, (
            "Not converging as epsilon decreases"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_weighted_slack_projection_with_a_sum_target_raises():
    # A uniform shift is not the weighted projection: for y = [0, 1],
    # weights [1, 3], sum 2 the optimum is [0.75, 1.25], not [0.5, 1.5].
    with pytest.raises(ValueError, match="uniform weights"):
        project_near_isotonic_euclidean(
            np.array([0.0, 1.0]), 0.0, sum_target=2.0, weights=np.array([1.0, 3.0])
        )
