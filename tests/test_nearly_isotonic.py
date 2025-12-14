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
    calibrate_admm,
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


class TestLambdaPenaltyIntegration:
    """Test lambda-penalty nearly isotonic integration with ADMM."""

    def test_admm_with_lambda_penalty(self):
        """Test ADMM with lambda-penalty nearly isotonic."""
        P, M = create_test_case("linear", N=12, J=3, seed=456)

        # Standard ADMM
        result_strict = calibrate_admm(P, M, max_iters=500, verbose=False)

        # Lambda-penalty ADMM
        nearly_params = {"mode": "lambda", "lam": 1.0}
        result_penalty = calibrate_admm(
            P, M, nearly=nearly_params, max_iters=500, verbose=False
        )

        # Both should satisfy constraints
        for result in [result_strict, result_penalty]:
            assert np.allclose(result.Q.sum(axis=1), 1.0, atol=1e-10)
            assert np.allclose(result.Q.sum(axis=0), M, atol=1e-8)

        # Both should be reasonable solutions
        assert np.isfinite(result_strict.Q).all()
        assert np.isfinite(result_penalty.Q).all()

    def test_lambda_penalty_trade_off(self):
        """Test lambda penalty creates trade-off between fit and isotonicity."""
        P = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.4, 0.6]])
        M = np.array([2.0, 2.0])

        lambdas = [0.1, 1.0, 10.0]
        distances = []
        violations = []

        for lam in lambdas:
            nearly_params = {"mode": "lambda", "lam": lam}
            result = calibrate_admm(
                P, M, nearly=nearly_params, max_iters=300, verbose=False
            )

            # Measure distance to original
            distance = np.linalg.norm(result.Q - P, "fro")
            distances.append(distance)

            # Measure isotonic violations
            total_violation = 0.0
            for j in range(P.shape[1]):
                order = np.argsort(P[:, j])
                sorted_vals = result.Q[order, j]
                diffs = np.diff(sorted_vals)
                violation = np.sum(np.maximum(0, -diffs))
                total_violation += violation
            violations.append(total_violation)

        # Generally expect trade-off: larger lambda -> smaller violations
        # (though exact monotonicity not guaranteed due to optimization)
        assert violations[-1] <= violations[0] + 1e-6, (
            "Large lambda should reduce violations"
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
        from rank_preserving_calibration.nearly import _pav_increasing

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

        # Invalid mode is ignored (falls back to standard isotonic)
        result_invalid = calibrate_dykstra(P, M, nearly={"mode": "invalid"})
        assert result_invalid.Q.shape == P.shape

        # Missing parameters use defaults
        result_eps_default = calibrate_dykstra(
            P, M, nearly={"mode": "epsilon"}
        )  # Uses default eps
        assert result_eps_default.Q.shape == P.shape

        result_lam_default = calibrate_admm(
            P, M, nearly={"mode": "lambda"}
        )  # Uses default lam
        assert result_lam_default.Q.shape == P.shape

    def test_empty_and_single_element_cases(self):
        """Test empty and single element edge cases."""
        # Single element
        v_single = np.array([1.5])

        z_eps = project_near_isotonic_euclidean(v_single, eps=0.1)
        assert len(z_eps) == 1 and z_eps[0] == v_single[0]

        z_prox = prox_near_isotonic(v_single, lam=1.0)
        assert len(z_prox) == 1 and np.allclose(z_prox, v_single)

        # Empty arrays should be handled gracefully (tested in main algorithms)
        empty_P = np.empty((0, 2))
        empty_M = np.array([0.0, 0.0])

        from rank_preserving_calibration import CalibrationError

        with pytest.raises(CalibrationError):  # Should raise CalibrationError
            calibrate_dykstra(empty_P, empty_M)


class TestNearlyIsotonicConsistency:
    """Test consistency between different nearly isotonic approaches."""

    def test_epsilon_lambda_consistency(self):
        """Test that epsilon and lambda approaches are roughly consistent."""
        P = np.array([[0.6, 0.4], [0.3, 0.7], [0.5, 0.5]])
        M = np.array([1.2, 1.8])

        # Epsilon-slack Dykstra
        result_eps = calibrate_dykstra(
            P, M, nearly={"mode": "epsilon", "eps": 0.05}, max_iters=1000, verbose=False
        )

        # Lambda-penalty ADMM
        result_lam = calibrate_admm(
            P, M, nearly={"mode": "lambda", "lam": 2.0}, max_iters=500, verbose=False
        )

        # Both should satisfy constraints
        for result in [result_eps, result_lam]:
            assert np.allclose(result.Q.sum(axis=1), 1.0, atol=1e-8)
            assert np.allclose(result.Q.sum(axis=0), M, atol=1e-6)

        # Results should be in similar ballpark (though not identical)
        distance_between = np.linalg.norm(result_eps.Q - result_lam.Q, "fro")
        max_reasonable_distance = np.linalg.norm(
            P, "fro"
        )  # Use input scale as reference

        assert distance_between < max_reasonable_distance, (
            f"Epsilon and lambda results too different: {distance_between}"
        )

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
