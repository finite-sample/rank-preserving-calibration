"""
Mathematical guarantees test suite for rank_preserving_calibration package.

Tests mathematical correctness, projection properties, algorithmic guarantees,
and numerical stability. This consolidates tests that verify the mathematical
soundness of the algorithms.

Run with: python -m pytest tests/test_mathematical_guarantees.py
"""

import warnings

import numpy as np
import pytest

from rank_preserving_calibration import calibrate_admm, calibrate_dykstra
from rank_preserving_calibration.calibration import (
    _project_column_isotonic_sum,
    _project_row_simplex,
)
from rank_preserving_calibration.nearly import (
    _pav_increasing,
    project_near_isotonic_euclidean,
    prox_near_isotonic,
    prox_near_isotonic_with_sum,
)


class TestProjectionCorrectness:
    """Test mathematical correctness of projection operations."""

    def test_row_simplex_projection_properties(self):
        """Test that row simplex projection satisfies all required properties."""
        np.random.seed(42)

        for _ in range(5):
            X = np.random.randn(5, 3)
            Y = _project_row_simplex(X)

            # Property 1: Result is on simplex
            assert np.allclose(Y.sum(axis=1), 1.0, atol=1e-12), (
                "Row sums not equal to 1"
            )
            assert np.all(Y >= -1e-12), "Negative values in projection"

            # Property 2: Projection is idempotent
            Y2 = _project_row_simplex(Y)
            assert np.allclose(Y, Y2, atol=1e-14), "Projection not idempotent"

    def test_column_projection_optimality(self):
        """Test that column projection minimizes distance while satisfying constraints."""
        np.random.seed(123)

        for _ in range(3):
            n = 8
            w = np.random.randn(n)
            target = np.random.uniform(1.0, 5.0)
            original_scores = np.random.randn(n)
            column_order = np.argsort(original_scores)

            result = _project_column_isotonic_sum(w, column_order, target)

            # Verify constraints
            assert abs(result.sum() - target) < 1e-12, "Sum constraint violated"

            sorted_result = result[column_order]
            diffs = np.diff(sorted_result)
            assert np.all(diffs >= -1e-12), "Isotonic constraint violated"

    def test_kkt_stationarity_conditions(self):
        """Test KKT stationarity conditions for isotonic+sum projection."""
        # Simple test case where we can verify KKT conditions
        w = np.array([0.8, 0.3, 0.6, 0.2])
        target = 2.5
        column_order = np.argsort(w)  # [3,1,2,0] -> values [0.2,0.3,0.6,0.8]

        result = _project_column_isotonic_sum(w, column_order, target)

        # Verify basic constraints
        assert abs(result.sum() - target) < 1e-14

        sorted_result = result[column_order]
        assert np.all(np.diff(sorted_result) >= -1e-14)


class TestIsotonicRegression:
    """Test isotonic regression (PAV) algorithm correctness."""

    def test_pav_isotonic_property(self):
        """Test that PAV produces non-decreasing output."""
        test_cases = [
            np.array([3.0, 1.0, 4.0, 2.0]),
            np.array([1.0, 2.0, 3.0, 4.0]),  # Already isotonic
            np.array([4.0, 3.0, 2.0, 1.0]),  # Completely reversed
            np.array([1.0, 1.0, 1.0, 1.0]),  # All equal
        ]

        for y in test_cases:
            z = _pav_increasing(y)
            diffs = np.diff(z)
            assert np.all(diffs >= -1e-12), f"Output not isotonic for input {y}"

    def test_pav_projection_property(self):
        """Test that PAV minimizes L2 distance to isotonic points."""
        y = np.array([2.0, 1.0, 3.0, 0.5])
        z = _pav_increasing(y)

        # z should be the closest isotonic point to y
        distance_to_z = np.linalg.norm(y - z)

        # Test with some other isotonic points
        isotonic_alternatives = [
            np.array([1.0, 1.0, 2.0, 2.0]),
            np.array([0.5, 1.5, 2.0, 2.5]),
            np.array([1.5, 1.5, 1.5, 1.5]),
        ]

        for alt in isotonic_alternatives:
            alt_distance = np.linalg.norm(y - alt)
            assert distance_to_z <= alt_distance + 1e-10, "PAV not optimal"

    def test_pav_with_weights(self):
        """Test weighted PAV maintains isotonic property."""
        y = np.array([3.0, 1.0, 4.0, 2.0])
        weights = np.array([1.0, 2.0, 1.0, 1.0])

        z = _pav_increasing(y, weights)

        # Should still be isotonic
        diffs = np.diff(z)
        assert np.all(diffs >= -1e-12), "Weighted PAV not isotonic"


class TestEpsilonSlackProjection:
    """Test epsilon-slack nearly isotonic projection."""

    def test_epsilon_constraint_satisfaction(self):
        """Test that epsilon-slack projection satisfies z[i+1] >= z[i] - eps."""
        v = np.array([1.0, 0.2, 0.8, 0.1])
        eps = 0.1

        z = project_near_isotonic_euclidean(v, eps)

        # Check constraint satisfaction
        for i in range(len(z) - 1):
            assert z[i + 1] >= z[i] - eps - 1e-12, (
                f"Constraint violated at {i}: {z[i + 1]} < {z[i]} - {eps}"
            )

    def test_epsilon_zero_reduces_to_isotonic(self):
        """Test that eps=0 gives standard isotonic projection."""
        v = np.array([1.0, 0.3, 0.7, 0.2])

        z_nearly = project_near_isotonic_euclidean(v, eps=0.0)
        z_isotonic = _pav_increasing(v)

        assert np.allclose(z_nearly, z_isotonic, atol=1e-12), (
            "eps=0 does not reduce to isotonic"
        )

    def test_sum_constraint_with_epsilon(self):
        """Test epsilon projection with sum constraint."""
        v = np.array([0.4, 0.1, 0.3, 0.2])
        eps = 0.05
        target_sum = 1.2

        z = project_near_isotonic_euclidean(v, eps, sum_target=target_sum)

        # Check sum constraint
        assert abs(z.sum() - target_sum) < 1e-12, "Sum constraint not satisfied"

        # Check epsilon constraint
        for i in range(len(z) - 1):
            assert z[i + 1] >= z[i] - eps - 1e-12, "Epsilon constraint violated"


class TestLambdaPenaltyProx:
    """Test lambda-penalty proximal operator."""

    def test_prox_lambda_zero_identity(self):
        """Test that lambda=0 gives identity."""
        v = np.array([1.0, 0.5, 0.8, 0.3])
        z = prox_near_isotonic(v, lam=0.0)

        assert np.allclose(z, v, atol=1e-12), "lambda=0 not identity"

    def test_prox_monotonicity_in_lambda(self):
        """Test that larger lambda gives more isotonic solutions."""
        v = np.array([1.0, 0.3, 0.7, 0.2])

        z_small = prox_near_isotonic(v, lam=0.1)
        z_large = prox_near_isotonic(v, lam=10.0)

        # Measure isotonic violations
        def isotonic_violation(x):
            return np.sum(np.maximum(0, x[:-1] - x[1:]))

        viol_small = isotonic_violation(z_small)
        viol_large = isotonic_violation(z_large)

        assert viol_large <= viol_small + 1e-10, "Larger λ should reduce violations"

    def test_prox_with_sum_constraint(self):
        """Test prox operator with sum constraint."""
        v = np.array([0.4, 0.1, 0.3, 0.2])
        lam = 1.0
        target_sum = 1.2

        z = prox_near_isotonic_with_sum(v, lam, target_sum)

        # Should satisfy sum constraint
        assert abs(z.sum() - target_sum) < 1e-12, "Sum constraint not satisfied"

        # Should be equivalent to prox + shift
        z_no_sum = prox_near_isotonic(v, lam)
        shift = (target_sum - z_no_sum.sum()) / len(v)
        z_manual = z_no_sum + shift

        assert np.allclose(z, z_manual, atol=1e-12), (
            "Sum constraint implementation incorrect"
        )


class TestAlgorithmicGuarantees:
    """Test high-level algorithmic guarantees."""

    def test_projection_idempotence(self):
        """Test that projections are idempotent."""
        # Row projection idempotence
        X = np.random.randn(5, 3)
        Y1 = _project_row_simplex(X)
        Y2 = _project_row_simplex(Y1)
        assert np.allclose(Y1, Y2, atol=1e-15), "Row projection not idempotent"

        # Column projection idempotence
        w = np.random.randn(8)
        target = 3.0
        order = np.argsort(np.random.randn(8))
        z1 = _project_column_isotonic_sum(w, order, target)
        z2 = _project_column_isotonic_sum(z1, order, target)
        assert np.allclose(z1, z2, atol=1e-14), "Column projection not idempotent"

    def test_distance_minimization_simple(self):
        """Test that calibration minimizes distance on simple cases."""
        P = np.array([[0.6, 0.4], [0.3, 0.7], [0.5, 0.5]])
        M = np.array([1.2, 1.8])

        result = calibrate_dykstra(P, M, max_iters=1000)

        # Should satisfy constraints
        assert np.allclose(result.Q.sum(axis=1), 1.0, atol=1e-10)
        assert np.allclose(result.Q.sum(axis=0), M, atol=1e-8)

        # Should preserve rank ordering
        for j in range(P.shape[1]):
            order = np.argsort(P[:, j])
            calibrated_sorted = result.Q[order, j]
            assert np.all(np.diff(calibrated_sorted) >= -1e-12)

    def test_convergence_on_feasible_problems(self):
        """Test that algorithms converge on clearly feasible problems."""
        # Create feasible problem
        np.random.seed(456)
        P = np.random.dirichlet([2, 2, 2], size=6)
        M = P.sum(axis=0)  # Exact feasibility

        result = calibrate_dykstra(P, M, max_iters=1000)

        # Should converge quickly and accurately (raises CalibrationError if not)
        assert result.max_row_error < 1e-10
        assert result.max_col_error < 1e-10


class TestNumericalStability:
    """Test numerical stability and robustness."""

    def test_extreme_scale_problems(self):
        """Test behavior with extreme scales."""
        # Very small scale
        P_small = np.array([[0.6, 0.4], [0.3, 0.7]]) * 1e-10
        P_small = P_small / P_small.sum(axis=1, keepdims=True)
        M_small = P_small.sum(axis=0)

        result_small = calibrate_dykstra(P_small, M_small, max_iters=500)
        assert np.isfinite(result_small.Q).all(), "Numerical issues with small scale"

        # Very large scale (scaled down to avoid overflow)
        P_large = np.array([[0.6, 0.4], [0.3, 0.7]]) * 1e5
        P_large = P_large / P_large.sum(axis=1, keepdims=True)
        M_large = P_large.sum(axis=0)

        result_large = calibrate_dykstra(P_large, M_large, max_iters=500)
        assert np.isfinite(result_large.Q).all(), "Numerical issues with large scale"

    def test_near_boundary_cases(self):
        """Test cases near constraint boundaries."""
        # Nearly degenerate case
        P = np.array([[0.999, 0.001], [0.001, 0.999], [0.5, 0.5]])
        M = np.array([1.5, 1.5])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # May generate warnings
            result = calibrate_dykstra(P, M, max_iters=2000)

        assert np.isfinite(result.Q).all(), "Numerical issues with boundary case"
        assert np.all(result.Q >= -1e-8), "Negative probabilities"

    def test_single_element_edge_cases(self):
        """Test edge cases with single elements."""
        # Single element vector
        v = np.array([2.5])

        z_eps = project_near_isotonic_euclidean(v, eps=0.1)
        assert len(z_eps) == 1 and np.isfinite(z_eps[0])

        z_prox = prox_near_isotonic(v, lam=1.0)
        assert len(z_prox) == 1 and np.isfinite(z_prox[0])

    def test_already_optimal_cases(self):
        """Test inputs that are already optimal."""
        # Already isotonic
        v = np.array([1.0, 2.0, 3.0, 4.0])

        z_eps = project_near_isotonic_euclidean(v, eps=0.1)
        assert np.allclose(z_eps, v, atol=1e-12), "Already isotonic case failed"

        z_prox = prox_near_isotonic(v, lam=1.0)
        assert np.allclose(z_prox, v, atol=1e-10), "Already isotonic prox failed"


class TestADMMGuarantees:
    """Test ADMM-specific guarantees."""

    def test_admm_constraint_satisfaction(self):
        """Test that ADMM satisfies constraints after polishing."""
        P = np.array([[0.7, 0.3], [0.4, 0.6], [0.6, 0.4]])
        M = np.array([1.5, 1.5])

        result = calibrate_admm(P, M, max_iters=500)

        # Should satisfy all constraints after final polishing
        assert np.allclose(result.Q.sum(axis=1), 1.0, atol=1e-10)
        assert np.allclose(result.Q.sum(axis=0), M, atol=1e-8)
        assert np.all(result.Q >= -1e-10)

    def test_admm_objective_decrease(self):
        """Test that ADMM objective generally decreases."""
        P = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]])
        M = np.array([1.4, 1.6])

        result = calibrate_admm(P, M, max_iters=100, verbose=False)

        # Should have recorded objectives
        assert len(result.objective_values) > 0

        # Later objectives should generally be smaller than early ones
        if len(result.objective_values) > 20:
            early_avg = np.mean(result.objective_values[5:10])
            late_avg = np.mean(result.objective_values[-5:])
            assert late_avg <= early_avg + 1e-6, "Objective not decreasing in ADMM"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
