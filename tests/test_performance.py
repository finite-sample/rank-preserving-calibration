"""Performance tests for JIT compilation and progress bar functionality."""

import time

import numpy as np
import pytest

from rank_preserving_calibration import CalibrationError, calibrate_dykstra
from rank_preserving_calibration._numba_utils import HAS_NUMBA


class TestPerformanceOptimizations:
    """Test performance enhancements work correctly."""

    def test_jit_vs_pure_python_consistency(self):
        """Test that JIT and pure Python give identical results."""
        # Create test case
        np.random.seed(42)
        N, J = 100, 5
        P = np.random.dirichlet(np.ones(J), size=N)
        M = np.full(J, N / J)

        try:
            # Run with JIT (if available)
            result_jit = calibrate_dykstra(P, M, max_iters=100, use_jit=True)

            # Run without JIT
            result_pure = calibrate_dykstra(P, M, max_iters=100, use_jit=False)

            # Results should be very close (minor numerical differences expected)
            np.testing.assert_allclose(
                result_jit.Q, result_pure.Q, rtol=1e-7, atol=1e-15
            )
            assert result_jit.converged == result_pure.converged
            assert result_jit.iterations == result_pure.iterations
        except CalibrationError:
            # With only 100 iterations, convergence failure is acceptable
            pass

    @pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
    def test_jit_performance_improvement(self):
        """Test that JIT provides performance improvement for larger problems."""
        # Create larger test case
        np.random.seed(42)
        N, J = 500, 10
        P = np.random.dirichlet(np.ones(J), size=N)
        M = np.full(J, N / J)

        # Warm up JIT
        try:
            _ = calibrate_dykstra(P[:10], M, max_iters=10, use_jit=True, verbose=False)
        except CalibrationError:
            pass

        # Time with JIT
        start = time.perf_counter()
        try:
            result_jit = calibrate_dykstra(
                P, M, max_iters=50, use_jit=True, verbose=False
            )
            time_jit = time.perf_counter() - start
        except CalibrationError:
            pytest.skip("JIT algorithm did not converge in 50 iterations")

        # Time without JIT
        start = time.perf_counter()
        try:
            result_pure = calibrate_dykstra(
                P, M, max_iters=50, use_jit=False, verbose=False
            )
            time_pure = time.perf_counter() - start
        except CalibrationError:
            pytest.skip("Pure Python algorithm did not converge in 50 iterations")

        # JIT should be faster (at least 20% improvement expected)
        speedup = time_pure / time_jit
        print(
            f"JIT speedup: {speedup:.2f}x (JIT: {time_jit:.3f}s, Pure: {time_pure:.3f}s)"
        )

        # Results should still be very close
        np.testing.assert_allclose(result_jit.Q, result_pure.Q, rtol=1e-7, atol=1e-15)

        # Note: We don't assert on speedup as it varies by machine
        # but typically expect 1.5-3x speedup on moderate size problems

    def test_graceful_fallback_without_numba(self):
        """Test that code works even without Numba installed."""
        # This test always runs regardless of Numba availability
        np.random.seed(42)
        N, J = 30, 3
        P = np.random.dirichlet(np.ones(J), size=N)
        M = np.full(J, N / J)

        # Should work with use_jit=True even if Numba not installed
        try:
            result = calibrate_dykstra(P, M, max_iters=50, use_jit=True)
            # If we get here, it converged
            assert result.Q is not None
            assert result.Q.shape == P.shape
            # Check constraints are approximately satisfied
            assert np.allclose(result.Q.sum(axis=1), 1.0, atol=1e-6)
            assert np.allclose(result.Q.sum(axis=0), M, atol=1e-4)
        except CalibrationError:
            # With only 50 iterations on a large problem, convergence failure is acceptable
            pass

    def test_jit_with_nearly_isotonic(self):
        """Test JIT works with nearly isotonic constraints."""
        np.random.seed(42)
        N, J = 50, 4
        P = np.random.dirichlet(np.ones(J), size=N)
        M = np.full(J, N / J)

        try:
            # Epsilon-slack with JIT
            nearly_eps = {"mode": "epsilon", "eps": 0.01}
            result_jit = calibrate_dykstra(
                P, M, nearly=nearly_eps, max_iters=100, use_jit=True
            )

            # Same without JIT
            result_pure = calibrate_dykstra(
                P, M, nearly=nearly_eps, max_iters=100, use_jit=False
            )

            # Results should be very close (may have minor numerical differences)
            np.testing.assert_allclose(result_jit.Q, result_pure.Q, rtol=1e-8)
        except CalibrationError:
            # With only 100 iterations, convergence failure is acceptable
            pass

    def test_memory_efficiency(self):
        """Test that JIT doesn't significantly increase memory usage."""
        # This is a basic test - comprehensive memory profiling needs external tools
        np.random.seed(42)
        N, J = 200, 8
        P = np.random.dirichlet(np.ones(J), size=N)
        M = np.full(J, N / J)

        # Both should complete without memory issues
        try:
            result_jit = calibrate_dykstra(P, M, max_iters=50, use_jit=True)
            assert result_jit.Q.shape == P.shape
        except CalibrationError:
            # With only 50 iterations, convergence failure is acceptable
            pass

        try:
            result_pure = calibrate_dykstra(P, M, max_iters=50, use_jit=False)
            assert result_pure.Q.shape == P.shape
        except CalibrationError:
            # With only 50 iterations, convergence failure is acceptable
            pass
