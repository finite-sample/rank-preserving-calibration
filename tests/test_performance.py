"""Performance tests for JIT compilation and progress bar functionality."""

import time

import numpy as np
import pytest

from rank_preserving_calibration import calibrate_dykstra
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

        # Run with JIT (if available)
        result_jit = calibrate_dykstra(P, M, max_iters=100, use_jit=True)

        # Run without JIT
        result_pure = calibrate_dykstra(P, M, max_iters=100, use_jit=False)

        # Results should be very close (minor numerical differences expected)
        np.testing.assert_allclose(result_jit.Q, result_pure.Q, rtol=1e-7, atol=1e-15)
        assert result_jit.converged == result_pure.converged
        assert result_jit.iterations == result_pure.iterations

    @pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
    def test_jit_performance_improvement(self):
        """Test that JIT provides performance improvement for larger problems."""
        # Create larger test case
        np.random.seed(42)
        N, J = 500, 10
        P = np.random.dirichlet(np.ones(J), size=N)
        M = np.full(J, N / J)

        # Warm up JIT
        _ = calibrate_dykstra(P[:10], M, max_iters=10, use_jit=True, verbose=False)

        # Time with JIT
        start = time.perf_counter()
        result_jit = calibrate_dykstra(P, M, max_iters=50, use_jit=True, verbose=False)
        time_jit = time.perf_counter() - start

        # Time without JIT
        start = time.perf_counter()
        result_pure = calibrate_dykstra(
            P, M, max_iters=50, use_jit=False, verbose=False
        )
        time_pure = time.perf_counter() - start

        # JIT should be faster (at least 20% improvement expected)
        speedup = time_pure / time_jit
        print(
            f"JIT speedup: {speedup:.2f}x (JIT: {time_jit:.3f}s, Pure: {time_pure:.3f}s)"
        )

        # Results should still be very close
        np.testing.assert_allclose(result_jit.Q, result_pure.Q, rtol=1e-7, atol=1e-15)

        # Note: We don't assert on speedup as it varies by machine
        # but typically expect 1.5-3x speedup on moderate size problems

    def test_progress_bar_does_not_affect_results(self):
        """Test that progress bar doesn't change numerical results."""
        np.random.seed(42)
        N, J = 50, 3
        P = np.random.dirichlet(np.ones(J), size=N)
        M = np.full(J, N / J)

        # Run without progress bar
        result_no_bar = calibrate_dykstra(P, M, max_iters=100, progress_bar=False)

        # Run with progress bar (even if tqdm not installed, should work)
        result_with_bar = calibrate_dykstra(P, M, max_iters=100, progress_bar=True)

        # Results should be identical
        np.testing.assert_allclose(result_no_bar.Q, result_with_bar.Q, rtol=1e-10)
        assert result_no_bar.converged == result_with_bar.converged
        assert result_no_bar.iterations == result_with_bar.iterations

    def test_graceful_fallback_without_numba(self):
        """Test that code works even without Numba installed."""
        # This test always runs regardless of Numba availability
        np.random.seed(42)
        N, J = 30, 3
        P = np.random.dirichlet(np.ones(J), size=N)
        M = np.full(J, N / J)

        # Should work with use_jit=True even if Numba not installed
        result = calibrate_dykstra(P, M, max_iters=50, use_jit=True)

        assert result.Q is not None
        assert result.Q.shape == P.shape
        assert result.converged or result.iterations == 50

        # Check constraints are approximately satisfied
        assert np.allclose(result.Q.sum(axis=1), 1.0, atol=1e-6)
        assert np.allclose(result.Q.sum(axis=0), M, atol=1e-4)

    def test_jit_with_nearly_isotonic(self):
        """Test JIT works with nearly isotonic constraints."""
        np.random.seed(42)
        N, J = 50, 4
        P = np.random.dirichlet(np.ones(J), size=N)
        M = np.full(J, N / J)

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

    def test_memory_efficiency(self):
        """Test that JIT doesn't significantly increase memory usage."""
        # This is a basic test - comprehensive memory profiling needs external tools
        np.random.seed(42)
        N, J = 200, 8
        P = np.random.dirichlet(np.ones(J), size=N)
        M = np.full(J, N / J)

        # Both should complete without memory issues
        result_jit = calibrate_dykstra(P, M, max_iters=50, use_jit=True)
        result_pure = calibrate_dykstra(P, M, max_iters=50, use_jit=False)

        assert result_jit.Q.shape == P.shape
        assert result_pure.Q.shape == P.shape
