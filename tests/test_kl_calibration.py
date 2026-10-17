"""
Test suite for KL-divergence rank-preserving calibration.

Tests cover:
- Basic KL calibration functionality
- Constraint satisfaction (row simplex, column sums, rank preservation)
- Geometric mean pooling in KL isotonic regression
- Multiplicative rescaling for sum constraints
- Anchor-reference decoupling
- Soft KL calibration with lambda penalty
- Pareto frontier computation
- Comparison with Euclidean methods

Run with: python -m pytest tests/test_kl_calibration.py -v
"""

import numpy as np
import pytest

from rank_preserving_calibration import (
    CalibrationError,
    KLCalibrationResult,
    KLParetoResult,
    calibrate_dykstra,
    calibrate_ipf,
    calibrate_kl,
    calibrate_kl_pareto,
    calibrate_kl_soft,
    kl_divergence,
    project_near_kl_isotonic,
    prox_kl_near_isotonic,
    reverse_kl_divergence,
    symmetrized_kl,
)

from .data_helpers import create_test_case


class TestCalibrateKL:
    """Test basic KL calibration functionality."""

    def test_simple_2x2_case(self):
        """Test simple 2x2 case that should converge easily."""
        P = np.array([[0.8, 0.2], [0.3, 0.7]])
        M = np.array([1.0, 1.0])

        result = calibrate_kl(P, M, verbose=False)

        assert isinstance(result, KLCalibrationResult)
        assert result.Q.shape == P.shape
        assert result.max_row_error < 1e-6
        assert result.max_col_error < 1e-6
        assert np.allclose(result.Q.sum(axis=1), 1.0, atol=1e-8)
        assert np.allclose(result.Q.sum(axis=0), M, atol=1e-6)

    def test_random_case(self):
        """Test on random generated data."""
        P, M = create_test_case("random", N=20, J=3, seed=42)

        result = calibrate_kl(P, M, verbose=False)

        assert result.converged
        assert result.max_row_error < 1e-5
        assert result.max_col_error < 1e-5

    def test_rank_preservation(self):
        """Test that rank ordering is preserved."""
        P, M = create_test_case("linear", N=10, J=3, seed=42)

        result = calibrate_kl(P, M, verbose=False)

        for j in range(P.shape[1]):
            original_order = np.argsort(P[:, j])
            calibrated_sorted = result.Q[original_order, j]
            diffs = np.diff(calibrated_sorted)
            assert np.all(diffs >= -1e-8), f"Rank violation in column {j}"

    def test_probability_constraints(self):
        """Test that probability constraints are satisfied."""
        P, M = create_test_case("skewed", N=15, J=4, seed=123)

        result = calibrate_kl(P, M, verbose=False)

        assert np.all(result.Q >= 0), "Negative probability found"
        row_sums = result.Q.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-8), "Row sums not 1"
        col_sums = result.Q.sum(axis=0)
        assert np.allclose(col_sums, M, atol=1e-6), "Column sums don't match M"

    def test_kl_divergence_finite(self):
        """Test that KL divergence in result is finite and non-negative."""
        P, M = create_test_case("random", N=20, J=4, seed=99)

        result = calibrate_kl(P, M, verbose=False)

        assert np.isfinite(result.kl_divergence)
        assert result.kl_divergence >= 0


def _shifted_problem(N=30, J=3, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(J), size=N)
    M = P.sum(axis=0) * np.linspace(1.3, 0.7, J)
    return P, M * N / M.sum()


class TestCalibrateKLSoft:
    """calibrate_kl_soft checked against problems with a known answer."""

    def test_no_penalty_is_the_kl_projection_ipf_computes(self):
        # min KL(Q||P) over rows-on-simplex and column sums M is what IPF
        # (Sinkhorn) converges to, by a different algorithm entirely.
        P, M = _shifted_problem()
        ipf = calibrate_ipf(P, M, max_iters=10000, tol=1e-14).Q
        np.testing.assert_allclose(calibrate_kl_soft(P, M, lam=0.0).Q, ipf, atol=1e-7)

    def test_large_penalty_equals_the_hard_problem(self):
        P, M = _shifted_problem()
        soft = calibrate_kl_soft(P, M, lam=1e4)
        hard = calibrate_kl(P, M)
        assert soft.max_rank_violation < 1e-8
        # KL is flat at its minimum, so the value is pinned far more tightly
        # than the matrix: 1e-10 against ~1e-6 here.
        assert soft.kl_divergence == pytest.approx(hard.kl_divergence, abs=1e-8)
        np.testing.assert_allclose(soft.Q, hard.Q, atol=1e-5)

    def test_constraints_hold_at_every_penalty(self):
        P, M = _shifted_problem()
        for lam in (0.0, 0.1, 10.0):
            result = calibrate_kl_soft(P, M, lam=lam)
            assert result.max_row_error < 1e-8
            assert result.max_col_error < 1e-8

    def test_rejects_negative_penalty(self):
        P, M = _shifted_problem()
        with pytest.raises(CalibrationError, match="lam"):
            calibrate_kl_soft(P, M, lam=-1.0)


class TestCalibrateKLOptimality:
    def test_reaches_the_constrained_optimum(self):
        """The alternating-projection version stopped at KL 1.5443 here."""
        cp = pytest.importorskip("cvxpy")
        P, M = _shifted_problem()
        N, J = P.shape
        Q = cp.Variable((N, J))
        cons = [cp.sum(Q, axis=1) == 1, cp.sum(Q, axis=0) == M]
        for j in range(J):
            o = np.argsort(P[:, j], kind="mergesort")
            cons.append(Q[o[1:], j] >= Q[o[:-1], j])
        problem = cp.Problem(cp.Minimize(cp.sum(cp.kl_div(Q, P))), cons)
        problem.solve(
            solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10
        )
        assert problem.status == "optimal"

        result = calibrate_kl(P, M)
        assert result.kl_divergence == pytest.approx(problem.value, abs=1e-7)
        assert result.max_rank_violation < 1e-8

    def test_beats_another_feasible_point(self):
        # Dykstra's Euclidean projection satisfies the same constraints, so its
        # KL can only be higher.
        P, M = _shifted_problem(seed=3)
        Q_euclid = calibrate_dykstra(P, M).Q
        assert calibrate_kl(P, M).kl_divergence <= kl_divergence(Q_euclid, P) + 1e-9


class TestAnchorReferenceDecoupling:
    """Test anchor-reference decoupling feature."""

    def test_different_anchor_produces_different_result(self):
        """Test that different anchor A produces different ordering."""
        P = np.array([[0.6, 0.4], [0.3, 0.7], [0.5, 0.5], [0.8, 0.2]])
        M = np.array([2.0, 2.0])

        A_different = np.array([[0.2, 0.8], [0.9, 0.1], [0.5, 0.5], [0.4, 0.6]])

        result_same = calibrate_kl(P, M, R=P, A=P, verbose=False)
        result_diff = calibrate_kl(P, M, R=P, A=A_different, verbose=False)

        assert not np.allclose(result_same.Q, result_diff.Q, atol=1e-4), (
            "Different anchors should produce different results"
        )

    def test_anchor_determines_rank_order(self):
        """Test that anchor determines which rankings to preserve."""
        P = np.array([[0.7, 0.3], [0.4, 0.6], [0.5, 0.5]])
        M = np.array([1.5, 1.5])

        A = np.array([[0.2, 0.8], [0.9, 0.1], [0.5, 0.5]])

        result = calibrate_kl(P, M, R=P, A=A, verbose=False)

        for j in range(P.shape[1]):
            anchor_order = np.argsort(A[:, j])
            q_sorted = result.Q[anchor_order, j]
            diffs = np.diff(q_sorted)
            assert np.all(diffs >= -1e-8), f"Rank violation in column {j}"


class TestKLvsEuclidean:
    """Compare KL and Euclidean calibration on same problems."""

    def test_both_satisfy_constraints(self):
        """Test that both methods satisfy the same constraints."""
        P, M = create_test_case("random", N=20, J=3, seed=42)

        result_kl = calibrate_kl(P, M, verbose=False)
        result_euc = calibrate_dykstra(P, M, verbose=False)

        assert np.allclose(result_kl.Q.sum(axis=1), 1.0, atol=1e-6)
        assert np.allclose(result_euc.Q.sum(axis=1), 1.0, atol=1e-6)

        assert np.allclose(result_kl.Q.sum(axis=0), M, atol=1e-5)
        assert np.allclose(result_euc.Q.sum(axis=0), M, atol=1e-5)

    def test_solutions_can_differ(self):
        """Test that KL and Euclidean can give different solutions."""
        rng = np.random.default_rng(789)
        P = rng.dirichlet(np.ones(4), size=30)
        target_shift = rng.normal(0, 0.1, 4)
        target_shift -= target_shift.mean()
        M = P.sum(axis=0) + target_shift * 3

        result_kl = calibrate_kl(P, M, verbose=False)
        result_euc = calibrate_dykstra(P, M, verbose=False)

        diff = np.linalg.norm(result_kl.Q - result_euc.Q)
        assert diff < 5.0


class TestKLPareto:
    """Test Pareto frontier computation."""

    def test_pareto_returns_multiple_solutions(self):
        """Test that Pareto solver returns multiple solutions."""
        P, M = create_test_case("random", N=15, J=3, seed=42)
        lambda_grid = [0.1, 1.0, 10.0]

        result = calibrate_kl_pareto(P, M, lambda_grid=lambda_grid, verbose=False)

        assert isinstance(result, KLParetoResult)
        assert len(result.solutions) >= 1
        assert len(result.kl_values) == len(result.solutions)
        assert len(result.rank_violations) == len(result.solutions)

    def test_pareto_is_monotone(self):
        """Exact solves trade KL for rank penalty, never both at once."""
        P, M = _shifted_problem()
        result = calibrate_kl_pareto(P, M)
        assert np.all(np.diff(result.kl_values) >= -1e-7)
        assert np.all(np.diff(result.rank_violations) <= 1e-7)
        assert result.lambda_path == sorted(result.lambda_path)

    def test_pareto_solutions_valid(self):
        """Test that all Pareto solutions satisfy basic constraints."""
        P, M = create_test_case("random", N=15, J=3, seed=456)
        lambda_grid = [0.1, 1.0, 10.0]

        result = calibrate_kl_pareto(P, M, lambda_grid=lambda_grid, verbose=False)

        for _, Q in result.solutions:
            row_sums = Q.sum(axis=1)
            assert np.allclose(row_sums, 1.0, atol=1e-8)
            assert np.all(Q >= -1e-10)


class TestKLNearlyIsotonic:
    """Test KL nearly-isotonic utilities."""

    def test_near_kl_isotonic_basic(self):
        """Test basic near-KL-isotonic projection."""
        v = np.array([0.5, 0.2, 0.3, 0.1])
        z = project_near_kl_isotonic(v, eps_slack=0.1)

        assert np.all(z > 0)
        assert z.shape == v.shape

    def test_near_kl_isotonic_with_sum(self):
        """Test near-KL-isotonic with sum constraint."""
        v = np.array([0.3, 0.4, 0.2, 0.1])
        target_sum = 2.0

        z = project_near_kl_isotonic(v, eps_slack=0.05, sum_target=target_sum)

        assert np.allclose(z.sum(), target_sum, rtol=1e-6)

    def test_near_kl_isotonic_zero_slack_is_strict(self):
        """Test that zero slack gives strict isotonic."""
        v = np.array([0.4, 0.1, 0.2])
        z = project_near_kl_isotonic(v, eps_slack=0.0)

        diffs = np.diff(z)
        assert np.all(diffs >= -1e-10)

    def test_prox_kl_near_isotonic_basic(self):
        """Test basic KL nearly-isotonic prox."""
        y = np.array([0.5, 0.2, 0.3, 0.4])
        z = prox_kl_near_isotonic(y, lam=1.0)

        assert isinstance(z, np.ndarray)
        assert z.shape == y.shape
        assert np.all(z > 0)

    def test_prox_kl_zero_lambda_unchanged(self):
        """Test that zero lambda returns input unchanged."""
        y = np.array([0.3, 0.5, 0.2])
        z = prox_kl_near_isotonic(y, lam=0.0)

        assert np.allclose(z, y)


class TestKLDivergenceMetrics:
    """Test KL divergence metric functions."""

    def test_kl_divergence_nonnegative(self):
        """Test that KL divergence is non-negative."""
        Q = np.array([[0.6, 0.4], [0.3, 0.7]])
        R = np.array([[0.5, 0.5], [0.4, 0.6]])

        kl = kl_divergence(Q, R)

        assert kl >= 0

    def test_kl_divergence_zero_for_same(self):
        """Test that KL(P||P) = 0."""
        P = np.array([[0.6, 0.4], [0.3, 0.7]])

        kl = kl_divergence(P, P)

        assert np.isclose(kl, 0.0, atol=1e-10)

    def test_kl_divergence_asymmetric(self):
        """Test that KL is asymmetric: KL(Q||R) != KL(R||Q) in general."""
        Q = np.array([[0.8, 0.2], [0.3, 0.7]])
        R = np.array([[0.5, 0.5], [0.4, 0.6]])

        kl_forward = kl_divergence(Q, R)
        kl_reverse = kl_divergence(R, Q)

        assert not np.isclose(kl_forward, kl_reverse)

    def test_reverse_kl_divergence(self):
        """Test reverse KL divergence."""
        Q = np.array([[0.6, 0.4], [0.3, 0.7]])
        R = np.array([[0.5, 0.5], [0.4, 0.6]])

        rkl = reverse_kl_divergence(Q, R)
        expected = kl_divergence(R, Q)

        assert np.isclose(rkl, expected)

    def test_symmetrized_kl(self):
        """Test symmetrized KL divergence."""
        Q = np.array([[0.6, 0.4], [0.3, 0.7]])
        R = np.array([[0.5, 0.5], [0.4, 0.6]])

        skl = symmetrized_kl(Q, R)
        expected = 0.5 * (kl_divergence(Q, R) + kl_divergence(R, Q))

        assert np.isclose(skl, expected)

    def test_symmetrized_kl_is_symmetric(self):
        """Test that symmetrized KL is symmetric."""
        Q = np.array([[0.6, 0.4], [0.3, 0.7]])
        R = np.array([[0.5, 0.5], [0.4, 0.6]])

        skl_qr = symmetrized_kl(Q, R)
        skl_rq = symmetrized_kl(R, Q)

        assert np.isclose(skl_qr, skl_rq)


class TestKLInputValidation:
    """Test input validation for KL calibration."""

    def test_invalid_R_shape_raises(self):
        """Test that mismatched R shape raises error."""
        P = np.array([[0.6, 0.4], [0.3, 0.7]])
        M = np.array([1.0, 1.0])
        R = np.array([[0.5, 0.5, 0.0]])

        with pytest.raises(CalibrationError):
            calibrate_kl(P, M, R=R)

    def test_invalid_A_shape_raises(self):
        """Test that mismatched A shape raises error."""
        P = np.array([[0.6, 0.4], [0.3, 0.7]])
        M = np.array([1.0, 1.0])
        A = np.array([[0.5, 0.5, 0.0]])

        with pytest.raises(CalibrationError):
            calibrate_kl(P, M, A=A)


class TestKLReferenceEdges:
    @staticmethod
    def _zeroed(P, rows):
        R = P.copy()
        R[rows, [0, 1, 2]] = 0.0
        return R / R.sum(axis=1, keepdims=True)

    def test_zero_reference_entries_get_zero_mass(self):
        # KL(Q||R) is infinite unless q = 0 wherever r = 0; the solver imposes
        # that directly, since the cone has no interior there. Zeros on each
        # column's lowest-ranked row keep the hard problem feasible.
        P, M = _shifted_problem(N=12)
        lowest = [int(np.argmin(P[:, j])) for j in range(3)]
        R = self._zeroed(P, lowest)
        for result in (calibrate_kl(P, M, R=R), calibrate_kl_soft(P, M, lam=1.0, R=R)):
            np.testing.assert_allclose(result.Q[lowest, [0, 1, 2]], 0.0, atol=1e-12)
            assert result.max_row_error < 1e-8
            assert result.max_col_error < 1e-8
            assert np.isfinite(result.kl_divergence)

    def test_zero_high_in_the_order_makes_the_hard_problem_infeasible(self):
        # A zero at rank 10 of 12 forces 11 entries of that column to zero, so
        # its total cannot be met; the soft problem only pays for the steps.
        P, M = _shifted_problem(N=12)
        R = self._zeroed(P, [0, 5, 9])
        with pytest.raises(CalibrationError, match="probably infeasible"):
            calibrate_kl(P, M, R=R)
        soft = calibrate_kl_soft(P, M, lam=1.0, R=R)
        np.testing.assert_allclose(soft.Q[[0, 5, 9], [0, 1, 2]], 0.0, atol=1e-12)

    def test_rejects_negative_reference(self):
        P, M = _shifted_problem(N=6)
        R = P.copy()
        R[0, 0] = -0.1
        with pytest.raises(CalibrationError, match="non-negative"):
            calibrate_kl(P, M, R=R)

    def test_soft_raises_on_infeasible_targets(self):
        P, M = _shifted_problem(N=6)
        with (
            pytest.warns(UserWarning, match="must equal N"),
            pytest.raises(CalibrationError, match="status"),
        ):
            calibrate_kl_soft(P, M * 1.1, lam=1.0)


class TestKLConfidentClassifiers:
    """Near-one-hot references, where the solver used to stall."""

    def test_reference_entries_near_zero(self):
        # Codex's case on #43: the smallest reference value is 1.4e-10, and
        # with r itself inside the exponential cone the default solve stalled.
        rng = np.random.default_rng(6)
        P = rng.dirichlet(np.full(2, 0.2), size=5)
        M = P.sum(axis=0) * np.array([1.1, 0.9])
        M *= 5 / M.sum()
        for result in (calibrate_kl(P, M), calibrate_kl_soft(P, M, lam=1.0)):
            assert result.max_col_error < 1e-7
            assert result.max_row_error < 1e-7

    @pytest.mark.parametrize("seed", range(8))
    def test_peaked_problems_solve(self, seed):
        rng = np.random.default_rng(seed)
        P = rng.dirichlet(np.full(3, 0.2), size=40)
        M = P.sum(axis=0) * np.array([1.2, 1.0, 0.8])
        M *= 40 / M.sum()
        hard = calibrate_kl(P, M)
        assert hard.max_rank_violation < 1e-6
        assert hard.max_col_error < 1e-6
        assert calibrate_kl_soft(P, M, lam=1.0).max_col_error < 1e-6


class TestKLConvergenceIsJudgedOnTheMatrix:
    """A solver status of Solved is not enough; the returned Q must fit."""

    @staticmethod
    def _off_target(monkeypatch):
        import rank_preserving_calibration.kl_calibration as kl

        real = kl._solve_kl

        def fake(R, A, M, lam, tol, verbose):
            Q, iterations, achieved = real(R, A, M, lam, tol, verbose)
            Q = Q.copy()
            Q[0] *= 1.001  # row 0 now sums to 1.001
            return Q, iterations, achieved

        monkeypatch.setattr(kl, "_solve_kl", fake)

    def test_soft_reports_and_warns(self, monkeypatch):
        P, M = _shifted_problem(N=10)
        self._off_target(monkeypatch)
        with pytest.warns(UserWarning, match="missed the constraints"):
            result = calibrate_kl_soft(P, M, lam=1.0)
        assert not result.converged
        assert result.max_row_error == pytest.approx(1e-3, rel=1e-3)

    def test_hard_raises(self, monkeypatch):
        P, M = _shifted_problem(N=10)
        self._off_target(monkeypatch)
        with pytest.raises(CalibrationError, match="could not certify"):
            calibrate_kl(P, M)


class TestKLTiedScores:
    def test_hard_problem_with_ties_reaches_the_constrained_optimum(self):
        # Tied scores used to read as violations under an unstable sort, so
        # the certificate rejected this feasible problem.
        cp = pytest.importorskip("cvxpy")
        P = np.random.default_rng(0).integers(1, 6, (20, 3)).astype(float)
        P /= P.sum(axis=1, keepdims=True)
        M = P.sum(axis=0) * np.array([1.3, 1.0, 0.7])
        M *= len(P) / M.sum()
        result = calibrate_kl(P, M)

        Q = cp.Variable(P.shape)
        cons = [cp.sum(Q, axis=1) == 1, cp.sum(Q, axis=0) == M]
        for j in range(P.shape[1]):
            o = np.argsort(P[:, j], kind="mergesort")
            cons.append(Q[o[1:], j] >= Q[o[:-1], j])
        problem = cp.Problem(cp.Minimize(cp.sum(cp.kl_div(Q, P))), cons)
        problem.solve(solver=cp.CLARABEL)
        assert problem.status == "optimal"
        assert result.kl_divergence == pytest.approx(problem.value, abs=1e-6)


class TestKLSlackProjection:
    """project_near_kl_isotonic against the convex problem it claims to solve."""

    @staticmethod
    def _oracle(v, eps, w, s):
        cp = pytest.importorskip("cvxpy")
        z = cp.Variable(len(v))
        cons = [z[i + 1] >= (1 - eps) * z[i] for i in range(len(v) - 1)]
        if s is not None:
            cons.append(cp.sum(z) == s)
        cp.Problem(cp.Minimize(cp.sum(cp.multiply(w, cp.kl_div(z, v)))), cons).solve(
            solver=cp.CLARABEL
        )
        return z.value

    @pytest.mark.parametrize(
        ("eps", "weighted", "s"),
        [(0.5, False, 1.0), (0.2, False, None), (0.2, False, 3.0), (0.2, True, None)],
    )
    def test_matches_the_kl_projection(self, eps, weighted, s):
        from scipy.special import kl_div

        rng = np.random.default_rng(1)
        v = np.array([0.6, 0.1, 0.3]) if eps == 0.5 else rng.uniform(0.05, 1, 12)
        w = rng.uniform(0.5, 3, v.size) if weighted else None
        ww = np.ones(v.size) if w is None else w
        ours = project_near_kl_isotonic(v, eps, sum_target=s, weights=w)
        expected = self._oracle(v, eps, ww, s)
        assert np.all(ours[1:] >= (1 - eps) * ours[:-1] - 1e-12)
        # The oracle is accurate to its own tolerance, so compare objectives.
        assert (ww * kl_div(ours, v)).sum() <= (ww * kl_div(expected, v)).sum() + 1e-7

    def test_weighted_with_a_sum_target_raises(self):
        with pytest.raises(ValueError, match="uniform weights"):
            project_near_kl_isotonic(
                np.array([0.6, 0.1, 0.3]),
                0.2,
                sum_target=1.0,
                weights=np.array([1, 2, 3]),
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
