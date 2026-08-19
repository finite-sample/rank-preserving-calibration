"""Tests for the exact QP solver and the `calibrate` dispatcher.

The point of keeping two solvers is that they check each other. Dykstra depends
only on numpy and is slow but well understood; the QP path is fast but leans on
an external interior-point solver. They solve the *same* convex problem, so
disagreement between them is a bug in one of them, and most of the assertions
here are about agreement rather than about either one alone.

A third opinion, cvxpy, is used where it is installed.
"""

from __future__ import annotations

import contextlib
import time

import numpy as np
import pytest

from rank_preserving_calibration import (
    calibrate,
    calibrate_dykstra,
    calibrate_qp,
)
from rank_preserving_calibration.calibration import CalibrationError, is_feasible


def feasible_problem(seed: int, N: int = 30, J: int = 4, tilt: float = 0.5):
    """Targets that differ from P's column sums but still sum to N.

    Parameters
    ----------
    seed
        Random seed.
    N
        Rows.
    J
        Classes.
    tilt
        How far the targets are pushed from P's own column sums.

    Returns
    -------
    tuple of ndarray
        Probability matrix and target column sums.
    """
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(J), size=N)
    base = P.sum(axis=0)
    direction = rng.normal(0.0, 1.0, J)
    direction -= direction.mean()
    M = base + tilt * base.mean() * direction
    M = np.maximum(M, 1e-3)
    M *= N / M.sum()
    return P, M


def rank_violations(P: np.ndarray, Q: np.ndarray, tol: float = 1e-9) -> int:
    """Count adjacent within-class inversions in the original score order."""
    bad = 0
    for j in range(Q.shape[1]):
        order = np.argsort(P[:, j], kind="mergesort")
        bad += int((np.diff(Q[order, j]) < -tol).sum())
    return bad


# --------------------------------------------------------------------------- #
# The two solvers must agree
# --------------------------------------------------------------------------- #


class TestSolversAgree:
    """Same convex problem, so the same optimum -- whichever route is taken."""

    @pytest.mark.parametrize("seed", range(5))
    def test_qp_matches_dykstra_objective(self, seed):
        """Objectives must coincide; the minimiser of a strictly convex QP is unique."""
        P, M = feasible_problem(seed, N=30, J=4)
        Q_qp = calibrate_qp(P, M).Q
        Q_dyk = calibrate_dykstra(P, M, max_iters=50000, tol=1e-10).Q
        assert float(np.sum((Q_qp - P) ** 2)) == pytest.approx(
            float(np.sum((Q_dyk - P) ** 2)), abs=1e-6
        )

    @pytest.mark.parametrize("seed", range(3))
    def test_qp_matches_dykstra_pointwise(self, seed):
        """The objective being equal is not enough: the matrices must match too."""
        P, M = feasible_problem(seed, N=25, J=3)
        Q_qp = calibrate_qp(P, M).Q
        Q_dyk = calibrate_dykstra(P, M, max_iters=50000, tol=1e-10).Q
        np.testing.assert_allclose(Q_qp, Q_dyk, atol=1e-5)

    @pytest.mark.parametrize(("N", "J"), [(10, 3), (20, 4), (30, 3)])
    def test_qp_matches_an_independent_solver(self, N, J):
        """Third opinion, so agreement is not two implementations of one mistake."""
        cp = pytest.importorskip("cvxpy")
        P, M = feasible_problem(11, N=N, J=J)

        Q = cp.Variable((N, J))
        cons = [Q >= 0, cp.sum(Q, axis=1) == 1, cp.sum(Q, axis=0) == M]
        for j in range(J):
            order = np.argsort(P[:, j], kind="mergesort")
            cons.extend(Q[order[i + 1], j] >= Q[order[i], j] for i in range(N - 1))
        problem = cp.Problem(cp.Minimize(cp.sum_squares(Q - P)), cons)
        problem.solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12)

        ours = calibrate_qp(P, M).Q
        assert float(np.sum((ours - P) ** 2)) == pytest.approx(
            float(problem.value), abs=1e-6
        )


# --------------------------------------------------------------------------- #
# The guarantee the package exists for
# --------------------------------------------------------------------------- #


class TestQPPreservesRank:
    """Zero violations, not few. This is a constraint, not an objective term."""

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize(("N", "J"), [(50, 3), (120, 4)])
    def test_no_within_class_inversions(self, seed, N, J):
        P, M = feasible_problem(seed, N=N, J=J)
        Q = calibrate_qp(P, M).Q
        assert rank_violations(P, Q) == 0

    def test_survives_heavy_ties(self):
        """Tied scores are the ordinary case and must not break the ordering."""
        rng = np.random.default_rng(3)
        P = np.round(rng.dirichlet(np.ones(4), size=150), 2)
        P /= P.sum(axis=1, keepdims=True)
        M = P.sum(axis=0)
        M *= P.shape[0] / M.sum()
        Q = calibrate_qp(P, M).Q
        assert rank_violations(P, Q) == 0

    @pytest.mark.parametrize("seed", range(3))
    def test_output_is_feasible(self, seed):
        """Rows sum to 1 and columns hit their targets, to absolute tolerance."""
        P, M = feasible_problem(seed, N=60, J=4)
        result = calibrate_qp(P, M)
        assert result.converged
        assert is_feasible(result.Q, M, row_atol=1e-8, col_atol=1e-8)
        assert np.all(result.Q >= 0.0)


# --------------------------------------------------------------------------- #
# The epsilon relaxation
# --------------------------------------------------------------------------- #


class TestEpsilonRelaxation:
    """Handled exactly, by lowering the isotonic bound rather than penalising."""

    @pytest.mark.parametrize("eps", [0.01, 0.05])
    def test_decreases_stay_within_the_bound(self, eps):
        P, M = feasible_problem(4, N=80, J=4)
        Q = calibrate_qp(P, M, nearly={"mode": "epsilon", "eps": eps}).Q
        worst = 0.0
        for j in range(Q.shape[1]):
            order = np.argsort(P[:, j], kind="mergesort")
            worst = min(worst, float(np.diff(Q[order, j]).min()))
        assert worst >= -eps - 1e-8

    def test_zero_epsilon_reduces_to_the_strict_solution(self):
        P, M = feasible_problem(5, N=40, J=3)
        strict = calibrate_qp(P, M).Q
        relaxed = calibrate_qp(P, M, nearly={"mode": "epsilon", "eps": 0.0}).Q
        np.testing.assert_allclose(strict, relaxed, atol=1e-8)

    def test_relaxing_cannot_worsen_the_fit(self):
        """A larger feasible set cannot have a worse optimum."""
        P, M = feasible_problem(6, N=60, J=4)
        strict = float(np.sum((calibrate_qp(P, M).Q - P) ** 2))
        loose = float(
            np.sum(
                (calibrate_qp(P, M, nearly={"mode": "epsilon", "eps": 0.05}).Q - P) ** 2
            )
        )
        assert loose <= strict + 1e-8

    def test_negative_epsilon_is_rejected(self):
        P, M = feasible_problem(7, N=20, J=3)
        with pytest.raises(CalibrationError, match="non-negative"):
            calibrate_qp(P, M, nearly={"mode": "epsilon", "eps": -0.1})

    def test_lambda_mode_is_rejected_with_an_explanation(self):
        """The penalty mode changes the objective, so it is not this projection."""
        P, M = feasible_problem(8, N=20, J=3)
        with pytest.raises(CalibrationError, match=r"lambda-penalty|mode"):
            calibrate_qp(P, M, nearly={"mode": "lambda", "lam": 1.0})


# --------------------------------------------------------------------------- #
# The dispatcher
# --------------------------------------------------------------------------- #


class TestCalibrateDispatcher:
    """`calibrate` is the recommended entry point."""

    def test_auto_selects_the_qp_solver(self):
        P, M = feasible_problem(9, N=30, J=3)
        np.testing.assert_allclose(calibrate(P, M).Q, calibrate_qp(P, M).Q, atol=1e-10)

    @pytest.mark.parametrize("method", ["auto", "qp", "dykstra"])
    def test_every_method_returns_a_feasible_rank_preserving_matrix(self, method):
        P, M = feasible_problem(10, N=25, J=3)
        Q = calibrate(P, M, method=method).Q
        assert is_feasible(Q, M, row_atol=1e-7, col_atol=1e-7)
        assert rank_violations(P, Q) == 0

    def test_unknown_method_is_rejected(self):
        P, M = feasible_problem(11, N=10, J=2)
        with pytest.raises(CalibrationError, match="method must be one of"):
            calibrate(P, M, method="newton")

    def test_infeasible_targets_still_warn(self):
        """The exact-feasibility rule holds whichever solver is chosen."""
        P, _ = feasible_problem(12, N=30, J=3)
        M = P.sum(axis=0) * 1.05
        with (
            pytest.warns(UserWarning, match="(?i)feasib|must equal"),
            contextlib.suppress(Exception),
        ):
            calibrate(P, M)


# --------------------------------------------------------------------------- #
# The ceiling must not silently return
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_scales_past_where_alternating_projections_stall():
    """N=1600 is far beyond Dykstra's practical ceiling of roughly N=200.

    Timing in a test is normally a smell, but the whole point of this solver is
    that the previous one took 274s at N=400 and did not converge. The budget is
    deliberately loose -- it is a regression guard, not a benchmark.
    """
    P, M = feasible_problem(13, N=1600, J=4)
    start = time.perf_counter()
    result = calibrate_qp(P, M)
    elapsed = time.perf_counter() - start

    assert result.converged
    assert rank_violations(P, result.Q) == 0
    assert elapsed < 60.0, f"N=1600 took {elapsed:.1f}s; it measured 1.4s when written"
