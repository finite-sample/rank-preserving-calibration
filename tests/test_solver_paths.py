"""Solver branches checked against an independent solver or exact arithmetic."""

import logging

import numpy as np
import pytest

from rank_preserving_calibration import (
    CalibrationError,
    calibrate,
    calibrate_admm,
    calibrate_dykstra,
    calibrate_kl,
    calibrate_qp,
    calibrate_soft,
    calibrate_soft_admm,
)
from rank_preserving_calibration.calibration import logger

cp = pytest.importorskip("cvxpy")


def _problem(N=30, J=3, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(J), size=N)
    M = P.sum(axis=0) * np.linspace(1.3, 0.7, J)
    return P, M * N / M.sum()


def _rank_violation(Q, P):
    worst = 0.0
    for j in range(P.shape[1]):
        q = Q[np.argsort(P[:, j], kind="mergesort"), j]
        worst = max(worst, float(np.max(q[:-1] - q[1:], initial=0.0)))
    return worst


class TestSoftSolversAgainstQP:
    """min 0.5||Q - P||^2 + 0.5 lam_m ||Q^T 1 - M||^2 over the row simplex."""

    @staticmethod
    def _oracle(P, M, lam_m, isotonic=False):
        N, J = P.shape
        Q = cp.Variable((N, J))
        cons = [Q >= 0, cp.sum(Q, axis=1) == 1]
        if isotonic:
            for j in range(J):
                o = np.argsort(P[:, j], kind="mergesort")
                cons.extend(Q[o[i + 1], j] >= Q[o[i], j] for i in range(N - 1))
        objective = 0.5 * cp.sum_squares(Q - P) + 0.5 * lam_m * cp.sum_squares(
            cp.sum(Q, axis=0) - M
        )
        cp.Problem(cp.Minimize(objective), cons).solve(
            solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12
        )
        return Q.value

    @pytest.mark.parametrize("solver", [calibrate_soft, calibrate_soft_admm])
    @pytest.mark.parametrize("lam_m", [1.0, 10.0])
    def test_no_rank_penalty(self, solver, lam_m):
        # Guards two bugs: soft ADMM weighted the marginal term by lam_m / N,
        # and soft gradient descent used a step past 1/L and oscillated
        # between simplex corners.
        P, M = _problem()
        Q = solver(P, M, lam_m=lam_m, lam_r=0.0, max_iters=50000, tol=1e-12).Q
        np.testing.assert_allclose(Q, self._oracle(P, M, lam_m), atol=1e-7)

    def test_admm_large_rank_penalty_is_hard_isotonic(self):
        P, M = _problem()
        Q = calibrate_soft_admm(
            P, M, lam_m=1.0, lam_r=1e6, max_iters=50000, tol=1e-12
        ).Q
        expected = self._oracle(P, M, 1.0, isotonic=True)
        np.testing.assert_allclose(Q, expected, atol=1e-6)
        assert _rank_violation(Q, P) < 1e-9


class TestDykstraPaths:
    def test_polish_reaches_the_constraint_set_after_an_early_stop(self):
        P, M = _problem(N=20)
        # The callback stops the main loop at iteration 2, far from feasible,
        # so only the polish can deliver a feasible, rank-preserving matrix.
        result = calibrate_dykstra(P, M, callback=lambda it, _change, _Q: it < 2)
        assert result.iterations == 2
        assert result.converged
        np.testing.assert_allclose(result.Q.sum(axis=1), 1.0, atol=1e-8)
        np.testing.assert_allclose(result.Q.sum(axis=0), M, atol=1e-8)
        assert _rank_violation(result.Q, P) == 0.0

    def test_cycle_detection_stops_a_stalled_run(self):
        P, M = _problem(N=20)
        # sum(M) = 1.05 N: the iterates settle without becoming feasible.
        with (
            pytest.warns(UserWarning, match="Cycling detected"),
            pytest.raises(CalibrationError),
        ):
            calibrate_dykstra(P, M * 1.05, detect_cycles=True, cycle_window=3)

    def test_verbose_attaches_a_handler_and_quiet_removes_it(self, capsys):
        P, M = _problem(N=10)
        calibrate_dykstra(P, M, verbose=True)
        assert logger.level == logging.DEBUG
        assert "Dykstra iteration 1" in capsys.readouterr().err
        calibrate_dykstra(P, M, verbose=False)
        assert logger.handlers == []
        assert logger.level == logging.WARNING


class TestADMMPaths:
    def test_rejects_unknown_ties(self):
        P, M = _problem(N=10)
        with pytest.raises(ValueError, match="ties"):
            calibrate_admm(P, M, ties="average")

    def test_raises_rather_than_return_an_unconverged_iterate(self):
        P, M = _problem(N=10)
        with (
            pytest.warns(UserWarning, match="ADMM failed to converge"),
            pytest.raises(CalibrationError, match="failed to converge after 1"),
        ):
            calibrate_admm(P, M, max_iters=1, verbose=True)

    def test_calibrate_dispatches_to_admm(self):
        P, M = _problem(N=10)
        np.testing.assert_array_equal(
            calibrate(P, M, method="admm").Q, calibrate_admm(P, M).Q
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_iters": 0}, "max_iters"),
        ({"max_iters": 1.5}, "max_iters"),
        ({"tol": 0.0}, "tol"),
        ({"feasibility_tol": -1.0}, "feasibility_tol"),
    ],
)
def test_rejects_bad_solver_parameters(kwargs, message):
    P, M = _problem(N=5)
    with pytest.raises(CalibrationError, match=message):
        calibrate_dykstra(P, M, **kwargs)


def test_qp_single_row_returns_the_targets():
    # With one row, the row-sum and column-sum constraints pin Q to M.
    Q = calibrate_qp(np.array([[0.6, 0.4]]), np.array([0.3, 0.7])).Q
    np.testing.assert_allclose(Q, [[0.3, 0.7]], atol=1e-8)


def test_kl_zero_target_class_gets_zero_mass():
    P, M = _problem(N=20)
    M[2] = 0.0
    M *= 20 / M.sum()
    Q = calibrate_kl(P, M).Q
    np.testing.assert_allclose(Q[:, 2], 0.0, atol=1e-12)
    np.testing.assert_allclose(Q.sum(axis=0), M, atol=1e-8)
