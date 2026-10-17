"""Every public solver rejects the same malformed inputs with CalibrationError."""

import numpy as np
import pytest

from rank_preserving_calibration import (
    CalibrationError,
    calibrate,
    calibrate_dykstra,
    calibrate_ipf,
    calibrate_kl,
    calibrate_kl_pareto,
    calibrate_kl_soft,
    calibrate_qp,
    calibrate_soft,
    calibrate_two_stage,
)

SOLVERS = [
    calibrate,
    calibrate_dykstra,
    calibrate_ipf,
    calibrate_kl,
    calibrate_kl_pareto,
    calibrate_kl_soft,
    calibrate_qp,
    calibrate_soft,
    calibrate_two_stage,
]

P_OK = np.array([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]])
M_OK = np.array([1.3, 1.7])

BAD_INPUTS = {
    "P not an array": ([[0.7, 0.3], [0.4, 0.6]], np.array([1.0, 1.0]), "P must be"),
    "P is 1-D": (np.array([0.7, 0.3]), np.array([0.5, 0.5]), "2D"),
    "P is empty": (np.empty((0, 2)), np.array([0.0, 0.0]), "empty"),
    "P has NaN": (np.array([[np.nan, 1.0], [0.5, 0.5]]), np.ones(2), "NaN"),
    "P is negative": (np.array([[-0.1, 1.1], [0.5, 0.5]]), np.ones(2), "non-negative"),
    "P has one column": (np.ones((3, 1)), np.array([3.0]), "at least 2 columns"),
    "M not an array": (P_OK, [1.3, 1.7], "M must be"),
    "M is 2-D": (P_OK, M_OK.reshape(1, 2), "1D"),
    "M wrong length": (P_OK, np.array([1.0, 1.0, 1.0]), "length 2"),
    "M has inf": (P_OK, np.array([np.inf, 1.7]), "NaN or infinite"),
    "M is negative": (P_OK, np.array([-1.0, 4.0]), "non-negative"),
}


@pytest.mark.parametrize("solver", SOLVERS, ids=lambda f: f.__name__)
@pytest.mark.parametrize(
    ("P", "M", "message"), BAD_INPUTS.values(), ids=list(BAD_INPUTS)
)
def test_rejects_malformed_input(solver, P, M, message):
    with pytest.raises(CalibrationError, match=message):
        solver(P, M)


def test_soft_rejects_negative_penalties():
    with pytest.raises(CalibrationError, match="non-negative"):
        calibrate_soft(P_OK, M_OK, lam_m=-1.0)


# Rows summing to 1 force the column total to N = 3, so M summing to 3.3
# leaves no feasible matrix.
M_INFEASIBLE = M_OK * 1.1


def test_dykstra_warns_and_raises_when_sum_of_m_is_not_n():
    with (
        pytest.warns(UserWarning, match="must equal N"),
        pytest.raises(CalibrationError),
    ):
        calibrate_dykstra(P_OK, M_INFEASIBLE, max_iters=50)


@pytest.mark.parametrize("N", [30, 1000])
def test_ipf_convergence_means_the_returned_columns_are_within_tol(N):
    # A per-row gap below tol let the column error after the final row
    # normalization reach 0.029 at N = 1000, tol = 1e-3, yet report success.
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(3), size=N)
    M = P.sum(axis=0) * np.linspace(1.3, 0.7, 3)
    result = calibrate_ipf(P, M * N / M.sum(), tol=1e-3)
    assert result.converged
    assert result.max_col_error < 1e-3


def test_ipf_converges_at_a_loose_tolerance():
    # The marginal test is part of the stopping rule at the caller's tol, not
    # a fixed 1e-6 applied after the loop has already stopped.
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(3), size=30)
    M = P.sum(axis=0) * np.linspace(1.3, 0.7, 3)
    result = calibrate_ipf(P, M * 30 / M.sum(), tol=1e-3)
    assert result.converged
    assert result.max_col_error < 1e-2


def test_ipf_warns_and_reports_failure_when_sum_of_m_is_not_n():
    # IPF settles on a fixed point whose columns miss M; that must not be
    # reported as convergence.
    with pytest.warns(UserWarning, match="must equal N"):
        result = calibrate_ipf(P_OK, M_INFEASIBLE)
    assert not result.converged
    assert result.max_col_error == pytest.approx(0.17)
