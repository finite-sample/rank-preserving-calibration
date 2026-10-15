"""Projection building blocks checked against implementations that share no code.

The row-simplex projection and the column isotonic step are what every solver
in the package is assembled from, so each is pinned against an independent
solver: cvxpy for the convex programs, scipy's PAVA for isotonic regression.
"""

import numpy as np
import pytest
from scipy.optimize import isotonic_regression

from rank_preserving_calibration import calibrate_dykstra
from rank_preserving_calibration.calibration import (
    _isotonic_regression,
    _project_row_simplex,
)

cp = pytest.importorskip("cvxpy")


def _solve(objective, constraints):
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12)
    return problem


class TestRowSimplexProjection:
    @pytest.mark.parametrize("seed", range(5))
    def test_matches_qp(self, seed):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(30, 5)) * 2
        X[0] = -3.0  # all negative and all tied
        X[1] = [0.4, 0.4, 0.1, 0.1, 0.0]  # ties above the threshold
        X[2] = [5.0, -5.0, 0.0, 0.0, 0.0]  # a single vertex

        Q = cp.Variable(X.shape)
        _solve(cp.sum_squares(Q - X), [Q >= 0, cp.sum(Q, axis=1) == 1])

        ours = _project_row_simplex(X)
        # The interior-point solver stops ~1e-5 short of the boundary, so
        # compare objectives: ours must be feasible and at least as close to X.
        assert ours.min() >= 0.0
        np.testing.assert_allclose(ours.sum(axis=1), 1.0, atol=1e-12)
        assert np.sum((ours - X) ** 2) <= np.sum((Q.value - X) ** 2) + 1e-9
        np.testing.assert_allclose(ours, Q.value, atol=1e-4)

    def test_rows_already_on_the_simplex_are_unchanged(self):
        P = np.random.default_rng(0).dirichlet(np.ones(4), size=50)
        np.testing.assert_allclose(_project_row_simplex(P), P, atol=1e-15)


class TestIsotonicRegression:
    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("ties", ["stable", "group"])
    def test_unweighted_matches_scipy(self, seed, ties):
        rng = np.random.default_rng(seed)
        # Rounding creates runs of exactly equal values for ties="group" to pool.
        y = np.round(rng.normal(size=60), 1)
        np.testing.assert_allclose(
            _isotonic_regression(y, ties=ties), isotonic_regression(y).x, atol=1e-12
        )

    @pytest.mark.parametrize(
        "weights",
        [
            np.array([1, 3, 2, 1, 5, 1, 2, 4]),  # integer path
            np.array([0.5, 1.5, 2.25, 1.0, 0.1, 3.0, 0.7, 1.2]),  # float path
        ],
    )
    @pytest.mark.parametrize("ties", ["stable", "group"])
    def test_weighted_matches_scipy(self, weights, ties):
        y = np.array([3.0, 1.0, 1.0, 2.0, 0.5, 0.5, 4.0, 3.5])
        expected = isotonic_regression(y, weights=weights).x
        np.testing.assert_allclose(
            _isotonic_regression(y, ties=ties, weights=weights), expected, atol=1e-12
        )

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"rtol": -1.0}, "rtol"),
            ({"ties": "average"}, "ties"),
            ({"weights": np.ones(3)}, "same shape"),
            ({"weights": np.array([1.0, 0.0, 1.0, 1.0])}, "positive"),
        ],
    )
    def test_rejects_bad_arguments(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            _isotonic_regression(np.array([2.0, 1.0, 3.0, 0.0]), **kwargs)


class TestGroupedTies:
    """ties="group": rows with equal scores in a column get equal calibrated values."""

    @staticmethod
    def _problem():
        rng = np.random.default_rng(3)
        N, J = 24, 3
        # Scores on a coarse grid so every column has several tied runs.
        P = rng.dirichlet(np.ones(J), size=N).round(1)
        P[:, -1] = 1.0 - P[:, :-1].sum(axis=1)
        P = np.clip(P, 0.0, None)
        P /= P.sum(axis=1, keepdims=True)
        M = P.sum(axis=0) * np.array([1.15, 1.0, 0.85])
        return P, M * N / M.sum()

    @staticmethod
    def _oracle(P, M):
        N, J = P.shape
        Q = cp.Variable((N, J))
        cons = [Q >= 0, cp.sum(Q, axis=1) == 1, cp.sum(Q, axis=0) == M]
        for j in range(J):
            scores = np.unique(P[:, j])
            groups = [np.flatnonzero(P[:, j] == s) for s in scores]
            for g in groups:
                cons.extend(Q[g[k], j] == Q[g[0], j] for k in range(1, len(g)))
            cons.extend(
                Q[groups[k + 1][0], j] >= Q[groups[k][0], j]
                for k in range(len(groups) - 1)
            )
        _solve(cp.sum_squares(Q - P), cons)
        return Q.value

    def test_problem_has_ties(self):
        P, _ = self._problem()
        assert all(len(np.unique(P[:, j])) < len(P) for j in range(P.shape[1]))

    def test_matches_qp(self):
        P, M = self._problem()
        expected = self._oracle(P, M)
        Q = calibrate_dykstra(P, M, ties="group", max_iters=20000, tol=1e-10).Q
        np.testing.assert_allclose(Q, expected, atol=1e-6)
