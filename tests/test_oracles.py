"""Projection building blocks checked against implementations that share no code.

The row-simplex projection and the column isotonic step are what every solver
in the package is assembled from, so each is pinned against an independent
solver: cvxpy for the convex programs. Isotonic regression is scipy's PAVA
inside the package, so it too is checked against cvxpy, not against scipy.
"""

import numpy as np
import pytest

from rank_preserving_calibration import calibrate_dykstra
from rank_preserving_calibration._isotonic import _isotonic_regression
from rank_preserving_calibration.calibration import _project_row_simplex
from rank_preserving_calibration.kl_nearly import _kl_pav_increasing

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


def _isotonic_oracle(y, w, kl=False):
    z = cp.Variable(y.size)
    loss = (
        cp.sum(cp.multiply(w, cp.kl_div(z, y)))
        if kl
        else cp.sum(cp.multiply(w, cp.square(z - y)))
    )
    cp.Problem(cp.Minimize(loss), [z[1:] >= z[:-1]]).solve(
        solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12
    )
    return z.value


class TestIsotonicRegression:
    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("weighted", [False, True])
    def test_matches_qp(self, seed, weighted):
        rng = np.random.default_rng(seed)
        # Rounding creates runs of exactly equal values.
        y = np.round(rng.normal(size=60), 1)
        w = rng.uniform(0.1, 3.0, y.size) if weighted else np.ones(y.size)
        ours = _isotonic_regression(y, w if weighted else None)
        np.testing.assert_allclose(ours, _isotonic_oracle(y, w), atol=1e-6)

    @pytest.mark.parametrize("seed", range(3))
    def test_kl_fit_matches_the_kl_program(self, seed):
        # Geometric-mean pooling minimises the generalised KL divergence
        # sum w (z log(z / y) - z + y), which is cvxpy's kl_div; dropping the
        # -z + y terms would move the optimum by a factor of e.
        rng = np.random.default_rng(seed)
        y = rng.uniform(0.05, 1.0, 25)
        w = rng.uniform(0.5, 3.0, y.size)
        np.testing.assert_allclose(
            _kl_pav_increasing(y, w), _isotonic_oracle(y, w, kl=True), atol=1e-5
        )

    @pytest.mark.parametrize(
        ("weights", "message"),
        [(np.ones(3), "same shape"), (np.array([1.0, 0.0, 1.0, 1.0]), "positive")],
    )
    def test_rejects_bad_weights(self, weights, message):
        with pytest.raises(ValueError, match=message):
            _isotonic_regression(np.array([2.0, 1.0, 3.0, 0.0]), weights)


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
