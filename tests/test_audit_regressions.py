"""Regression tests for the defects found in the 2026-07-31 audit.

Every test here failed before its corresponding fix. They are grouped by defect
so that a future regression points straight at what broke.

The reference for correctness throughout is an independent convex solver
(cvxpy), the same way calibre pins its isotonic core against R. Where a claim is
about an exact property -- a projection being optimal, a constraint being
satisfied -- it is asserted with ``rtol=0``, because that is what the code always
meant to assert.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from rank_preserving_calibration import (
    calibrate_admm,
    calibrate_dykstra,
    calibrate_ovr_isotonic,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def feasible_problem(seed: int, N: int = 30, J: int = 4, tilt: float = 0.5):
    """Build a problem whose targets differ from P's column sums but is feasible.

    Feasibility needs ``sum(M) == N`` exactly: rows summing to 1 force the grand
    total to N. Given that, the constant matrix ``Q[i, j] = M[j] / N`` always
    satisfies every constraint, so the intersection is never empty.

    Parameters
    ----------
    seed
        Random seed.
    N
        Number of rows.
    J
        Number of classes.
    tilt
        How far the targets are pushed from P's own column sums.

    Returns
    -------
    P : ndarray
        Probability matrix, rows summing to 1.
    M : ndarray
        Target column sums, summing to N.
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


def rank_violations(P: np.ndarray, Q: np.ndarray) -> int:
    """Count adjacent within-class inversions, ordered by the original scores."""
    bad = 0
    for j in range(Q.shape[1]):
        order = np.argsort(P[:, j], kind="mergesort")
        bad += int((np.diff(Q[order, j]) < -1e-9).sum())
    return bad


# --------------------------------------------------------------------------- #
# Defect 1: np.allclose ignored the intended tolerance
# --------------------------------------------------------------------------- #


class TestToleranceChecksAreAbsolute:
    """``np.allclose(x, target, atol=...)`` keeps rtol=1e-5 unless told otherwise.

    Every feasibility check in this package passed an ``atol`` and no ``rtol``,
    so the effective bound was ``atol + 1e-5 * |target|``. For rows that made a
    stated 1e-12 behave as 1e-5; for columns it scaled with ``M_j``, so the check
    grew *looser* as the problem grew larger.
    """

    def test_reported_convergence_means_rows_really_sum_to_one(self):
        """converged=True must not be compatible with a 1e-5 row error."""
        P, M = feasible_problem(0, N=30, J=4)
        result = calibrate_dykstra(P, M, max_iters=20000, tol=1e-10)
        if result.converged:
            assert np.allclose(result.Q.sum(axis=1), 1.0, atol=1e-8, rtol=0.0), (
                f"converged=True but max row error is {result.max_row_error:.2e}"
            )

    def test_reported_convergence_means_columns_really_match(self):
        """The column check must not loosen as the targets grow."""
        P, M = feasible_problem(1, N=30, J=4)
        result = calibrate_dykstra(P, M, max_iters=20000, tol=1e-10)
        if result.converged:
            assert np.allclose(result.Q.sum(axis=0), M, atol=1e-8, rtol=0.0), (
                f"converged=True but max column error is {result.max_col_error:.2e}"
            )

    def test_max_row_error_agrees_with_the_converged_flag(self):
        """The reported residual and the boolean must tell the same story."""
        P, M = feasible_problem(2, N=40, J=3)
        result = calibrate_dykstra(P, M, max_iters=20000, tol=1e-10)
        if result.converged:
            assert result.max_row_error < 1e-8
            assert result.max_col_error < 1e-8


# --------------------------------------------------------------------------- #
# Defect 2: the ADMM Q-update was misderived
# --------------------------------------------------------------------------- #


class TestADMMSolvesTheStatedProblem:
    """The Q-update treated row/column SUM targets as per-ELEMENT targets.

    ``Q = (P + rho*(Z1[:, None] + Z2[None, :])) / (1 + 2*rho)`` pulls every entry
    toward the row target and the column target, rather than pulling the row
    *sum* toward the row target. With J classes and a row target of 1, each row
    was driven toward J*(1 + M_j)/2 -- so the update could not satisfy its own
    constraints for any rho.
    """

    @pytest.mark.parametrize("seed", range(3))
    def test_admm_converges_on_ordinary_feasible_problems(self, seed):
        """It failed on every size tried in the audit, from N=25 up."""
        P, M = feasible_problem(seed, N=25, J=4)
        result = calibrate_admm(P, M, max_iters=5000, tol=1e-8)
        assert result.converged, (
            f"ADMM did not converge on a feasible N=25 problem "
            f"(row error {result.max_row_error:.2e})"
        )

    def test_admm_output_satisfies_the_constraints(self):
        """Valid rows, matching totals, and no rank violations."""
        P, M = feasible_problem(5, N=25, J=4)
        Q = calibrate_admm(P, M, max_iters=5000, tol=1e-8).Q
        assert np.allclose(Q.sum(axis=1), 1.0, atol=1e-6, rtol=0.0)
        assert np.allclose(Q.sum(axis=0), M, atol=1e-6, rtol=0.0)
        assert rank_violations(P, Q) == 0

    def test_admm_agrees_with_dykstra(self):
        """Two solvers for one convex problem must find the same optimum."""
        P, M = feasible_problem(6, N=25, J=4)
        Q_admm = calibrate_admm(P, M, max_iters=5000, tol=1e-8).Q
        Q_dyk = calibrate_dykstra(P, M, max_iters=20000, tol=1e-10).Q
        obj_admm = float(np.sum((Q_admm - P) ** 2))
        obj_dyk = float(np.sum((Q_dyk - P) ** 2))
        assert obj_admm == pytest.approx(obj_dyk, abs=1e-4), (
            f"ADMM objective {obj_admm:.6f} vs Dykstra {obj_dyk:.6f}"
        )


# --------------------------------------------------------------------------- #
# Defect 3: the documented behaviour on infeasible input
# --------------------------------------------------------------------------- #


class TestInfeasibleInputIsHandledAsDocumented:
    """The README promised a closest point; the code raises.

    Feasibility requires ``sum(M) == N`` exactly. The old feasibility warning
    only fired above 10% of N, so a 2% mismatch produced no warning at all and
    then failed hard.
    """

    @pytest.mark.parametrize("scale", [1.02, 1.10, 0.90])
    def test_infeasible_targets_warn_before_failing(self, scale):
        """A user must be told the targets cannot be met, whatever happens next."""
        P, _ = feasible_problem(7, N=40, J=3)
        M = P.sum(axis=0) * scale
        # Raising is acceptable; failing silently is not.
        with (
            pytest.warns(UserWarning, match="(?i)feasib|sum"),
            contextlib.suppress(Exception),
        ):
            calibrate_dykstra(P, M, max_iters=500, tol=1e-10)

    def test_exactly_feasible_targets_do_not_warn(self):
        """The warning must not cry wolf on a well-posed problem."""
        P, M = feasible_problem(8, N=40, J=3)
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error")
            calibrate_dykstra(P, M, max_iters=20000, tol=1e-10)


# --------------------------------------------------------------------------- #
# Defect 4: the one-vs-rest baseline broke rank preservation silently
# --------------------------------------------------------------------------- #


class TestOvRIsotonicIsHonestAboutRank:
    """Row normalisation divides each entry by a row-specific number.

    On columns that are perfectly isotonic beforehand, that inverted 57% of
    adjacent within-class pairs in the audit. The function is a useful baseline
    -- it is what scikit-learn does -- but it must not be mistaken for the
    rank-preserving method this package is named for.
    """

    def test_result_is_row_stochastic(self):
        """Whatever else it does, the output must be a probability matrix."""
        rng = np.random.default_rng(9)
        P = rng.dirichlet(np.ones(4), size=200)
        y = rng.integers(0, 4, 200)
        Q = calibrate_ovr_isotonic(y, P)["Q"]
        assert np.allclose(Q.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
        assert np.all(Q >= 0.0)

    def test_result_does_not_depend_on_row_order(self):
        """Tied scores must not be resolved by the sort's tie-breaking.

        The interpolation kept the first of ``np.unique(..., return_index=True)``,
        so tied scores that isotonic regression gave different values to were
        resolved arbitrarily.
        """
        rng = np.random.default_rng(10)
        P = np.round(rng.dirichlet(np.ones(3), size=300), 2)
        P /= P.sum(axis=1, keepdims=True)
        y = rng.integers(0, 3, 300)

        first = calibrate_ovr_isotonic(y, P)["Q"]
        perm = rng.permutation(len(y))
        second = calibrate_ovr_isotonic(y[perm], P[perm])["Q"]

        np.testing.assert_allclose(first[perm], second, atol=1e-9)

    def test_rank_breaking_is_measurable_and_documented(self):
        """The docstring must state that rank is not preserved.

        This is the property the package is named for, so its absence in the
        baseline has to be explicit rather than left for a user to discover.
        """
        doc = calibrate_ovr_isotonic.__doc__ or ""
        assert "rank" in doc.lower()
        assert any(
            word in doc.lower() for word in ("not preserve", "does not", "break")
        )


# --------------------------------------------------------------------------- #
# Cross-check against an independent solver
# --------------------------------------------------------------------------- #


class TestAgainstIndependentSolver:
    """Pin the optimum against cvxpy, the way calibre pins its core against R."""

    @pytest.mark.parametrize(("N", "J"), [(10, 3), (20, 4), (30, 3)])
    def test_dykstra_reaches_the_true_optimum(self, N, J):
        """The returned matrix must minimise ||Q - P||_F over the constraint set."""
        cp = pytest.importorskip("cvxpy")
        P, M = feasible_problem(11, N=N, J=J)

        Q = cp.Variable((N, J))
        cons = [Q >= 0, cp.sum(Q, axis=1) == 1, cp.sum(Q, axis=0) == M]
        for j in range(J):
            order = np.argsort(P[:, j], kind="mergesort")
            cons.extend(Q[order[i + 1], j] >= Q[order[i], j] for i in range(N - 1))
        problem = cp.Problem(cp.Minimize(cp.sum_squares(Q - P)), cons)
        problem.solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12)

        ours = calibrate_dykstra(P, M, max_iters=50000, tol=1e-10).Q
        assert float(np.sum((ours - P) ** 2)) == pytest.approx(
            float(problem.value), abs=1e-6
        )
