"""Small OSQP front-end for the fixed-shape QPs solved every control tick.

Both control layers re-solve the *same* problem shape hundreds of times per
second with new numbers, so the expensive parts of OSQP -- allocating and
symbolically factorising the KKT system -- should happen exactly once. That
requires the sparsity pattern to be pinned up front, which is what
:class:`Sparsity` is for: it records which entries a matrix is *allowed* to
occupy and then repackages new numbers into that pattern with a single fancy
index.

Matrices that never change (the MPC's friction constraints, the WBC's diagonal
cost) are handed over once at setup and skipped on every later solve.
"""

from __future__ import annotations

import numpy as np
import osqp
import scipy.sparse as sparse
from osqp import SolverStatus

#: Statuses whose primal iterate is worth using. "Solved inaccurate" means OSQP
#: ran out of iterations near a solution, which is still a usable command.
_ACCEPTED_STATUS = frozenset(
    {SolverStatus.OSQP_SOLVED, SolverStatus.OSQP_SOLVED_INACCURATE}
)


class Sparsity:
    """A fixed CSC pattern that new dense values can be poured into."""

    def __init__(self, mask: np.ndarray) -> None:
        mask = np.asarray(mask, dtype=bool)
        n_rows, n_cols = mask.shape
        self.shape = (n_rows, n_cols)
        columns = [np.flatnonzero(mask[:, j]) for j in range(n_cols)]
        self.indptr = np.concatenate(
            ([0], np.cumsum([len(c) for c in columns]))
        ).astype(np.int32)
        self.indices = np.concatenate(columns).astype(np.int32) if n_cols else np.zeros(0, np.int32)
        self._flat = np.concatenate(
            [c + j * n_rows for j, c in enumerate(columns)]
        ).astype(np.int64)

    @classmethod
    def dense(cls, n_rows: int, n_cols: int) -> "Sparsity":
        return cls(np.ones((n_rows, n_cols), dtype=bool))

    @classmethod
    def upper_triangular(cls, n: int) -> "Sparsity":
        return cls(np.triu(np.ones((n, n), dtype=bool)))

    @classmethod
    def diagonal(cls, n: int) -> "Sparsity":
        return cls(np.eye(n, dtype=bool))

    def values(self, dense: np.ndarray) -> np.ndarray:
        return np.asarray(dense, dtype=float).ravel(order="F")[self._flat]

    def matrix(self, dense: np.ndarray) -> sparse.csc_matrix:
        return sparse.csc_matrix(
            (self.values(dense), self.indices, self.indptr), shape=self.shape
        )


class QP:
    """``min ½ zᵀPz + qᵀz`` subject to ``l ≤ Az ≤ u``.

    ``hessian_pattern`` / ``constraint_pattern`` may be ``None`` for a matrix
    that is fixed for the lifetime of the problem, in which case the value
    passed to the first :meth:`solve` is reused forever.

    A failed solve returns the previous solution rather than raising, so a
    momentarily infeasible instant (a foot leaving the ground, say) degrades
    into a held command instead of killing the control loop.
    """

    def __init__(
        self,
        name: str,
        *,
        hessian_pattern: Sparsity | None = None,
        constraint_pattern: Sparsity | None = None,
        **settings,
    ) -> None:
        self.name = name
        self.hessian_pattern = hessian_pattern
        self.constraint_pattern = constraint_pattern
        self.settings = {
            "verbose": False,
            "eps_abs": 1e-4,
            "eps_rel": 1e-4,
            "max_iter": 200,
            "polishing": False,
            "warm_starting": True,
            **settings,
        }
        self._solver: osqp.OSQP | None = None
        self._solution: np.ndarray | None = None
        self.failures = 0
        self.last_status = "unsolved"

    def solve(
        self,
        hessian: np.ndarray,
        gradient: np.ndarray,
        constraint: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray:
        gradient = np.asarray(gradient, dtype=float)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

        if self._solver is None:
            hessian_sp = (
                self.hessian_pattern.matrix(hessian)
                if self.hessian_pattern is not None
                else sparse.csc_matrix(np.triu(hessian))
            )
            constraint_sp = (
                self.constraint_pattern.matrix(constraint)
                if self.constraint_pattern is not None
                else sparse.csc_matrix(constraint)
            )
            self._solver = osqp.OSQP()
            self._solver.setup(
                P=hessian_sp,
                q=gradient,
                A=constraint_sp,
                l=lower,
                u=upper,
                **self.settings,
            )
        else:
            updates = {"q": gradient, "l": lower, "u": upper}
            if self.hessian_pattern is not None:
                updates["Px"] = self.hessian_pattern.values(hessian)
            if self.constraint_pattern is not None:
                updates["Ax"] = self.constraint_pattern.values(constraint)
            self._solver.update(**updates)

        # Failures are handled below by inspecting the status; an exception in
        # the middle of a control tick would be strictly worse than a held
        # command. Passing this explicitly also pins the upcoming default flip.
        result = self._solver.solve(raise_error=False)
        solution = np.asarray(result.x, dtype=float)
        # On an infeasible or truncated solve OSQP hands back its last iterate,
        # which is finite but can be astronomically large -- checking the
        # status is the only way to tell that apart from a real answer.
        status = result.info.status_val
        if (
            status not in _ACCEPTED_STATUS
            or solution.shape != (hessian.shape[0],)
            or not np.all(np.isfinite(solution))
        ):
            self.failures += 1
            self.last_status = getattr(result.info, "status", "unknown")
            if self._solution is None:
                return np.zeros(hessian.shape[0])
            return self._solution.copy()

        self.last_status = "solved"
        self._solution = solution
        return solution.copy()
