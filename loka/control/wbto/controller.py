"""Receding-horizon whole-body controller over a contact plan.

Every knot period the horizon slides one knot along the plan and FDDP is
re-solved from the measured state, warm-started from the previous solution.
Between solves the torque is the first control of the horizon plus the
solver's Riccati feedback on the deviation from the planned state, evaluated
at the control rate. Past the end of the plan the last knot (both feet
planted, CoM between them) repeats, so the controller holds the final stance.

The locked joints (arms) are held at their stand pose by a joint PD with
gravity compensation from the belief model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import crocoddyl
import mujoco
import numpy as np

from loka.control.wbto.model import WholeBodyModel
from loka.control.wbto.ocp import Knot, OcpBuilder


@dataclass
class MpcParams:
    horizon: int = 60  # knots; 1.2 s at the default 20 ms knot
    max_iter: int = 2  # FDDP iterations per knot period
    cold_iter: int = 100  # iterations of the first solve
    threads: int = 0
    feedback: bool = True  # Riccati feedback between solves
    #: Knot periods a solve may take. With 1, the torque of a knot period comes from a solve
    #: started one period earlier from the predicted state, so the solver has a full period of
    #: wall time; with 0 the solution is applied to the state it was computed from.
    latency: int = 0
    feedback_scale: float = 1.0
    predict: bool = True
    arm_kp: float = 60.0
    arm_kd: float = 3.0


class WholeBodyMpc:
    """Tracks a knot sequence on the plant. One instance per contact plan."""

    def __init__(
        self,
        wb: WholeBodyModel,
        builder: OcpBuilder,
        knots: list[Knot],
        yaw: float = 0.0,
        params: MpcParams | None = None,
    ) -> None:
        self.wb = wb
        self.builder = builder
        self.knots = knots
        self.yaw = float(yaw)
        self.p = params or MpcParams()
        self.state = builder.state
        self.tick = -1  # index of the knot period in progress
        self.problem = None
        self.solver = None
        self.xs: list[np.ndarray] = []
        self.us: list[np.ndarray] = []
        self.gain = np.zeros((wb.nu, self.state.ndx))
        self.head = 0  # index of the knot at the head of the horizon
        self._active = None  # (x_from, x_to, u, K) in effect during the current knot period
        self._pending = None  # the same for the next period, when the solve runs one period ahead
        self.solve_ms: list[float] = []
        self.iterations: list[int] = []
        self.failures = 0
        self._belief = mujoco.MjData(wb.mj_model)

    # -- horizon ----------------------------------------------------------

    def knot(self, index: int) -> Knot:
        return self.knots[min(index, len(self.knots) - 1)]

    @property
    def finished(self) -> bool:
        return self.tick >= len(self.knots) - 1

    def _start(self, x: np.ndarray) -> None:
        n = self.p.horizon
        running = [self.builder.model(self.knot(i), self.yaw) for i in range(n)]
        terminal = self.builder.model(self.knot(n), self.yaw, terminal=True)
        self.problem = crocoddyl.ShootingProblem(x.copy(), running, terminal)
        if self.p.threads:
            self.problem.nthreads = int(self.p.threads)
        self.solver = crocoddyl.SolverFDDP(self.problem)
        self.solver.th_stop = 1e-6
        xs = [x.copy() for _ in range(n + 1)]
        us = self.problem.quasiStatic(xs[:-1])
        self._solve(xs, us, self.p.cold_iter)

    def _solve(self, xs, us, iterations: int) -> None:
        t0 = time.perf_counter()
        self.solver.solve(xs, us, iterations, False)
        self.solve_ms.append(1e3 * (time.perf_counter() - t0))
        self.iterations.append(int(self.solver.iter))
        new_xs = [np.array(v) for v in self.solver.xs]
        new_us = [np.array(v) for v in self.solver.us]
        if not (np.all(np.isfinite(new_us[0])) and np.all(np.isfinite(new_xs[1]))):
            # Keep the previous plan, advanced one knot.
            self.failures += 1
            if self.xs:
                self.xs = self.xs[1:] + [self.xs[-1]]
                self.us = self.us[1:] + [self.us[-1]]
            return
        self.xs, self.us = new_xs, new_us
        self.gain = np.array(self.solver.K[0])

    def _policy(self):
        """The first knot period of the current solution: states at its ends, torque, gain."""
        return self.xs[0], self.xs[1], self.us[0], self.gain

    def _advance(self, x: np.ndarray) -> None:
        """Slide the horizon one knot and re-solve with ``x`` as its first state."""
        self.head += 1
        n = self.p.horizon
        model = self.builder.model(self.knot(self.head + n - 1), self.yaw)
        self.problem.circularAppend(model, model.createData())
        self.problem.terminalModel = self.builder.model(
            self.knot(self.head + n), self.yaw, terminal=True
        )
        self.problem.x0 = x.copy()
        xs = [x.copy()] + self.xs[2:] + [self.xs[-1]]
        us = self.us[1:] + [self.us[-1]]
        self._solve(xs, us, self.p.max_iter)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        """The state one knot period ahead of ``x`` under the policy now in effect."""
        model, data = self.problem.runningModels[0], self.problem.runningDatas[0]
        model.calc(data, x, self.torque(x, 0.0))
        return np.array(data.xnext)

    def update(self, x: np.ndarray) -> None:
        """Start the next knot period from the measured state ``x``.

        Without latency the horizon is re-solved from ``x`` and its first
        torque applies at once. With latency the policy solved during the
        previous period takes effect, and the solve started now is for the
        period after this one, from the state the model predicts for its start.
        """
        self.tick += 1
        if self.problem is None:
            self._start(x)
            self._active = self._policy()
        elif not self.p.latency:
            self._advance(x)
            self._active = self._policy()
            return
        else:
            self._active = self._pending
        if self.p.latency:
            self._advance(self._predict(x) if self.p.predict else x)
            self._pending = self._policy()

    # -- torque -----------------------------------------------------------

    def torque(self, x: np.ndarray, alpha: float = 0.0) -> np.ndarray:
        """Joint torques of the reduced model, ``alpha`` in [0, 1) into the knot period."""
        x_from, x_to, u, gain = self._active
        u = u.copy()
        if self.p.feedback:
            ref = self.state.integrate(x_from, alpha * self.state.diff(x_from, x_to))
            u -= self.p.feedback_scale * (gain @ self.state.diff(ref, x))
        return np.clip(u, -self.wb.torque_limit, self.wb.torque_limit)

    def actuator_torque(self, qpos: np.ndarray, qvel: np.ndarray, alpha: float = 0.0) -> np.ndarray:
        """Torque for every actuator of the robot, in MuJoCo actuator order."""
        wb = self.wb
        x = wb.state_from_mujoco(qpos, qvel)
        out = np.zeros(wb.mj_model.nu)
        out[wb.act_of_joint] = self.torque(x, alpha)
        # Locked joints: PD about the stand pose plus the belief's gravity and Coriolis torque.
        d = self._belief
        d.qpos[:] = qpos[: wb.mj_model.nq]
        d.qvel[:] = qvel[: wb.mj_model.nv]
        mujoco.mj_kinematics(wb.mj_model, d)
        mujoco.mj_comPos(wb.mj_model, d)
        mujoco.mj_comVel(wb.mj_model, d)
        mujoco.mj_rne(wb.mj_model, d, 0, d.qfrc_bias)
        dof = wb.locked_dof
        arm = (
            self.p.arm_kp * (wb.locked_q0 - qpos[dof + 1])
            - self.p.arm_kd * qvel[dof]
            + d.qfrc_bias[dof]
        )
        limit = np.abs(wb.mj_model.actuator_ctrlrange[wb.locked_act, 1])
        out[wb.locked_act] = np.clip(arm, -limit, limit)
        return out
