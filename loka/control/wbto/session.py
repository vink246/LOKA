"""Optional whole-body terrain session for :class:`LocomotionController`.

Engaged only when ``TERRAIN_STACK`` is ``wbto`` or ``auto``, Crocoddyl is
importable, the terrain belief warrants it (any raised curb / stairs / gap),
and the gait has an active world goal. Otherwise the classical / heuristic
plant runs unchanged.

Mass / CoM / gear belief edits still write to ``controller.robot.model``; the
Pinocchio twin used here is built from the MJCF path and is not yet synced
to those edits (terrain volumes are the belief channel that drives planning).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from loka.control import terrain as terrain_mod
from loka.control.terrain import TerrainBelief, ensure_stance_support, terrain_wants_wbto

if TYPE_CHECKING:
    from loka.control.locomotion import LocomotionController

_CROCO_OK: bool | None = None


def crocoddyl_available() -> bool:
    global _CROCO_OK
    if _CROCO_OK is None:
        try:
            import crocoddyl  # noqa: F401
            import pinocchio  # noqa: F401

            _CROCO_OK = True
        except ImportError:
            _CROCO_OK = False
    return bool(_CROCO_OK)


def wbto_stack_requested() -> bool:
    return terrain_mod.TERRAIN_STACK in ("wbto", "auto")


def should_use_wbto(belief: TerrainBelief) -> bool:
    """Whether this tick's terrain belief should hand control to WBTO."""
    if not wbto_stack_requested() or not crocoddyl_available():
        return False
    if terrain_mod.TERRAIN_STACK == "wbto":
        return bool(belief.active)
    return terrain_wants_wbto(belief)


class WbtoSession:
    """Plans once to the gait goal and tracks with whole-body FDDP."""

    def __init__(self, controller: "LocomotionController") -> None:
        self.controller = controller
        self.active = False
        self._mpc = None
        self._wb = None
        self._builder = None
        self._per_knot = 1
        self._tick = 0
        self._belief_version = -1
        self._goal = np.zeros(2)
        self._warned_missing = False
        self.torque: np.ndarray | None = None
        self.solve_ms = 0.0
        self.last_error: str | None = None

    def reset(self) -> None:
        self.active = False
        self._mpc = None
        self._wb = None
        self._builder = None
        self._tick = 0
        self._belief_version = -1
        self.torque = None
        self.solve_ms = 0.0
        self.last_error = None

    def maybe_step(self, qpos: np.ndarray, qvel: np.ndarray, started: float) -> bool:
        """If WBTO owns this tick, set ``self.torque`` and return True."""
        ctrl = self.controller
        belief = ctrl.terrain
        gait = ctrl.gait.config
        goal_on = float(gait.goal_active) >= 0.5
        # Stay off once the plant has latched arrival (classical holds the stand).
        want = (
            should_use_wbto(belief)
            and goal_on
            and not bool(getattr(ctrl.gait, "_goal_arrived", False))
        )

        if wbto_stack_requested() and not crocoddyl_available():
            if not self._warned_missing:
                print(
                    "[wbto] TERRAIN_STACK wants whole-body terrain control, but "
                    "crocoddyl/pinocchio are not installed; staying on classical."
                )
                self._warned_missing = True
            self._release()
            self.torque = None
            return False

        if not want:
            self._release()
            self.torque = None
            return False

        goal = np.array([float(gait.goal_x), float(gait.goal_y)], dtype=float)
        need_plan = (
            not self.active
            or self._mpc is None
            or belief.version != self._belief_version
            or float(np.linalg.norm(goal - self._goal)) > 0.15
        )
        if need_plan:
            if not self._engage(qpos, qvel, goal):
                self.torque = None
                return False

        assert self._mpc is not None and self._wb is not None
        mpc = self._mpc
        wb = self._wb
        if self._tick % self._per_knot == 0:
            mpc.update(wb.state_from_mujoco(qpos[: wb.mj_model.nq], qvel[: wb.mj_model.nv]))
        alpha = (self._tick % self._per_knot) / self._per_knot
        tau = mpc.actuator_torque(qpos, qvel, alpha)
        self._tick += 1
        self.torque = tau
        self.solve_ms = (time.perf_counter() - started) * 1e3
        self._publish_telemetry(qpos, qvel, tau, started)

        # Arrival: plan done and pelvis inside the gait tolerance. Keep this
        # tick's torque; classical resumes on the next call after release.
        if mpc.finished:
            here = np.asarray(qpos[:2], dtype=float)
            if float(np.linalg.norm(here - self._goal)) <= float(gait.goal_tolerance):
                ctrl.gait._goal_arrived = True
                ctrl.gait._arrival_stand = True
                gait.mode = 0.0  # MODE_STAND
                self._release()
        return True

    def _engage(self, qpos: np.ndarray, qvel: np.ndarray, goal: np.ndarray) -> bool:
        from loka.control.wbto.controller import MpcParams, WholeBodyMpc
        from loka.control.wbto.footsteps import plan_contacts
        from loka.control.wbto.model import WholeBodyModel
        from loka.control.wbto.ocp import OcpBuilder, OcpParams, build_knots

        ctrl = self.controller
        try:
            if self._wb is None:
                self._wb = WholeBodyModel(ctrl.config.model_path)
                self._builder = OcpBuilder(self._wb, OcpParams())
            wb = self._wb
            builder = self._builder
            assert builder is not None
            nq, nv = wb.mj_model.nq, wb.mj_model.nv
            x0 = wb.state_from_mujoco(qpos[:nq], qvel[:nv])
            feet = np.array([m.translation for m in wb.foot_placements(x0[: wb.nq])])
            # Vision near-pads often start ahead of the robot; without support
            # under the soles WBTO plans floor-height contacts and collapses.
            if ensure_stance_support(ctrl.terrain, feet):
                print("[wbto] extended belief under stance soles")
            plan = plan_contacts(ctrl.terrain, feet, goal)
            if not plan.steps:
                self.last_error = "empty contact plan"
                self._release()
                return False
            knots = build_knots(plan, wb.com(x0[: wb.nq]), builder.p)
            self._mpc = WholeBodyMpc(
                wb,
                builder,
                knots,
                plan.heading,
                MpcParams(latency=1, max_iter=2, cold_iter=80),
            )
            self._per_knot = max(1, int(round(builder.p.dt / ctrl.config.control_dt)))
            self._tick = 0
            self._belief_version = ctrl.terrain.version
            self._goal = goal.copy()
            self.active = True
            self.last_error = None
            # Hand the gait clock a stand so classical state does not fight us.
            ctrl.gait.config.mode = 0.0
            print(
                f"[wbto] engaged: {len(plan.steps)} steps, "
                f"plan {plan.duration:.1f}s → goal ({goal[0]:.2f}, {goal[1]:.2f})"
            )
            return True
        except Exception as exc:  # noqa: BLE001 — keep classical alive
            self.last_error = str(exc)
            print(f"[wbto] engage failed ({exc}); staying on classical")
            self._release()
            return False

    def _release(self) -> None:
        if self.active:
            print("[wbto] released → classical")
        self.active = False
        self._mpc = None

    def _publish_telemetry(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        tau: np.ndarray,
        started: float,
    ) -> None:
        from loka.control.locomotion import LocomotionTelemetry
        from loka.control.robot import quat_to_rpy

        ctrl = self.controller
        robot = ctrl.robot
        robot.update(qpos, qvel)
        dynamics = robot.dynamics(np.asarray(qvel, dtype=float))
        rpy = quat_to_rpy(np.asarray(qpos[3:7], dtype=float))
        n_c = robot.num_contacts
        ctrl.telemetry = LocomotionTelemetry(
            com=dynamics.com,
            com_reference=dynamics.com.copy(),
            com_error=np.zeros(3),
            com_velocity=dynamics.com_vel,
            com_velocity_reference=np.zeros(3),
            rpy=rpy,
            torque=tau,
            contact_forces=np.zeros((n_c, 3)),
            desired_forces=np.zeros((n_c, 3)),
            contact_mask=np.ones(n_c, dtype=bool),
            mpc_cost=None,
            solve_ms=(time.perf_counter() - started) * 1e3,
            walking=True,
            gait_phase=0.0,
            yaw_ref=float(rpy[2]),
            heading_goal=float(ctrl.gait.config.heading),
            heading_error=0.0,
        )
