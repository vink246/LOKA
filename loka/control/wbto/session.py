"""Optional whole-body terrain session for :class:`LocomotionController`.

Engaged only when ``TERRAIN_STACK`` is ``wbto`` or ``auto``, Crocoddyl is
importable, the terrain belief warrants it, and the gait has an active world
goal. Under ``auto``, that means gap-like beliefs (coplanar raised pads / a
raised→floor→raised trench); curb and stairs stay classical. Under ``wbto``,
any active belief engages. Otherwise the classical / heuristic plant runs
unchanged.

Mass / CoM / gear belief edits still write to ``controller.robot.model``; the
Pinocchio twin used here is built from the MJCF path and is not yet synced
to those edits (terrain volumes are the belief channel that drives planning).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from loka.control import terrain as terrain_mod
from loka.control.contact_estimate import SWING_CONTACT_N
from loka.control.terrain import (
    STANCE_HEIGHT_TOL,
    TerrainBelief,
    auto_wbto_stance_ready,
    carve_through_contact,
    ensure_stance_support,
    shrink_declared_gap_edges,
    terrain_wants_wbto,
)

if TYPE_CHECKING:
    from loka.control.locomotion import LocomotionController

_CROCO_OK: bool | None = None

#: Touchdown height error that triggers a proprioceptive refine event [m].
#: Mirrored from terrain_refine.REFINE_TOL without importing the agent package.
_REFINE_TOL = 0.01
#: Solver failure streak before releasing back to classical.
_FAIL_STREAK_RELEASE = 5
#: Consecutive saturated-torque ticks before release (together with solver
#: failures — saturation alone is common on a curb step-up).
_TORQUE_SAT_RELEASE = 40
#: Fraction of torque limit treated as saturated.
_TORQUE_SAT_FRAC = 0.98
#: Min ticks between belief-version cold replans while tracking.
_REPLAN_COOLDOWN_TICKS = 250  # 0.5 s at 500 Hz
#: Min ticks between through-contact / touchdown emits per foot.
_EVENT_COOLDOWN_TICKS = 100


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
        #: After a failed engage, stay classical for this many control ticks
        #: before retrying (avoids planning+logging every 2 ms on a bad belief).
        self._fail_cooldown_ticks = 250  # 0.5 s at 500 Hz
        self._fail_retry_at_tick = 0
        self._fail_belief_version: int | None = None
        self._fail_goal: np.ndarray | None = None
        self._global_tick = 0
        self._last_fail_print: str | None = None
        self._replan_after_tick = 0
        self._sat_ticks = 0
        self._event_leg_tick = [-10_000, -10_000]
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
        self._fail_retry_at_tick = 0
        self._fail_belief_version = None
        self._fail_goal = None
        self._global_tick = 0
        self._last_fail_print = None
        self._replan_after_tick = 0
        self._sat_ticks = 0
        self._event_leg_tick = [-10_000, -10_000]
        self.torque = None
        self.solve_ms = 0.0
        self.last_error = None

    def maybe_step(self, qpos: np.ndarray, qvel: np.ndarray, started: float) -> bool:
        """If WBTO owns this tick, set ``self.torque`` and return True."""
        ctrl = self.controller
        belief = ctrl.terrain
        gait = ctrl.gait.config
        self._global_tick += 1
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

        # Fresh kinematics for engage gates and proprioceptive events.
        ctrl.robot.update(qpos, qvel)
        feet = ctrl.robot.foot_center_positions()

        # ``auto``: classical owns floor→raised step-ups and seed-only stance
        # so touchdown refine can correct VLM heights before WBTO commits.
        if terrain_mod.TERRAIN_STACK == "auto" and not self.active:
            if not auto_wbto_stance_ready(belief, feet):
                # Still try stance repair so classical sees matching heights,
                # but do not engage until soles are raised on real support.
                ensure_stance_support(belief, feet)
                self.torque = None
                return False

        goal = np.array([float(gait.goal_x), float(gait.goal_y)], dtype=float)
        need_plan = (
            not self.active
            or self._mpc is None
            or belief.version != self._belief_version
            or float(np.linalg.norm(goal - self._goal)) > 0.15
        )
        if need_plan and self.active and belief.version != self._belief_version:
            if self._global_tick < self._replan_after_tick:
                need_plan = False
        if need_plan:
            same_fail = (
                self._fail_belief_version is not None
                and int(belief.version) == int(self._fail_belief_version)
                and self._fail_goal is not None
                and float(np.linalg.norm(goal - self._fail_goal)) <= 0.15
            )
            if same_fail and self._global_tick < self._fail_retry_at_tick:
                if not self.active:
                    self.torque = None
                    return False
                need_plan = False
            if need_plan:
                replacing = bool(self.active and self._mpc is not None)
                if not self._engage(qpos, qvel, goal, replace=replacing):
                    self._fail_belief_version = int(belief.version)
                    self._fail_goal = goal.copy()
                    self._fail_retry_at_tick = (
                        self._global_tick + int(self._fail_cooldown_ticks)
                    )
                    self._replan_after_tick = self._fail_retry_at_tick
                    if not self.active:
                        self.torque = None
                        return False
                    # Keep the previous contact plan; retry the replan later.
                else:
                    self._fail_belief_version = None
                    self._fail_goal = None
                    self._fail_retry_at_tick = 0
                    self._last_fail_print = None
                    self._replan_after_tick = self._global_tick + int(
                        _REPLAN_COOLDOWN_TICKS
                    )

        assert self._mpc is not None and self._wb is not None
        mpc = self._mpc
        wb = self._wb

        if self._maybe_release_on_tracking(qpos, qvel, feet):
            self.torque = None
            return False

        self._emit_proprio_events(feet)

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

    def _engage(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        goal: np.ndarray,
        *,
        replace: bool = False,
    ) -> bool:
        from loka.control.wbto.controller import MpcParams, WholeBodyMpc
        from loka.control.wbto.footsteps import plan_contacts
        from loka.control.wbto.model import WholeBodyModel
        from loka.control.wbto.ocp import OcpBuilder, OcpParams, build_knots

        ctrl = self.controller

        def _fail(msg: str) -> bool:
            self.last_error = msg
            full = f"[wbto] engage failed ({msg}); staying on classical"
            if full != self._last_fail_print:
                print(full)
                self._last_fail_print = full
            # A failed *replan* must not tear down a plan that was tracking.
            if not replace:
                self._release()
            return False

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
            repaired = ensure_stance_support(ctrl.terrain, feet)
            if shrink_declared_gap_edges(ctrl.terrain):
                repaired = True
                print("[wbto] opened sealed seam between coplanar pads")
            if repaired:
                print("[wbto] extended belief under stance soles")
                self._queue_belief_persist()
            if terrain_mod.TERRAIN_STACK == "auto" and not auto_wbto_stance_ready(
                ctrl.terrain, feet
            ):
                return _fail("auto stance not ready after repair")
            plan = plan_contacts(ctrl.terrain, feet, goal)
            if not plan.steps:
                return _fail("empty contact plan")
            if not self._plan_looks_safe(plan, feet):
                return _fail("unsafe contact plan vs stance")
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
            self._sat_ticks = 0
            self.last_error = None
            # Hand the gait clock a stand so classical state does not fight us.
            ctrl.gait.config.mode = 0.0
            print(
                f"[wbto] engaged: {len(plan.steps)} steps, "
                f"plan {plan.duration:.1f}s → goal ({goal[0]:.2f}, {goal[1]:.2f})"
            )
            return True
        except Exception as exc:  # noqa: BLE001 — keep classical alive
            return _fail(str(exc))

    def _plan_looks_safe(self, plan, feet: np.ndarray) -> bool:
        """Refuse plans that land on the floor while the soles are already raised.

        A gap cross lands on the far pad (raised), never in the trench. Floor
        landings from a raised stance mean the belief sealed the walkway into
        a false hole underfoot.
        """
        sole_z = float(np.median(np.asarray(feet, dtype=float).reshape(2, 3)[:, 2]))
        if sole_z < terrain_mod.WBTO_RAISED_TOP:
            return True
        for step in plan.steps[:6]:
            z = float(step.end.pos[2])
            if z < terrain_mod.WBTO_RAISED_TOP:
                return False
        return True

    def _maybe_release_on_tracking(
        self, qpos: np.ndarray, qvel: np.ndarray, feet: np.ndarray
    ) -> bool:
        mpc = self._mpc
        assert mpc is not None
        if int(getattr(mpc, "failures", 0)) >= _FAIL_STREAK_RELEASE:
            print("[wbto] tracking failures — releasing to classical")
            self._arm_fail_cooldown()
            self._release()
            return True
        tau = self.torque
        if tau is not None and tau.size:
            wb = self._wb
            assert wb is not None
            limits = np.abs(wb.mj_model.actuator_ctrlrange[:, 1])
            frac = float(np.max(np.abs(tau) / np.maximum(limits, 1e-3)))
            if frac >= _TORQUE_SAT_FRAC:
                self._sat_ticks += 1
            else:
                self._sat_ticks = 0
            if (
                self._sat_ticks >= _TORQUE_SAT_RELEASE
                and int(getattr(mpc, "failures", 0)) >= 2
            ):
                print("[wbto] torque saturation + solver fails — releasing to classical")
                self._arm_fail_cooldown()
                self._release()
                return True
        return False

    def _emit_proprio_events(self, feet: np.ndarray) -> None:
        """Touchdowns / through-contacts so runtime can correct VLM belief."""
        ctrl = self.controller
        belief = ctrl.terrain
        load = np.asarray(ctrl.foot_load, dtype=float).reshape(-1, 3)
        dt = float(ctrl.config.control_dt)
        for leg in (0, 1):
            if float(load[leg, 2]) <= SWING_CONTACT_N:
                continue
            if self._global_tick - self._event_leg_tick[leg] < _EVENT_COOLDOWN_TICKS:
                continue
            xy = feet[leg, :2]
            measured = float(feet[leg, 2])
            believed = float(belief.height_at(xy, 0.0))
            err = measured - believed
            # Through-contact: sole dropped to near-floor under believed solid
            # (gap overhang). A few centimetres of height error on a pad is a
            # refine, not a carve.
            fell_through = (
                believed >= terrain_mod.WBTO_RAISED_TOP
                and measured < terrain_mod.WBTO_RAISED_TOP
                and believed - measured > STANCE_HEIGHT_TOL
            )
            if fell_through and carve_through_contact(belief, xy, measured):
                print(
                    f"[wbto] through-contact leg={leg} at "
                    f"({xy[0]:+.2f}, {xy[1]:+.2f}): believed {believed:.3f} "
                    f"measured {measured:.3f}; carved and released"
                )
                self._queue_belief_persist(bump=True)
                self._event_leg_tick[leg] = self._global_tick
                self._arm_fail_cooldown()
                self._release()
                return
            # Ignore centimetre-scale snap noise; stance support already handles it.
            if abs(err) <= STANCE_HEIGHT_TOL:
                continue
            # Height mismatch on solid contact: emit touchdown for refine.
            ctrl.terrain_events.append(
                {
                    "kind": "touchdown",
                    "t": self._tick * dt,
                    "leg": int(leg),
                    "xy": [float(xy[0]), float(xy[1])],
                    "yaw": float(ctrl.robot.foot_yaws()[leg]),
                    "believed_z": believed,
                    "measured_z": measured,
                    "step": -1,
                }
            )
            self._event_leg_tick[leg] = self._global_tick

    def _queue_belief_persist(self, *, bump: bool = False) -> None:
        """Ask the runtime to write controller volumes back into loka_state."""
        ctrl = self.controller
        ctrl._terrain_volume_patch = {
            "volumes": [v.to_dict() for v in ctrl.terrain.volumes],
            "bump": bool(bump),
        }

    def _arm_fail_cooldown(self) -> None:
        belief = self.controller.terrain
        self._fail_belief_version = int(belief.version)
        self._fail_goal = self._goal.copy()
        self._fail_retry_at_tick = self._global_tick + int(self._fail_cooldown_ticks)

    def _release(self) -> None:
        if self.active:
            print("[wbto] released → classical")
        self.active = False
        self._mpc = None
        self._sat_ticks = 0

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
