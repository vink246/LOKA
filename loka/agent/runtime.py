"""Dual-rate standing LOKA runtime: fast plant + async orchestrator."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from loka.control.locomotion import LocomotionController
from loka.agent.error_spec import get_tracking_error
from loka.agent.model_state import apply_loka_mutations, zero_dead_actuator_commands
from loka.agent.session import FailureEpisode, OperatorSession
from loka.sim import Push, Simulation
from loka.agent.apply import apply_stand_scratchpad
from loka.agent.compress import ANOMALY_COLLECTION_S, synthesize_stand_telemetry
from loka.agent.directives import heartbeat_due, register_directive, take_fired
from loka.agent.context import (
    build_stand_capabilities,
    build_stand_robot_context,
    default_stand_objective,
    format_stand_configuration,
)
from loka.agent.anomaly import assess_stand_anomaly
from loka.agent.error_defaults import (
    default_stand_error_spec,
    default_walk_error_spec,
)
from loka.agent.faults import FaultSpec, apply_plant_fault, capture_nominal_params, clear_plant_faults
from loka.agent.interactive import FaultVizState
from loka.agent.llm import load_stand_system_prompt, stand_llm_worker
from loka.agent.plateau import (
    EpisodeMetrics,
    build_accept_residual_scratchpad,
    exceeds_residual_floor,
    is_improved,
    should_stop_episode,
    snapshot_metrics,
)
from loka.agent.session_log import StandSessionLog, default_log_path

COOLDOWN_S = 4.0
HEARTBEAT_S = 30.0
NOMINAL_BASELINE_S = 2.0
# Collect a short window of early clues before dispatching (faster than catastrophic waits).
EARLY_ANOMALY_COLLECTION_S = 0.4
# After operator / Task_Targets changes, ignore early plant clues while the pose settles.
TASK_SETTLE_SUPPRESS_S = 5.0


@dataclass
class LokaConfig:
    enable_llm: bool = True
    cooldown_s: float = COOLDOWN_S
    anomaly_collection_s: float = ANOMALY_COLLECTION_S
    heartbeat_s: float = HEARTBEAT_S
    objective: str = field(default_factory=default_stand_objective)
    log_path: Path | None = None
    print_compressor: bool = True
    enable_session_log: bool = False
    # Failure-episode intervention budget + plateau gate (see plateau.py).
    max_interventions_per_episode: int = 5
    plateau_min_interventions: int = 2
    plateau_stall_rounds: int = 2
    plateau_rel_improve: float = 0.08
    plateau_abs_score_eps: float = 0.03


class LokaRuntime:
    """Owns a ``Simulation`` plus LOKA session state."""

    def __init__(
        self,
        sim: Simulation,
        config: LokaConfig | None = None,
        faults: list[FaultSpec] | None = None,
    ) -> None:
        self.sim = sim
        self.config = config or LokaConfig()
        self.controller: LocomotionController = sim.controller
        self.faults = list(faults or [])
        self._fault_backups = []
        self._fired_faults: set[int] = set()
        self.viz = FaultVizState(external_body_id=self.sim.pelvis_id)

        belief = self.controller.robot.model
        self.loka_state: dict = {
            "mutations": [],
            "nominal_gears": belief.actuator_gear[:, 0].copy(),
            "nominal_friction": belief.geom_friction.copy(),
            "nominal_mass": belief.body_mass.copy(),
            "nominal_inertia": belief.body_inertia.copy(),
            "nominal_ipos": belief.body_ipos.copy(),
            "nominal_params": capture_nominal_params(belief),
            "primary_objective": self.config.objective,
            "mission_directives": [],
            "listeners": [],
            "error_spec_owner": "default",
            "error_spec": default_stand_error_spec(
                nominal_height=float(self.sim.data.qpos[2])
            ),
        }
        self.capabilities = build_stand_capabilities(self.controller)
        self.robot_context = build_stand_robot_context(self.controller)
        self.system_prompt = load_stand_system_prompt(
            robot_context=self.robot_context,
            objective=self.config.objective,
            capabilities=self.capabilities,
            error_spec=self.loka_state["error_spec"],
        )

        self.nominal_baseline: deque = deque()
        self.anomaly_buffer: deque = deque()
        self._baseline_full = False
        self._anomaly_collect_t0: float | None = None
        self._last_dispatch_t = -1e9
        self._last_invoke_t = 0.0

        self.failure_episode: FailureEpisode | None = None
        self.operator_session = OperatorSession()
        self.active_session = None  # FailureEpisode | OperatorSession | None

        self.llm_queue: queue.Queue = queue.Queue()
        self.llm_busy = False
        self.operator_queue: queue.Queue = queue.Queue()
        self.history: list[dict] = []
        self._pending_user_content: str | None = None
        self._pending_dispatch_reason: str = "unknown"
        self._fall_logged = False
        self._suppress_early_until = -1.0
        # After Task_Targets / Error_Tracking change, clear standing baseline and
        # re-capture healthy telemetry under the new mission.
        self._baseline_stale = False
        self._rebase_ready_at = -1.0
        # After plateau/cap accept: only re-open if plant health exceeds this floor.
        self._residual_floor: EpisodeMetrics | None = None
        self._prev_wbc_failures = 0
        self._prev_mpc_failures = 0
        self._was_walking = False
        self._cross_track = 0.0
        self._heading_watch = (0.0, 0.0, 0.0)

        self.log: StandSessionLog | None = None
        if self.config.enable_session_log:
            path = self.config.log_path or default_log_path()
            self.log = StandSessionLog(path)
            print(f"[LOKA] session log → {self.log.path.resolve()}")
            print(f"[LOKA] session jsonl → {self.log.jsonl_path.resolve()}")
            self.log.note(
                f"LLM={'on' if self.config.enable_llm else 'off'}; "
                f"faults={len(self.faults)}"
            )

    # -- frame bookkeeping --------------------------------------------------

    def _frame(self) -> dict:
        tel = self.controller.telemetry
        robot = self.controller.robot
        qpos = self.sim.data.qpos.copy()
        qvel = self.sim.data.qvel.copy()
        torque = tel.torque.copy() if tel is not None else np.zeros(robot.nu)
        limit = np.maximum(robot.torque_limit, 1e-3)
        util = np.abs(torque) / limit

        contact_speed = 0.0
        if tel is not None:
            # approximate from site velocities via dynamics contact_vel if available
            try:
                dyn = robot.dynamics(qvel)
                contact_speed = float(np.linalg.norm(dyn.contact_vel, axis=1).max())
            except Exception:
                contact_speed = 0.0

        cmd = self.controller.task_snapshot()
        walking = self.controller.gait.wants_walk()
        wbc_total = int(self.controller.wbc._qp.failures)
        mpc_total = int(self.controller.mpc._qp.failures)
        wbc_delta = max(0, wbc_total - self._prev_wbc_failures)
        mpc_delta = max(0, mpc_total - self._prev_mpc_failures)
        self._prev_wbc_failures = wbc_total
        self._prev_mpc_failures = mpc_total
        return {
            "time": float(self.sim.data.time),
            "qpos": qpos,
            "qvel": qvel,
            "ctrl": self.sim.data.ctrl.copy(),
            "torque": torque,
            "joint_util": util,
            "com_error": tel.com_error.copy() if tel is not None else np.zeros(3),
            "rpy": tel.rpy.copy() if tel is not None else np.zeros(3),
            "contact_mask": (
                tel.contact_mask.copy() if tel is not None else np.ones(8, dtype=bool)
            ),
            "contact_forces": (
                tel.contact_forces.copy() if tel is not None else np.zeros((8, 3))
            ),
            "desired_forces": (
                tel.desired_forces.copy() if tel is not None else np.zeros((8, 3))
            ),
            "mpc_cost": float(tel.mpc_cost) if tel is not None else 0.0,
            "support_margin": robot.support_margin(),
            "contact_speed_max": contact_speed,
            # Per-tick deltas for anomaly; totals for compressor display.
            "wbc_failures": wbc_delta,
            "mpc_failures": mpc_delta,
            "wbc_failures_total": wbc_total,
            "mpc_failures_total": mpc_total,
            "walking": walking,
            "cmd_height": cmd["height"],
            "cmd_lean_x": cmd["lean_x"],
            "cmd_lean_y": cmd["lean_y"],
            "cmd_yaw": cmd["yaw"],
            "yaw_ref": float(tel.yaw_ref) if tel is not None else 0.0,
            "heading_goal": float(tel.heading_goal) if tel is not None else 0.0,
            "heading_error_raw": float(tel.heading_error) if tel is not None else 0.0,
            "heading_error": self._gated_heading_error(tel),
            "cross_track": self._update_cross_track(tel),
            "cmd_mode": cmd.get("gait.mode"),
            "cmd_speed": cmd.get("gait.speed"),
            "cmd_heading": cmd.get("gait.heading"),
            "planar_speed": float(np.linalg.norm(qvel[:2])),
        }

    def _gated_heading_error(self, tel) -> float:
        """Heading error, zero until the turn's rate budget has elapsed."""
        if tel is None:
            return 0.0
        from loka.control.gait import wrap_angle

        goal = float(tel.heading_goal)
        watched = self._heading_watch[2]
        if abs(wrap_angle(goal - watched)) > 1e-3:
            self._heading_watch = (float(self.sim.data.time), float(tel.yaw_ref), goal)
        elapsed = float(self.sim.data.time) - self._heading_watch[0]
        rate = max(float(self.controller.gait.config.turn_rate), 1e-3)
        need = abs(wrap_angle(goal - self._heading_watch[1])) / rate
        if elapsed < need:
            return 0.0
        return float(tel.heading_error)

    def _update_cross_track(self, tel) -> float:
        """Leaky integral of lateral velocity in the yaw-reference frame [m]."""
        if tel is None:
            return self._cross_track
        from loka.control.gait import heading_frame

        _, left = heading_frame(float(tel.yaw_ref))
        vel = np.asarray(self.sim.data.qvel[:2], dtype=float)
        dt = float(self.controller.config.control_dt)
        self._cross_track = 0.98 * self._cross_track + float(vel @ left) * dt
        return float(self._cross_track)

    def _sync_gait_mission(self, now: float, walking: bool) -> None:
        """Swap Error_Tracking / suppress early anomaly on stand↔walk transitions."""
        if walking == self._was_walking:
            return
        self._was_walking = walking
        if self.loka_state.get("error_spec_owner") == "loka":
            self._suppress_early_until = max(
                self._suppress_early_until,
                now + (6.0 if walking else TASK_SETTLE_SUPPRESS_S),
            )
            return
        height = float(self.controller.command.height or self.controller.nominal_height)
        if walking:
            self.loka_state["error_spec"] = default_walk_error_spec(
                nominal_height=max(height, 0.55)
            )
            reason = "gait.mode=walk"
        else:
            self.loka_state["error_spec"] = default_stand_error_spec(
                nominal_height=max(height, 0.55)
            )
            reason = "gait.mode=stand"
        self._rebuild_prompt()
        self._invalidate_mission_baseline(now, reason=reason, close_failure_episode=True)
        # Extra settle so the first steps do not look like a plant fault.
        self._suppress_early_until = max(
            self._suppress_early_until, now + (6.0 if walking else TASK_SETTLE_SUPPRESS_S)
        )
        print(
            f"[LOKA] Error_Tracking → "
            f"{'walk' if walking else 'stand'} defaults at t={now:.2f}s"
        )

    def _push_baseline(self, frame: dict) -> None:
        self.nominal_baseline.append(frame)
        max_n = max(1, int(NOMINAL_BASELINE_S / self.sim.config.control_dt))
        while len(self.nominal_baseline) > max_n:
            self.nominal_baseline.popleft()
        if len(self.nominal_baseline) >= max_n:
            self._baseline_full = True

    def _invalidate_mission_baseline(
        self,
        now: float,
        *,
        reason: str,
        close_failure_episode: bool = False,
    ) -> None:
        """Drop prior-mission nominals; re-collect after the pose settles."""
        self.nominal_baseline.clear()
        self._baseline_full = False
        self._baseline_stale = True
        self._rebase_ready_at = now + TASK_SETTLE_SUPPRESS_S
        self._suppress_early_until = max(
            self._suppress_early_until, self._rebase_ready_at
        )
        self.anomaly_buffer.clear()
        self._anomaly_collect_t0 = None
        # Operator objective changes end fault episodes; recovery Task_Targets do not.
        if (
            close_failure_episode
            and self.failure_episode is not None
            and not self.llm_busy
        ):
            print(f"[LOKA] failure episode closed ({reason})")
            if self.log is not None:
                self.log.note(f"failure episode closed ({reason})", sim_time=now)
            if self.active_session is self.failure_episode:
                self.active_session = None
            self.failure_episode = None
        print(
            f"[LOKA] mission baseline invalidated at t={now:.2f}s "
            f"({reason}); will re-baseline after settle"
        )
        if self.log is not None:
            self.log.note(
                f"mission baseline invalidated ({reason})",
                sim_time=now,
            )

    def _maybe_rebaseline(self, frame: dict, *, healthy: bool) -> None:
        if not self._baseline_stale:
            return
        now = float(frame["time"])
        if now < self._rebase_ready_at or not healthy:
            return
        self._push_baseline(frame)
        if self._baseline_full:
            self._baseline_stale = False
            print(
                f"[LOKA] mission nominal baseline rebased at t={now:.2f}s "
                f"({len(self.nominal_baseline)} frames)"
            )
            if self.log is not None:
                self.log.note("mission nominal baseline rebased", sim_time=now)
            # Keep any open episode's reference aligned with the new mission.
            if self.failure_episode is not None:
                self.failure_episode.nominal_baseline = list(self.nominal_baseline)

    # -- plateau / intervention cap -----------------------------------------

    def _refresh_belief_mass(self) -> None:
        """Keep robot.total_mass and MPC mass aligned with belief body_mass."""
        robot = self.controller.robot
        robot.total_mass = float(robot.model.body_mass.sum())
        if hasattr(self.controller, "mpc") and hasattr(self.controller.mpc, "mass"):
            self.controller.mpc.mass = float(robot.total_mass)

    def _accept_residual(
        self,
        metrics: EpisodeMetrics,
        *,
        reason: str,
        now: float,
        live_pelvis_z: float | None = None,
    ) -> None:
        """Stop thrashing: absorb residual into mission + mass belief, close episode."""
        episode = self.failure_episode
        scratchpad = build_accept_residual_scratchpad(
            metrics, self.controller, reason=reason
        )
        if live_pelvis_z is not None and "Error_Tracking" in scratchpad:
            # Retarget pelvis floor to the actual settled height.
            for term in scratchpad["Error_Tracking"].get("terms", []):
                if term.get("name") == "pelvis_height":
                    term["target"] = float(live_pelvis_z) + 0.04
                    break
        scratchpad.pop("_suggested_pelvis_target", None)

        print(
            f"\n[LOKA] accepting residual at t={now:.2f}s "
            f"(reason={reason}, interventions="
            f"{len(episode.interventions) if episode else 0}, "
            f"score={metrics.anomaly_score:.2f}, "
            f"‖f*-f‖={metrics.force_mismatch_n:.1f} N)"
        )
        if self.log is not None:
            self.log.note(
                f"accept residual ({reason}) score={metrics.anomaly_score:.2f}",
                sim_time=now,
            )

        summary = apply_stand_scratchpad(
            self.controller,
            scratchpad,
            self.loka_state,
            plant_model=self.sim.model,
            sim_time=now,
            episode=episode,
        )
        if summary.get("mutations"):
            self._refresh_belief_mass()

        self._residual_floor = metrics
        if episode is not None:
            episode.accepted_residual = metrics

        self._invalidate_mission_baseline(
            now,
            reason=f"accept residual ({reason})",
            close_failure_episode=True,
        )
        self.anomaly_buffer.clear()
        self._anomaly_collect_t0 = None
        self._rebuild_prompt()
        if self.log is not None:
            self.log.apply_summary(sim_time=now, summary=summary)

    def _gate_failure_dispatch(
        self, metrics: EpisodeMetrics | None
    ) -> str | None:
        """Update stall counters; return stop reason or None to dispatch LLM."""
        episode = self.failure_episode
        if episode is None or metrics is None:
            return None
        cfg = self.config
        if episode.metric_history:
            if is_improved(
                metrics,
                episode.metric_history[-1],
                rel_improve=cfg.plateau_rel_improve,
                abs_score_eps=cfg.plateau_abs_score_eps,
            ):
                episode.stall_count = 0
            else:
                episode.stall_count += 1
        episode.metric_history.append(metrics)
        return should_stop_episode(
            n_interventions=len(episode.interventions),
            stall_count=episode.stall_count,
            max_interventions=cfg.max_interventions_per_episode,
            plateau_min_interventions=cfg.plateau_min_interventions,
            plateau_stall_rounds=cfg.plateau_stall_rounds,
        )

    # -- LLM dispatch -------------------------------------------------------

    def _rebuild_prompt(self) -> None:
        self.system_prompt = load_stand_system_prompt(
            robot_context=self.robot_context,
            objective=self.loka_state["primary_objective"],
            capabilities=self.capabilities,
            error_spec=self.loka_state["error_spec"],
        )

    def _telemetry_for(self, frame: dict, *, directive: str) -> str:
        window = list(self.anomaly_buffer) or list(self.nominal_baseline) or [frame]
        return synthesize_stand_telemetry(
            self.nominal_baseline,
            window,
            control_dt=self.sim.config.control_dt,
            error_spec=self.loka_state["error_spec"],
            directive=directive,
        )

    def _consume_listeners(self, frame: dict) -> list:
        armed = list(self.loka_state.get("listeners") or [])
        if not armed:
            return []
        stay, fired = take_fired(armed, frame)
        if fired:
            self.loka_state["listeners"] = stay
            print(
                f"[LOKA] listener fired at t={float(frame['time']):.2f}s "
                f"({len(fired)} condition(s))"
            )
        return fired

    def _dispatch(self, session, user_content: str, *, reason: str = "unknown") -> None:
        if not self.config.enable_llm:
            return
        self.llm_busy = True
        self.active_session = session
        self._pending_user_content = user_content
        self._pending_dispatch_reason = reason
        self._last_dispatch_t = self.sim.data.time
        self._last_invoke_t = float(self.sim.data.time)
        messages = session.compose_api_messages(self.system_prompt, user_content)
        thread = threading.Thread(
            target=stand_llm_worker,
            args=(messages, self.llm_queue),
            daemon=True,
        )
        thread.start()
        print(
            f"\n[LOKA] dispatched orchestrator at t={self.sim.data.time:.2f}s "
            f"({reason})"
        )
        if self.config.print_compressor:
            print("\n" + "-" * 55)
            print(f"[LOKA] COMPRESSOR / USER TURN ({reason})")
            print("-" * 55)
            print(user_content.rstrip())
            print("-" * 55 + "\n")
        if self.log is not None:
            self.log.dispatch(
                sim_time=float(self.sim.data.time),
                reason=reason,
                user_content=user_content,
                system_prompt=self.system_prompt,
            )

    def _drain_llm(self) -> None:
        try:
            result = self.llm_queue.get_nowait()
        except queue.Empty:
            return
        self.llm_busy = False
        if not result or not result.get("scratchpad"):
            print("[LOKA] empty orchestrator result")
            if self.log is not None:
                self.log.note(
                    "empty orchestrator result",
                    sim_time=float(self.sim.data.time),
                )
            self._pending_user_content = None
            return
        scratchpad = result["scratchpad"]
        raw = result.get("raw_yaml", "")
        if self.log is not None:
            self.log.response(
                sim_time=float(self.sim.data.time),
                raw_yaml=raw,
                scratchpad=scratchpad if isinstance(scratchpad, dict) else None,
            )
        if self.active_session is not None and self._pending_user_content is not None:
            self.active_session.append_exchange(self._pending_user_content, raw)
        self._pending_user_content = None
        summary = apply_stand_scratchpad(
            self.controller,
            scratchpad,
            self.loka_state,
            plant_model=self.sim.model,
            sim_time=float(self.sim.data.time),
            episode=self.active_session if isinstance(self.active_session, FailureEpisode) else None,
        )
        now = float(self.sim.data.time)
        # Task setpoint changes move the operating point — re-baseline after settle.
        # Error_Tracking-only edits during fault recovery keep the pose baseline.
        if summary.get("task_targets"):
            self._invalidate_mission_baseline(
                now,
                reason="Task_Targets update",
                close_failure_episode=False,
            )
        if self.log is not None:
            self.log.apply_summary(sim_time=now, summary=summary)
        self._rebuild_prompt()
        self.anomaly_buffer.clear()
        self._anomaly_collect_t0 = None

    def submit_operator(self, text: str) -> None:
        self.operator_queue.put(text.strip())

    # -- faults -------------------------------------------------------------

    def _record_friction_viz(self, backup, mu: float) -> None:
        floor_id = int(backup.payload["id"])
        self.viz.floor_geom_id = floor_id
        if self.viz.floor_rgba_nominal is None:
            self.viz.floor_rgba_nominal = self.sim.model.geom_rgba[floor_id].copy()
        self.viz.friction_mu = float(mu)
        # Strong ice tint when slippery; restore color on clear.
        if mu < 0.6:
            self.sim.model.geom_rgba[floor_id] = np.array(
                [0.15, 0.75, 1.0, 1.0], dtype=np.float32
            )
        elif self.viz.floor_rgba_nominal is not None:
            self.sim.model.geom_rgba[floor_id] = self.viz.floor_rgba_nominal

    def inject_fault_now(self, fault: FaultSpec) -> bool:
        """Apply a plant fault immediately (interactive / scripted)."""
        now = float(self.sim.data.time)
        fault = FaultSpec(fault.kind, now, dict(fault.params))
        if fault.kind == "push":
            impulse = float(fault.params.get("impulse", 6.0))
            direction = np.asarray(
                fault.params.get("direction", (1.0, 0.0, 0.0)), dtype=float
            )
            n = float(np.linalg.norm(direction[:2]))
            if n < 1e-9:
                direction = np.array([1.0, 0.0, 0.0])
            else:
                direction = np.array([direction[0] / n, direction[1] / n, 0.0])
            self.sim.pushes.append(
                Push(impulse=impulse, direction=direction, time=now)
            )
            msg = f"[FAULT] push {impulse:.1f} N.s dir=({direction[0]:+.1f},{direction[1]:+.1f}) at t={now:.2f}s"
            print(msg)
            if self.log is not None:
                self.log.note(msg, sim_time=now)
            return True

        backup = apply_plant_fault(self.sim.model, fault)
        if backup is None:
            print(f"[FAULT] failed to apply {fault.kind} params={fault.params}")
            return False
        self._fault_backups.append(backup)
        self.viz.backups.append(backup)
        if backup.kind == "mass":
            delta = max(0.0, float(fault.params.get("delta_kg", 0.0)))
            body_id = int(backup.payload["id"])
            self.viz.mass_loads.append((body_id, delta))
        elif backup.kind == "friction":
            self._record_friction_viz(backup, float(fault.params.get("mu", 0.25)))
        elif backup.kind == "actuator_dead":
            self.viz.dead_actuator = str(backup.payload.get("name", "?"))
        msg = f"[FAULT] {fault.kind} applied at t={now:.2f}s params={fault.params}"
        print(msg)
        if self.log is not None:
            self.log.note(msg, sim_time=now)
        return True

    def clear_interactive_faults(self) -> None:
        """Restore mass / friction / dead-actuator plant edits (not pushes)."""
        if not self._fault_backups:
            print("[FAULT] nothing to clear")
            return
        clear_plant_faults(self.sim.model, self._fault_backups)
        if (
            self.viz.floor_geom_id >= 0
            and self.viz.floor_rgba_nominal is not None
        ):
            self.sim.model.geom_rgba[self.viz.floor_geom_id] = (
                self.viz.floor_rgba_nominal
            )
        self._fault_backups.clear()
        self.viz.backups.clear()
        self.viz.mass_loads.clear()
        self.viz.friction_mu = None
        self.viz.dead_actuator = None
        self._residual_floor = None
        msg = f"[FAULT] cleared plant mutations at t={self.sim.data.time:.2f}s"
        print(msg)
        if self.log is not None:
            self.log.note(msg, sim_time=float(self.sim.data.time))

    def _maybe_inject_faults(self) -> None:
        now = self.sim.data.time
        for index, fault in enumerate(self.faults):
            if index in self._fired_faults or now < fault.time:
                continue
            self._fired_faults.add(index)
            self.inject_fault_now(fault)

    # -- step ---------------------------------------------------------------

    def _control_step(self) -> None:
        """One control period with dead-actuator zeroing on the command."""
        import mujoco

        sim = self.sim
        torque = self.controller.compute_torque(sim.data.qpos, sim.data.qvel)
        zero_dead_actuator_commands(torque, self.loka_state)
        sim.torque = torque
        force = sim._applied_force()
        self.viz.external_force = force.copy()
        self.viz.external_body_id = sim.pelvis_id
        for _ in range(sim.steps_per_control):
            sim.data.ctrl[:] = torque
            sim.data.xfrc_applied[sim.pelvis_id, :3] = force
            mujoco.mj_step(sim.model, sim.data)
        sim.data.xfrc_applied[sim.pelvis_id, :3] = 0.0

        telemetry = self.controller.telemetry
        sim.history.append(
            {
                "t": sim.data.time,
                "com_error": float(np.linalg.norm(telemetry.com_error)),
                "tilt": float(np.linalg.norm(telemetry.rpy[:2])),
                "torque": float(np.abs(telemetry.torque).max()),
                "solve_ms": telemetry.solve_ms,
                "pelvis_z": float(sim.data.qpos[2]),
            }
        )

    def step(self) -> dict:
        self._maybe_inject_faults()
        self._drain_llm()

        # Belief on controller model each tick (idempotent).
        apply_loka_mutations(
            self.controller.robot.model,
            self.loka_state,
            self.loka_state["nominal_gears"],
        )
        self._refresh_belief_mass()

        self._control_step()

        frame = self._frame()
        self.history.append(
            {
                "t": frame["time"],
                "com_error": float(np.linalg.norm(frame["com_error"])),
                "tracking_error": float(
                    get_tracking_error(
                        self.sim.data, self.loka_state["error_spec"], frame
                    )
                ),
                "fell": self.sim.fell,
            }
        )

        now = frame["time"]
        self._sync_gait_mission(now, bool(frame.get("walking", False)))

        if not self._baseline_full:
            self._push_baseline(frame)
            return frame

        error = get_tracking_error(self.sim.data, self.loka_state["error_spec"], frame)

        # Operator directives are locked text. They do not replace the charter.
        if not self.llm_busy:
            try:
                request = self.operator_queue.get_nowait()
            except queue.Empty:
                request = None
            if request:
                directive = register_directive(self.loka_state, request, now)
                goal = (
                    f" parsed xy={directive.goal_xy}"
                    if directive.goal_xy is not None
                    else ""
                )
                print(
                    f"[LOKA] locked mission directive at t={now:.2f}s: "
                    f"{directive.text}{goal}"
                )
                if self.config.enable_llm:
                    self._rebuild_prompt()
                    self._invalidate_mission_baseline(
                        now,
                        reason=f"directive:{request[:40]}",
                        close_failure_episode=True,
                    )
                    telemetry = self._telemetry_for(
                        frame,
                        directive=(
                            f"Locked directive: {request}\n"
                            "Set Task_Targets, gait, Error_Tracking, and a Listener. "
                            "The directive text itself cannot change.\n"
                        ),
                    )
                    cfg_text = format_stand_configuration(self.controller)
                    user = self.operator_session.build_request_turn(
                        request, telemetry, self.loka_state, now, cfg_text
                    )
                    self._dispatch(self.operator_session, user, reason="directive")
                return frame

            fired = self._consume_listeners(frame)
            if fired and self.config.enable_llm:
                telemetry = self._telemetry_for(
                    frame,
                    directive=(
                        "Listener fired. Read the message and the full state. "
                        "Continue the locked directives.\n"
                    ),
                )
                cfg_text = format_stand_configuration(self.controller)
                user = self.operator_session.build_listener_turn(
                    [item.message for item in fired],
                    telemetry,
                    self.loka_state,
                    now,
                    cfg_text,
                )
                self._dispatch(
                    self.operator_session,
                    user,
                    reason="listener",
                )
                return frame

        mission_breach = error > self.loka_state["error_spec"].trigger_threshold
        early = assess_stand_anomaly(frame)
        frame["anomaly_score"] = early.score
        frame["anomaly_clues"] = list(early.clues)
        settling = now < self._suppress_early_until
        plant_anomaly = early.triggered and not settling
        # After accepting a residual, ignore the same steady-state mismatch until
        # plant health gets meaningfully worse.
        if plant_anomaly and self._residual_floor is not None:
            window = list(self.anomaly_buffer) + [frame]
            cur = snapshot_metrics(window)
            if cur is not None and not exceeds_residual_floor(cur, self._residual_floor):
                plant_anomaly = False
        # While re-baselining after a mission change, Error_Tracking may still
        # describe the *old* objective — do not treat that lag as a fault.
        mission_for_dispatch = (
            mission_breach and not self._baseline_stale and not settling
        )

        # Re-capture MissionNominal from healthy frames after objective changes.
        if self._baseline_stale:
            self._maybe_rebaseline(frame, healthy=early.balance_healthy)

        if mission_for_dispatch or plant_anomaly:
            self.anomaly_buffer.append(frame)
            if self._anomaly_collect_t0 is None:
                self._anomaly_collect_t0 = now
            collect_s = (
                self.config.anomaly_collection_s
                if mission_for_dispatch
                else EARLY_ANOMALY_COLLECTION_S
            )
            ready = (now - self._anomaly_collect_t0) >= collect_s
            cooled = (now - self._last_dispatch_t) >= self.config.cooldown_s
            if (
                self.config.enable_llm
                and ready
                and cooled
                and not self.llm_busy
            ):
                if self.failure_episode is None:
                    self.failure_episode = FailureEpisode(
                        started_at=now,
                        nominal_baseline=list(self.nominal_baseline),
                    )
                metrics = snapshot_metrics(self.anomaly_buffer)
                stop_reason = self._gate_failure_dispatch(metrics)
                if stop_reason is not None and metrics is not None:
                    live_z = float(frame["qpos"][2])
                    self._accept_residual(
                        metrics,
                        reason=stop_reason,
                        now=now,
                        live_pelvis_z=live_z,
                    )
                    return frame

                reason = "mission" if mission_for_dispatch else "early_anomaly"
                clue_txt = (
                    " ".join(early.clues) if early.clues else "(none)"
                )
                telemetry = synthesize_stand_telemetry(
                    self.failure_episode.nominal_baseline,
                    self.anomaly_buffer,
                    control_dt=self.sim.config.control_dt,
                    error_spec=self.loka_state["error_spec"],
                    directive=(
                        f"Plant-health anomaly (score={early.score:.2f}). "
                        f"Clues: {clue_txt}\n"
                        "Mission Error_Tracking "
                        f"{'BREACHED' if mission_breach else 'still inside band'}.\n"
                        "Diagnose from telemetry only (no privileged fault label). "
                        "Prefer 1–3 allowlisted Controller_Targets and/or Task_Targets. "
                        "If gait.mode=walk: do NOT lower friction_mu / kp_base_position "
                        "for normal swing slip or CoM residuals — prefer gait.* knobs "
                        "or lean; crouch only if tipping. Omit locked QP weights.\n"
                    ),
                )
                cfg_text = format_stand_configuration(self.controller)
                if self.failure_episode.prior_turn_count == 0:
                    user = self.failure_episode.build_initial_user_turn(
                        telemetry, self.loka_state, now, cfg_text
                    )
                else:
                    user = self.failure_episode.build_user_turn(
                        telemetry, self.loka_state, now, cfg_text
                    )
                self._dispatch(self.failure_episode, user, reason=reason)
        else:
            # Healthy under the current mission: refresh / rebuild nominal baseline.
            self.anomaly_buffer.clear()
            self._anomaly_collect_t0 = None
            if self._baseline_stale:
                self._maybe_rebaseline(frame, healthy=early.balance_healthy)
            else:
                self._push_baseline(frame)
            if (
                self.failure_episode is not None
                and not self.llm_busy
                and error == 0.0
                and not plant_anomaly
            ):
                print(f"[LOKA] failure episode closed at t={now:.2f}s")
                if self.log is not None:
                    self.log.note("failure episode closed", sim_time=now)
                self.failure_episode = None
                self.active_session = None
            self._maybe_heartbeat(frame, now)

        if self.sim.fell and self.log is not None and not self._fall_logged:
            self._fall_logged = True
            self.log.note(
                f"FELL pelvis_z={float(self.sim.data.qpos[2]):.4f} "
                f"tasks={self.controller.task_snapshot()}",
                sim_time=now,
            )

        return frame

    def _maybe_heartbeat(self, frame: dict, now: float) -> None:
        """Wake LOKA on a fixed period when no listener has called it."""
        if (
            not self.config.enable_llm
            or self.llm_busy
            or self.failure_episode is not None
        ):
            return
        if not heartbeat_due(now, self._last_invoke_t, self.config.heartbeat_s):
            return
        telemetry = self._telemetry_for(
            frame,
            directive=(
                "Periodic check. Recorrect if the locked directives are not "
                "being met. Listeners did not fire this interval.\n"
            ),
        )
        cfg_text = format_stand_configuration(self.controller)
        user = self.operator_session.build_heartbeat_turn(
            telemetry,
            self.loka_state,
            now,
            cfg_text,
            period_s=self.config.heartbeat_s,
        )
        self._dispatch(self.operator_session, user, reason="heartbeat")

    def run(self, duration: float, stop_on_fall: bool = True) -> list[dict]:
        while self.sim.data.time < duration:
            self.step()
            if stop_on_fall and self.sim.fell:
                break
        return self.history
