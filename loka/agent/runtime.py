"""Dual-rate standing LOKA runtime: fast plant + async orchestrator."""

from __future__ import annotations

import copy
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from loka.control.locomotion import CONTACT_HEIGHT_TOLERANCE, LocomotionController
from loka.agent.model_state import (
    actuator_torque_scale,
    apply_loka_mutations,
    joint_torque_to_command,
    zero_dead_actuator_commands,
)
from loka.agent.session import FailureEpisode, OperatorSession
from loka.sim import Simulation
from loka.agent.apply import apply_stand_scratchpad
from loka.agent.compress import synthesize_stand_telemetry
from loka.agent.loka_config import loka_config
from loka.agent.directives import heartbeat_due, register_directive, take_fired
from loka.agent.context import (
    default_stand_objective,
    format_stand_configuration,
)
from loka.agent.robot_context import build_compact_robot_context, build_robot_model_context
from loka.agent.anomaly import (
    EffortDeviationObserver,
    belief_actuator_deviation,
    motor_deficit,
)
from loka.agent.belief_search import BeliefSearchRunner, parse_belief_searches
from loka.agent.faults import FaultSpec, capture_nominal_params
from loka.agent.interactive import FaultVizState, commit_fault, restore_faults
from loka.agent.llm import PROMPT_STYLES, load_stand_system_prompt, stand_llm_worker
from loka.agent.session_log import StandSessionLog, default_log_path
from loka.scene_props import prop_contacts
from loka.vision import EgoCameraSpec, EgoRenderer, camera_pose
import mujoco

_SESSION = loka_config()["session"]
_BIAS = loka_config().get("effort_bias", {})
_VISION = loka_config().get("vision", {}) or {}

MU_SLIDE_MPS = 0.05  # planted-site speed that counts as sliding
MU_MIN_NORMAL_N = 20.0
MU_WINDOW_S = 2.0
MU_MIN_SAMPLES = 50
MU_PERCENTILE = 75.0  # partly stuck contacts read below the cone


@dataclass
class LokaConfig:
    enable_llm: bool = True
    #: Attach an ego_cam frame to every user turn. The plant must be built
    #: with the camera (``Simulation(camera=...)``).
    enable_vision: bool = bool(_VISION.get("enabled", False))
    vision: EgoCameraSpec = field(default_factory=lambda: EgoCameraSpec.from_config(_VISION))
    cooldown_s: float = float(_SESSION["cooldown_s"])
    heartbeat_s: float = float(_SESSION["heartbeat_s"])
    telemetry_window_s: float = float(_SESSION["telemetry_window_s"])
    max_turns: int = int(_SESSION.get("max_turns", 5))
    max_memory: int = int(_SESSION.get("max_memory", _SESSION.get("max_turns", 5)))
    stand_on_fault: bool = bool(_BIAS.get("stand_on_fault", True))
    #: Orchestrator model. None keeps ``LOKA_STAND_MODEL`` / the worker default.
    llm_model: str | None = None
    #: ``wall`` applies a reply on the step it arrives. ``fixed_sim`` lets the
    #: plant run for ``llm_latency_s`` of sim time after the dispatch, then
    #: blocks until the reply (or ``llm_request_timeout_s`` of wall time).
    #: ``pause`` is ``fixed_sim`` with a latency of 0.
    llm_latency_mode: str = "wall"
    llm_latency_s: float = 2.0
    llm_request_timeout_s: float = 120.0
    #: ``full`` is the prompt LOKA is evaluated with. ``compact`` keeps the same
    #: rules and reply schema but drops the MJCF dump, the per-actuator effort
    #: list, the state cost and the MPC costs, for small local models.
    prompt_style: str = "full"
    #: Before each turn with a camera frame, ask the model to describe the frame
    #: (one line per object with its direction and a rough distance), then
    #: append that description, as written, to the turn. Needs vision.
    describe_first: bool = False

    def __post_init__(self) -> None:
        assert 0 <= self.max_memory <= self.max_turns, (
            f"session.max_memory ({self.max_memory}) must be <= "
            f"session.max_turns ({self.max_turns})"
        )
        if self.llm_latency_mode not in ("wall", "fixed_sim", "pause"):
            raise ValueError(
                f"llm_latency_mode must be wall, fixed_sim, or pause, "
                f"got {self.llm_latency_mode!r}"
            )
        if float(self.llm_latency_s) < 0.0:
            raise ValueError("llm_latency_s must be >= 0")
        if float(self.llm_request_timeout_s) < 0.0:
            raise ValueError("llm_request_timeout_s must be >= 0")
        if self.prompt_style not in PROMPT_STYLES:
            raise ValueError(f"prompt_style must be one of {PROMPT_STYLES}, got {self.prompt_style!r}")
    objective: str = field(default_factory=default_stand_objective)
    log_path: Path | None = None
    print_compressor: bool = True
    enable_session_log: bool = False


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
        self._fired_faults: set[int] = set()
        self.viz = FaultVizState(external_body_id=self.sim.pelvis_id)
        # Same list the visualizer restores from, so a key and a scripted fault
        # cannot diverge.
        self._fault_backups = self.viz.backups

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
            "plan": None,
        }
        self.compact = self.config.prompt_style == "compact"
        if self.compact:
            self.robot_context = build_compact_robot_context(self.controller.robot.model)
        else:
            self.robot_context = build_robot_model_context(
                self.controller.robot.model,
                str(self.controller.config.model_path),
            )
        self.ego_renderer: EgoRenderer | None = None
        self.vision_block: str | None = None
        self._frame_count = 0
        self._rebuild_prompt()

        self.recent_frames: deque = deque()
        self._last_edit_t = -1e9
        self._last_invoke_t = 0.0

        self.failure_episode: FailureEpisode | None = None
        self.operator_session = OperatorSession()
        self.active_session = None  # FailureEpisode | OperatorSession | None

        self.llm_queue: queue.Queue = queue.Queue()
        self.llm_busy = False
        self.turns: list[dict] = []
        self._dispatch_sim_t = 0.0
        self._dispatch_wall_t = 0.0
        self.operator_queue: queue.Queue = queue.Queue()
        self.history: list[dict] = []
        self._pending_user_content: str | None = None
        self._pending_dispatch_reason: str = "unknown"
        self._fall_logged = False
        self._prev_wbc_failures = 0
        self._prev_mpc_failures = 0
        self._walk_fault_stand = False
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
                f"faults={len(self.faults)}; "
                f"vision={'on' if self.config.enable_vision and self.config.enable_llm else 'off'}"
            )
        if self.config.enable_vision and self.config.enable_llm:
            self._start_vision()
            self._rebuild_prompt()

        # Driver command limit in actuator units. After a gear search accepts,
        # the QP's joint-torque limit is this times that belief, refreshed
        # every step.
        self._command_limit = self.controller.robot.torque_limit.copy()
        self.effort_observer = EffortDeviationObserver()
        self.effort_observer.dt = float(self.sim.config.control_dt)
        self.effort_observer.names = tuple(
            mujoco.mj_id2name(self.sim.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or str(i)
            for i in range(self.sim.model.nu)
        )
        self.belief_search = BeliefSearchRunner(self)
        self._stance_model_key = self._mass_model_key()
        # Contact-force ratio |f_t| / f_n on sliding planted feet. A sliding
        # contact sits on its friction cone, so the ratio reads mu. The window
        # is copied when an out-of-bounds streak trips, for the friction search.
        self._mu_samples: deque = deque()
        self.mu_fault_window: list[float] = []
        self._foot_geom_leg = self._foot_geoms(self.sim.model)
        self._floor_geom = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._mu_capped = False
        self._base_friction_mu = (
            float(self.controller.config.mpc.friction_mu),
            float(self.controller.config.wbc.friction_mu),
        )

    def note_search_result(self, line: str) -> None:
        """Attach a finished search to the fix the next orchestrator turn reads."""
        session = self.failure_episode or self.active_session
        if session is not None and session.interventions:
            session.interventions[-1].setdefault("search_results", []).append(line)
            return
        self.loka_state.setdefault("search_results", []).append(line)

    # -- frame bookkeeping --------------------------------------------------

    def _frame(self) -> dict:
        tel = self.controller.telemetry
        robot = self.controller.robot
        qpos = self.sim.robot_qpos.copy()
        qvel = self.sim.robot_qvel.copy()
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
        fails = self.controller.qp_failures()
        wbc_total = int(fails.get("wbc", 0))
        mpc_total = int(fails.get("mpc", 0))
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
            "mpc_cost": (
                None if tel is None or tel.mpc_cost is None else float(tel.mpc_cost)
            ),
            "stack": getattr(self.controller.config, "stack", "legacy_dcm"),
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
            "goal_active": float(self.controller.gait.config.goal_active) >= 0.5,
            "goal_x": float(self.controller.gait.config.goal_x),
            "goal_y": float(self.controller.gait.config.goal_y),
            "bias_nm": self._bias_nm(),
            "bias_threshold_nm": self._bias_threshold_nm(),
            "belief_search_busy": self.belief_search.busy,
            "prop_contacts": prop_contacts(self.sim.model, self.sim.data, self.sim.prop_layout),
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

    def _push_recent(self, frame: dict) -> None:
        self.recent_frames.append(frame)
        max_n = max(1, int(self.config.telemetry_window_s / self.sim.config.control_dt))
        while len(self.recent_frames) > max_n:
            self.recent_frames.popleft()

    def _refresh_belief_mass(self) -> None:
        """Keep robot.total_mass and MPC mass aligned with belief body_mass."""
        robot = self.controller.robot
        robot.total_mass = float(robot.model.body_mass.sum())
        if hasattr(self.controller, "mpc") and hasattr(self.controller.mpc, "mass"):
            self.controller.mpc.mass = float(robot.total_mass)
        # The stance height and walking knee posture come from the keyframe
        # CoM, which moves with the mass belief. A search probe is a trial
        # value, so only a settled belief retunes the stance.
        key = self._mass_model_key()
        if key != self._stance_model_key and not self.belief_search.busy:
            self._stance_model_key = key
            if hasattr(self.controller, "recalibrate_stance"):
                self.controller.recalibrate_stance()

    def _refresh_belief_friction(self) -> None:
        """Cap the MPC and WBC friction cones at the believed floor friction.

        Both QPs use a square pyramid, |f_x|, |f_y| <= mu f_n, whose corners
        reach sqrt(2) mu. Dividing by sqrt(2) keeps the pyramid inside the
        believed cone. A belief at or above nominal leaves the configured mu.
        """
        model = self.controller.robot.model
        floor = self._floor_geom
        if floor < 0:
            return
        feet = list(self._foot_geom_leg)
        floor_mu = float(model.geom_friction[floor, 0])
        if feet and int(model.geom_priority[floor]) <= int(model.geom_priority[feet].max()):
            floor_mu = max(floor_mu, float(model.geom_friction[feet, 0].max()))
        cap = floor_mu / np.sqrt(2.0)
        base_mpc, base_wbc = self._base_friction_mu
        cfg = self.controller.config
        if cap < base_mpc - 1e-6 or cap < base_wbc - 1e-6:
            want = {"mpc.friction_mu": min(base_mpc, cap), "wbc.friction_mu": min(base_wbc, cap)}
            self._mu_capped = True
        elif self._mu_capped:
            want = {"mpc.friction_mu": base_mpc, "wbc.friction_mu": base_wbc}
            self._mu_capped = False
        else:
            return
        if (
            abs(float(cfg.mpc.friction_mu) - want["mpc.friction_mu"]) > 1e-6
            or abs(float(cfg.wbc.friction_mu) - want["wbc.friction_mu"]) > 1e-6
        ):
            self.controller.update_weights(**want)

    @staticmethod
    def _foot_geoms(model) -> dict[int, int]:
        legs = {}
        for g in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if name.startswith("left_foot"):
                legs[g] = 0
            elif name.startswith("right_foot"):
                legs[g] = 1
        return legs

    def _sample_friction_ratio(self, tang: np.ndarray, planted: np.ndarray) -> None:
        sim = self.sim
        now = float(sim.data.time)
        per_foot = len(tang) // 2
        sliding = [
            bool(np.any(planted[leg * per_foot:(leg + 1) * per_foot]))
            and float(tang[leg * per_foot:(leg + 1) * per_foot][planted[leg * per_foot:(leg + 1) * per_foot]].max()) > MU_SLIDE_MPS
            for leg in (0, 1)
        ]
        if any(sliding):
            force = np.zeros(6)
            for i in range(sim.data.ncon):
                con = sim.data.contact[i]
                leg = self._foot_geom_leg.get(int(con.geom1), self._foot_geom_leg.get(int(con.geom2)))
                if leg is None or not sliding[leg]:
                    continue
                mujoco.mj_contactForce(sim.model, sim.data, i, force)
                if force[0] > MU_MIN_NORMAL_N:
                    self._mu_samples.append((now, float(np.hypot(force[1], force[2]) / force[0])))
        while self._mu_samples and self._mu_samples[0][0] < now - MU_WINDOW_S:
            self._mu_samples.popleft()

    def estimate_friction(self) -> tuple[float, int] | None:
        """Floor mu from the window captured when the fault tripped."""
        ratios = np.array([r for _, r in self.mu_fault_window], dtype=float)
        if ratios.size < MU_MIN_SAMPLES:
            return None
        return float(np.percentile(ratios, MU_PERCENTILE)), int(ratios.size)

    def _mass_model_key(self) -> bytes:
        model = self.controller.robot.model
        return model.body_mass.tobytes() + model.body_ipos.tobytes()

    # -- LLM dispatch -------------------------------------------------------

    def _rebuild_prompt(self) -> None:
        self.system_prompt = load_stand_system_prompt(
            robot_context=self.robot_context,
            vision=self.vision_block,
            style=self.config.prompt_style,
        )

    def _config_text(self) -> str:
        return format_stand_configuration(self.controller, compact=self.compact)

    # -- vision -------------------------------------------------------------

    def _start_vision(self) -> None:
        from loka.agent.context import format_compact_vision_block, format_vision_block
        from loka.vision import floor_cell_size, nominal_camera_pose

        if not self.sim.has_camera:
            raise ValueError(
                "enable_vision needs the plant built with the head camera: "
                "Simulation(camera=EgoCameraSpec().recipe())"
            )
        spec = self.config.vision
        block = format_compact_vision_block if self.compact else format_vision_block
        self.vision_block = block(
            spec,
            nominal_camera_pose(self.sim.model),
            floor_cell_size(self.sim.model),
        )
        self.ego_renderer = EgoRenderer(self.sim.recipe)
        if self.log is not None and spec.save_frames:
            frames = self._frames_dir()
            frames.mkdir(parents=True, exist_ok=True)
            for old in frames.glob("turn_*.png"):
                old.unlink()
        print(
            f"[LOKA] vision on: {spec.width}x{spec.height} ego_cam, "
            f"fovy {spec.fovy_deg:g} deg, pitch down {spec.pitch_down_deg:g} deg "
            f"({self.ego_renderer.backend} renderer)"
        )

    def _frames_dir(self) -> Path:
        return self.log.path.parent / "frames"

    def _vision_request(self, user_content: str) -> tuple[str, dict | None]:
        """Snapshot the plant for this turn and add the CAMERA FRAME section."""
        if self.ego_renderer is None:
            return user_content, None
        from loka.agent.context import format_camera_frame

        snapshot = self.sim.snapshot()
        pose = camera_pose(self.sim.model, self.sim.data)
        self._frame_count += 1
        frame_path = None
        if self.log is not None and self.config.vision.save_frames:
            frame_path = self._frames_dir() / f"turn_{self._frame_count:03d}.png"
        user_content = user_content.rstrip() + "\n\n" + format_camera_frame(pose, self.config.vision)
        request = {
            "render": self.ego_renderer.render,
            "snapshot": snapshot,
            "detail": self.config.vision.detail,
            "frame_path": frame_path,
        }
        if self.config.describe_first:
            from loka.agent.scene_describe import build_describe_prompt

            request["describe"] = {
                "prompt": build_describe_prompt(self._directive_text(), self.config.vision, pose),
            }
        return user_content, request

    def _directive_text(self) -> str:
        directives = self.loka_state.get("mission_directives") or []
        if directives:
            return "; ".join(d.text.strip() for d in directives)
        return "no directive yet: hold balance and wait for a goal"

    def close(self) -> None:
        if self.ego_renderer is not None:
            self.ego_renderer.close()
            self.ego_renderer = None

    def _telemetry_for(self, frame: dict) -> str:
        window = list(self.recent_frames) or [frame]
        return synthesize_stand_telemetry(
            window,
            control_dt=self.sim.config.control_dt,
            observer=self.effort_observer,
            compact=self.compact,
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

    def _in_cooldown(self, now: float) -> bool:
        return (now - self._last_edit_t) < self.config.cooldown_s

    def _dispatch(self, session, user_content: str, *, reason: str = "unknown") -> bool:
        now = float(self.sim.data.time)
        if (
            not self.config.enable_llm
            or self.llm_busy
            or self._in_cooldown(now)
            or self.belief_search.busy
        ):
            return False
        user_content, vision = self._vision_request(user_content)
        self.llm_busy = True
        self.active_session = session
        self._pending_user_content = user_content
        self._pending_dispatch_reason = reason
        self._last_invoke_t = now
        self._dispatch_sim_t = now
        self._dispatch_wall_t = time.perf_counter()
        # A fresh queue so a reply abandoned after a timeout cannot be
        # applied as the next turn.
        self.llm_queue = queue.Queue()
        messages = session.compose_api_messages(
            self.system_prompt,
            user_content,
            max_memory=self.config.max_memory,
        )
        # The worker attaches the image to its own copy, so the logged
        # messages below stay text-only.
        thread = threading.Thread(
            target=stand_llm_worker,
            args=(copy.deepcopy(messages), self.llm_queue),
            kwargs={"model": self.config.llm_model, "vision": vision},
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
                messages=messages,
            )
        return True

    def _drain_llm(self) -> None:
        if not self.llm_busy:
            return
        mode = self.config.llm_latency_mode
        now = float(self.sim.data.time)
        timed_out = False
        if mode in ("fixed_sim", "pause"):
            latency = 0.0 if mode == "pause" else float(self.config.llm_latency_s)
            if now + 1e-9 < float(self._dispatch_sim_t) + latency:
                return
            timeout = max(0.0, float(self.config.llm_request_timeout_s))
            try:
                result = self.llm_queue.get(timeout=timeout)
            except queue.Empty:
                result = None
                timed_out = True
                self.llm_queue = queue.Queue()
        else:
            try:
                result = self.llm_queue.get_nowait()
            except queue.Empty:
                return
        self._accept_llm_result(result, timed_out=timed_out)

    def _accept_llm_result(self, result, *, timed_out: bool) -> None:
        self.llm_busy = False
        now = float(self.sim.data.time)
        self._last_edit_t = now
        wall_latency = time.perf_counter() - float(self._dispatch_wall_t)
        reason = self._pending_dispatch_reason
        user_content = self._pending_user_content
        frame_path = (result or {}).get("frame_path")
        if self.log is not None and result:
            if frame_path is not None:
                self.log.frame(sim_time=now, path=frame_path, size=result.get("image_bytes"))
            if result.get("image_error"):
                self.log.note(f"camera frame: {result['image_error']}", sim_time=now)
        if timed_out or not result or not result.get("scratchpad"):
            if timed_out:
                note = f"orchestrator timed out after {self.config.llm_request_timeout_s:.0f}s"
            elif result and result.get("parse_error"):
                note = f"unparseable orchestrator reply ({result['parse_error']})"
            else:
                note = "empty orchestrator result"
            print(f"[LOKA] {note}; cooldown {self.config.cooldown_s:.1f}s")
            if self.log is not None:
                self.log.note(note, sim_time=now)
            self.turns.append(
                {
                    "reason": reason,
                    "sim_time": float(self._dispatch_sim_t),
                    "applied_sim_time": now,
                    "wall_latency_s": wall_latency,
                    "user_content": user_content,
                    "raw_yaml": None if not result else result.get("raw_yaml"),
                    "scratchpad": None,
                    "summary": None,
                    "timed_out": timed_out,
                    "model": self.config.llm_model,
                }
            )
            self._pending_user_content = None
            return
        scratchpad = result["scratchpad"]
        raw = result.get("raw_yaml", "")
        if self.log is not None:
            if result.get("scene_description"):
                self.log.note(f"scene description:\n{result['scene_description']}", sim_time=now)
            if result.get("raw_reply"):
                self.log.note(
                    f"reply repaired ({'; '.join(result.get('parse_notes') or [])}); "
                    f"before repair:\n{result['raw_reply']}",
                    sim_time=now,
                )
            self.log.response(
                sim_time=float(self.sim.data.time),
                raw_yaml=raw,
                scratchpad=scratchpad if isinstance(scratchpad, dict) else None,
            )
        if self.active_session is not None and self._pending_user_content is not None:
            self.active_session.append_exchange(self._pending_user_content, raw)
        self._pending_user_content = None
        self._sync_walk_block()
        summary = apply_stand_scratchpad(
            self.controller,
            scratchpad,
            self.loka_state,
            plant_model=self.sim.model,
            sim_time=float(self.sim.data.time),
            episode=self.active_session if isinstance(self.active_session, FailureEpisode) else None,
        )
        searches = parse_belief_searches(scratchpad if isinstance(scratchpad, dict) else {})
        if searches:
            summary["belief_searches"] = [
                {"kind": spec.kind, "name": spec.name, "axis": spec.axis} for spec in searches
            ]
            self.belief_search.enqueue(searches)
        if self.log is not None:
            self.log.apply_summary(sim_time=now, summary=summary)
        self.turns.append(
            {
                "reason": reason,
                "sim_time": float(self._dispatch_sim_t),
                "applied_sim_time": now,
                "wall_latency_s": wall_latency,
                "user_content": user_content,
                "raw_yaml": raw,
                "scratchpad": scratchpad if isinstance(scratchpad, dict) else None,
                "summary": summary,
                "timed_out": False,
                "model": self.config.llm_model,
                "frame_path": None if frame_path is None else str(frame_path),
            }
        )
        self._rebuild_prompt()
        print(
            f"[LOKA] edit applied at t={now:.2f}s; "
            f"no orchestrator calls for {self.config.cooldown_s:.1f}s"
        )

    def submit_operator(self, text: str) -> None:
        self.operator_queue.put(text.strip())

    # -- faults -------------------------------------------------------------

    def inject_fault_now(self, fault: FaultSpec) -> bool:
        """Apply a plant fault immediately (interactive / scripted)."""
        ok, msg = commit_fault(self.sim, self.viz, fault)
        print(msg)
        if self.log is not None:
            self.log.note(msg, sim_time=float(self.sim.data.time))
        return ok

    def clear_interactive_faults(self) -> None:
        """Restore mass / friction / actuator plant edits (not pushes)."""
        msg = restore_faults(self.sim, self.viz)
        if msg is None:
            print("[FAULT] nothing to clear")
            return
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
        # The command scale is the committed belief gear. A gear search holds
        # the previous constant while it probes, and the accepted value is
        # what the QP plans against on the next step. On hardware the command
        # is the current or torque setpoint. The plant gear is not read.
        robot = self.controller.robot
        scale = actuator_torque_scale(
            robot.model,
            self.loka_state,
            belief_gear=self.belief_search.command_gear_hold,
        )
        achievable = self._command_limit * scale
        robot.torque_limit[:] = achievable
        if self.controller.wbc.torque_limit is not robot.torque_limit:
            self.controller.wbc.torque_limit[:] = achievable
        torque = self.controller.compute_torque(sim.robot_qpos, sim.robot_qvel)
        zero_dead_actuator_commands(torque, self.loka_state)
        command = joint_torque_to_command(torque, scale, self._command_limit)
        sim.torque = command
        force = sim._applied_force()
        self.viz.external_force = force.copy()
        self.viz.external_body_id = sim.pelvis_id

        for _ in range(sim.steps_per_control):
            sim.data.ctrl[:] = command
            sim.data.xfrc_applied[sim.pelvis_id, :3] = force
            mujoco.mj_step(sim.model, sim.data)

        sim.data.xfrc_applied[sim.pelvis_id, :3] = 0.0

        # Post-step qpos is ahead of xpos. The mass residual is the measured
        # generalized force minus the belief model's inverse dynamics at that
        # state. Actuator torque stands in for current, and the contact
        # constraint force for foot sensors. The plant's inertias are not read.
        mujoco.mj_kinematics(sim.model, sim.data)
        mujoco.mj_comPos(sim.model, sim.data)
        robot = self.controller.robot
        nv = sim.robot_nv
        robot.update(sim.robot_qpos, sim.robot_qvel)
        contact_jac = robot.contact_jacobian()
        belief = robot.data
        belief.qacc[:] = sim.data.qacc[:nv]
        belief_inverse = np.empty(robot.model.nv)
        mujoco.mj_rne(robot.model, belief, 1, belief_inverse)
        measured = (
            sim.data.qfrc_actuator[:nv]
            + sim.data.qfrc_constraint[:nv]
            + sim.data.qfrc_applied[:nv]
        )
        deviation = belief_actuator_deviation(
            measured, belief_inverse, contact_jac
        )
        deficit, gear_error = motor_deficit(
            sim.model,
            robot.model.actuator_gear[:, 0],
            self.loka_state["nominal_gears"],
            sim.data.ctrl,
            sim.data.qfrc_actuator[:nv],
        )
        heights = robot.data.site_xpos[robot.contact_site_ids][:, 2]
        planted = heights < self.controller._ground_height + CONTACT_HEIGHT_TOLERANCE
        tang = np.linalg.norm(
            (contact_jac @ sim.robot_qvel).reshape(-1, 3)[:, :2],
            axis=1,
        )
        slip = float(tang[planted].max()) if np.any(planted) else 0.0
        self._sample_friction_ratio(tang, planted)
        tel = self.controller.telemetry
        self.effort_observer.is_walking = bool(tel is not None and tel.walking)
        self.effort_observer.step_period = float(self.controller.gait.config.step_period)
        self.effort_observer.update(
            deviation,
            self.controller.mpc.cost_components,
            printout=self.effort_observer.printout,
            xyz=sim.data.qpos[:3],
            motor=deficit,
            slip=slip,
            gear_error=gear_error,
        )
        self._sync_walk_block()

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
        self.sim.apply_prop_requests()
        self._maybe_inject_faults()
        self._drain_llm()

        # Belief on controller model each tick (idempotent).
        apply_loka_mutations(
            self.controller.robot.model,
            self.loka_state,
            self.loka_state["nominal_gears"],
        )
        self._refresh_belief_mass()
        self._refresh_belief_friction()

        self._control_step()
        self.belief_search.tick()

        frame = self._frame()
        self.history.append(
            {
                "t": frame["time"],
                "com_error": float(np.linalg.norm(frame["com_error"])),
                "fell": self.sim.fell,
            }
        )

        now = frame["time"]
        self._push_recent(frame)

        # Operator directives are locked text. They do not replace the charter.
        # Nothing is dequeued during cooldown, so the request or listener waits.
        if not self.llm_busy and not self._in_cooldown(now):
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
                    telemetry = self._telemetry_for(frame)
                    cfg_text = self._config_text()
                    user = self.operator_session.build_request_turn(
                        request,
                        telemetry,
                        self.loka_state,
                        now,
                        cfg_text,
                        max_memory=self.config.max_memory,
                    )
                    self._dispatch(self.operator_session, user, reason="directive")
                return frame

            fired = self._consume_listeners(frame)
            if fired and self.config.enable_llm:
                telemetry = self._telemetry_for(frame)
                cfg_text = self._config_text()
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

        obs = self.effort_observer
        effort_fault = obs.consecutive_out_of_bounds >= obs.consecutive_steps
        if obs.consecutive_out_of_bounds == obs.consecutive_steps:
            self.mu_fault_window = list(self._mu_samples)
        recovered = (obs.consecutive_in_bounds * obs.dt) > obs.clear_below_s
        just_stood = False
        if effort_fault and self.config.enable_llm and self.config.stand_on_fault:
            just_stood = self._stand_for_walking_bias(now, frame)
        if (
            effort_fault
            and self.config.enable_llm
            and not self.llm_busy
            and not self._in_cooldown(now)
        ):
            if self.failure_episode is None:
                self.failure_episode = FailureEpisode(started_at=now)
                pending = self.loka_state.pop("search_results", None)
                if pending:
                    self.failure_episode.interventions.append({
                        "time": now,
                        "hypothesis": "earlier belief search",
                        "search_results": list(pending),
                    })
            if self._walk_fault_stand:
                self.failure_episode.interrupted_walk = True
            if self.failure_episode.turns_sent >= self.config.max_turns:
                if not self.failure_episode._turn_cap_noted:
                    self.failure_episode._turn_cap_noted = True
                    note = (
                        f"failure episode turn cap ({self.config.max_turns}) "
                        f"reached; no further turns until the bias is back in bounds"
                    )
                    print(f"[LOKA] {note} at t={now:.2f}s")
                    if self.log is not None:
                        self.log.note(note, sim_time=now)
            else:
                telemetry = self._telemetry_for(frame)
                cfg_text = self._config_text()
                if self.failure_episode.prior_turn_count == 0:
                    user = self.failure_episode.build_initial_user_turn(
                        telemetry,
                        self.loka_state,
                        now,
                        cfg_text,
                        just_stood=just_stood,
                        max_turns=self.config.max_turns,
                        max_memory=self.config.max_memory,
                    )
                else:
                    user = self.failure_episode.build_user_turn(
                        telemetry,
                        self.loka_state,
                        now,
                        cfg_text,
                        just_stood=just_stood,
                        max_turns=self.config.max_turns,
                        max_memory=self.config.max_memory,
                    )
                if self._dispatch(self.failure_episode, user, reason="effort_bias"):
                    self.failure_episode.turns_sent += 1
        elif recovered and not self.belief_search.busy:
            if self.failure_episode is not None and not self.llm_busy:
                print(f"[LOKA] failure episode closed at t={now:.2f}s")
                if self.log is not None:
                    self.log.note("failure episode closed", sim_time=now)
                self.failure_episode = None
                self.active_session = None
            if not self.llm_busy:
                self._walk_fault_stand = False
            self._maybe_heartbeat(frame, now)

        if self.sim.fell and self.log is not None and not self._fall_logged:
            self._fall_logged = True
            self.log.note(
                f"FELL pelvis_z={float(self.sim.data.qpos[2]):.4f} "
                f"tasks={self.controller.task_snapshot()}",
                sim_time=now,
            )

        return frame

    def _bias_nm(self) -> float | None:
        obs = self.effort_observer
        if obs.joint_bias is None:
            return None
        return float(np.sum(np.abs(obs.joint_bias)))

    def _bias_threshold_nm(self) -> float:
        obs = self.effort_observer
        if obs.is_walking:
            return float(obs.walk_threshold_nm)
        return float(obs.stand_threshold_nm)

    def _sync_walk_block(self) -> None:
        """Refuse a walk command while |bias| is above the mode threshold."""
        total = self._bias_nm()
        if total is None:
            self.controller.gait.block_walk = False
            return
        self.controller.gait.block_walk = self.effort_observer.out_of_bounds

    def _stand_for_walking_bias(self, now: float, frame: dict) -> bool:
        """Stand before the orchestrator is invoked for a walking effort fault.

        Returns True only on the step that commands the stand. Later steps of
        the same episode leave ``_walk_fault_stand`` set so the prompt still
        tells the model to walk back to the goal.
        """
        from loka.control.gait import MODE_WALK

        gait = self.controller.gait
        if float(gait.config.mode) < MODE_WALK:
            return False
        if not bool(frame.get("walking")) and not gait.wants_walk():
            return False
        self.controller.set_task_targets({"gait.mode": "stand", "gait.speed": 0.0})
        self._walk_fault_stand = True
        frame["walking"] = False
        frame["cmd_mode"] = float(gait.config.mode)
        frame["cmd_speed"] = float(gait.config.speed)
        frame["goal_active"] = float(gait.config.goal_active) >= 0.5
        print(
            f"[LOKA] effort bias while walking: gait.mode=stand at t={now:.2f}s"
        )
        if self.log is not None:
            self.log.note("effort bias while walking: gait.mode=stand", sim_time=now)
        return True

    def _maybe_heartbeat(self, frame: dict, now: float) -> None:
        """Wake LOKA on a fixed period when no listener has called it."""
        if (
            not self.config.enable_llm
            or self.llm_busy
            or self._in_cooldown(now)
            or self.failure_episode is not None
        ):
            return
        if not heartbeat_due(now, self._last_invoke_t, self.config.heartbeat_s):
            return
        telemetry = self._telemetry_for(frame)
        cfg_text = self._config_text()
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