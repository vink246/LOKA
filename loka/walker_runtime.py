"""Headless Walker MJPC + LOKA step loop shared by the viewer and the suite."""

from __future__ import annotations

import queue
import sys
import threading
from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

from loka.compressor import (
    ANOMALY_COLLECTION_S,
    DEFAULT_TELEMETRY_SECTIONS,
    get_tracking_error,
    synthesize_generalized_telemetry,
    TelemetrySections,
    TELEMETRY_WINDOW_S,
)
from loka.model_state import apply_loka_mutations, zero_dead_actuator_commands
from loka.orchestrator import (
    apply_scratchpad,
    llm_worker,
    load_system_prompt,
    make_planner_agent,
)
from loka.robot_context import (
    build_robot_model_context,
    capture_nominal_params,
    discover_capabilities,
    format_current_mpc_configuration,
    format_live_model_parameters,
)
from loka.session import FailureEpisode, OperatorSession
from loka.task_catalog import initial_error_spec, resolve_xml_path
from loka.walker_suite.faults import (
    PlantFaults,
    ResolvedPerturbation,
    assert_belief_isolated,
    plant_friction_geom_ids,
    resolve_perturbation,
    sanitize_belief_worldview,
)
from loka.walker_suite.stochastic import apply_seeded_state_noise

DEFAULT_TASK_ID = "Walker"
DEFAULT_OBJECTIVE = (
    "Sustained locomotion: maintain Height Goal (~1.2 m) and Speed Goal (1.0 m/s)."
)
OPERATOR_REQUEST_PREFIX = "loka:"
# After a LOKA fix is applied, wait this long before checking for a new
# anomaly or starting another collection window.
COOLDOWN_PERIOD = 5.0
PLANNER_TIMESTEP = 0.01
ACTUATOR_NAMES = (
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
)


def start_operator_input_thread(request_queue: queue.Queue) -> None:
    """Read operator requests from stdin without blocking the sim loop."""

    def _reader():
        while True:
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if not line:
                break

            text = line.strip()
            if not text:
                continue
            if text.lower().startswith(OPERATOR_REQUEST_PREFIX):
                text = text[len(OPERATOR_REQUEST_PREFIX) :].strip()
            if text:
                request_queue.put(text)
                print(f"\n[LOKA] Queued operator request: {text}")

    threading.Thread(target=_reader, daemon=True).start()


def distance_objective(goal_distance_m: float) -> str:
    return (
        f"Walk forward until you reach {goal_distance_m:g} meters. "
        "Recover from disturbances and keep moving toward that distance goal."
    )


def snapshot_mpc(agent, model) -> dict:
    capabilities = discover_capabilities(agent, model)
    return {
        "cost_weights": capabilities.get("cost_weights", {}),
        "planner": capabilities.get("planner", {}),
        "task_parameters": capabilities.get("task_parameters", {}),
    }


def build_snapshot_telemetry(nominal_buffer, anomaly_buffer, model, sections, error_spec):
    """Build telemetry from the most recent frames for operator requests."""
    recent_frames = int(TELEMETRY_WINDOW_S / model.opt.timestep)
    source = list(nominal_buffer) if nominal_buffer else list(anomaly_buffer)
    if not source:
        return "No telemetry frames available yet."

    recent = source[-recent_frames:] if len(source) > recent_frames else source
    baseline = list(nominal_buffer) if nominal_buffer else recent
    return synthesize_generalized_telemetry(
        baseline,
        deque(recent),
        model,
        sections=sections,
        error_spec=error_spec,
    )


def dispatch_to_orchestrator(user_turn, session, system_prompt, llm_queue, label):
    api_messages = session.compose_api_messages(system_prompt, user_turn)
    print(user_turn)
    print(f"\n[INFO] Dispatching to LOKA Orchestrator ({label})...")
    threading.Thread(
        target=llm_worker,
        args=(api_messages, llm_queue),
        daemon=True,
    ).start()
    return user_turn, session


def rebuild_system_prompt(loka_state, robot_context, agent, model):
    capabilities = discover_capabilities(agent, model)
    return load_system_prompt(
        robot_context=robot_context,
        objective=loka_state.get("primary_objective"),
        capabilities=capabilities,
        error_spec=loka_state.get("error_spec"),
    )


@dataclass
class WalkerStep:
    time: float
    error: float
    in_failure: bool
    llm_is_busy: bool
    status: str
    frame: dict
    failure_round: int | None
    queued_operator_requests: int
    fault_active: bool


class WalkerRuntime:
    """One physics step of Walker MJPC, with optional LOKA orchestration."""

    def __init__(
        self,
        *,
        task_id: str = DEFAULT_TASK_ID,
        xml_path: str | None = None,
        objective: str | None = None,
        enable_llm: bool = True,
        speed_goal: float | None = 1.0,
        telemetry_sections: TelemetrySections | None = None,
        operator_request_queue: queue.Queue | None = None,
    ):
        self.task_id = task_id
        self.xml_path = resolve_xml_path(task_id, xml_path)
        self.enable_llm = bool(enable_llm)
        self.telemetry_sections = telemetry_sections or DEFAULT_TELEMETRY_SECTIONS
        self.operator_request_queue = operator_request_queue or queue.Queue()

        # Two MjModels: belief is what MJPC is allowed to know. Plant is ground
        # truth physics. Perturbations mutate only the plant; LOKA Model_Mutations
        # mutate only belief and are pushed into the agent via re-Init.
        self.belief_model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        if self.model is self.belief_model:
            raise RuntimeError("Plant and MPC belief must be distinct MjModel objects")
        self.data = mujoco.MjData(self.model)
        # Lock backpack/box on belief *before* the C++ agent serializes it.
        # Otherwise the planner's worldview would still carry the suite XML
        # overlays (collidable box) even though Python belief is sanitized later.
        sanitize_belief_worldview(self.belief_model)
        self._snapshot_belief_locks()
        self.plant = PlantFaults(self.model, self.data)
        if self.plant.model is self.belief_model:
            raise RuntimeError("PlantFaults is bound to the MPC belief model")
        self.agent = make_planner_agent(
            task_id, self.belief_model, plant_model=self.model
        )
        if self.agent.model is self.model:
            raise RuntimeError(
                "MJPC Agent.model aliases the plant; perturbations would leak "
                "into planning"
            )
        if self.agent.model is not self.belief_model:
            raise RuntimeError("MJPC Agent.model is not the sanitized belief model")
        self.robot_context = build_robot_model_context(self.belief_model, self.xml_path)

        if objective is None:
            objective = (
                DEFAULT_OBJECTIVE
                if task_id == "Walker"
                else f"Complete the '{task_id}' task successfully."
            )

        floor_id = int(
            mujoco.mj_name2id(self.belief_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        )
        self.loka_state = {
            "mutations": [],
            "nominal_params": capture_nominal_params(self.belief_model),
            "nominal_gears": self.belief_model.actuator_gear[:, 0].copy(),
            "nominal_friction": self.belief_model.geom_friction.copy(),
            "nominal_mass": self.belief_model.body_mass.copy(),
            "nominal_inertia": self.belief_model.body_inertia.copy(),
            "nominal_ipos": self.belief_model.body_ipos.copy(),
            "floor_geom_id": floor_id if floor_id >= 0 else None,
            "friction_contact_ids": (
                plant_friction_geom_ids(self.belief_model, floor_id)
                if floor_id >= 0
                else ()
            ),
            "primary_objective": objective,
            "error_spec": initial_error_spec(task_id, self.belief_model),
        }
        self.system_prompt = rebuild_system_prompt(
            self.loka_state, self.robot_context, self.agent, self.belief_model
        )

        if speed_goal is not None:
            try:
                task_params = self.agent.get_task_parameters()
                if "Speed Goal" in task_params:
                    self.agent.set_task_parameter("Speed Goal", float(speed_goal))
            except Exception as exc:
                print(f"Warning: Could not set Speed Goal ({exc}).")

        buffer_maxlen = int(TELEMETRY_WINDOW_S / self.model.opt.timestep)
        self.nominal_buffer = deque(maxlen=buffer_maxlen)
        self.anomaly_buffer = deque(maxlen=buffer_maxlen)

        self.last_dispatch_time = -COOLDOWN_PERIOD
        self.anomaly_collect_since = None
        self.last_planner_time = 0.0
        self.llm_queue: queue.Queue = queue.Queue()
        self.llm_is_busy = False
        self.failure_episode = None
        self.pending_user_turn = None
        self.pending_session = None
        self.operator_session = OperatorSession()
        self.loka_turns: list[dict] = []
        self.mpc_snapshots: list[dict] = []
        self._record_mpc_snapshot(self.data.time)

        self.actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            or ACTUATOR_NAMES[i]
            for i in range(self.model.nu)
        ]

    def _snapshot_belief_locks(self) -> None:
        """Remember compiled nominal appearance so plant overlays cannot leak."""
        self._belief_obstacle_id = mujoco.mj_name2id(
            self.belief_model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle"
        )
        self._belief_backpack_id = mujoco.mj_name2id(
            self.belief_model, mujoco.mjtObj.mjOBJ_GEOM, "backpack"
        )
        if self._belief_obstacle_id >= 0:
            self._belief_obstacle_body_id = int(
                self.belief_model.geom_bodyid[self._belief_obstacle_id]
            )
            self._belief_obstacle_pos = self.belief_model.geom_pos[
                self._belief_obstacle_id
            ].copy()
            self._belief_obstacle_body_pos = (
                self.belief_model.body_pos[self._belief_obstacle_body_id].copy()
                if self._belief_obstacle_body_id > 0
                else None
            )
            self._belief_obstacle_size = self.belief_model.geom_size[
                self._belief_obstacle_id
            ].copy()
            self._belief_obstacle_contype = 0
            self._belief_obstacle_conaffinity = 0
        else:
            self._belief_obstacle_body_id = -1
            self._belief_obstacle_pos = None
            self._belief_obstacle_body_pos = None
            self._belief_obstacle_size = None
            self._belief_obstacle_contype = 0
            self._belief_obstacle_conaffinity = 0
        if self._belief_backpack_id >= 0:
            self._belief_backpack_rgba = self.belief_model.geom_rgba[
                self._belief_backpack_id
            ].copy()
        else:
            self._belief_backpack_rgba = None

    def _record_mpc_snapshot(self, sim_time: float) -> None:
        self.mpc_snapshots.append(
            {"time": float(sim_time), **snapshot_mpc(self.agent, self.belief_model)}
        )

    def reset(self, *, seed: int | None = None, init_noise: float = 0.0) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.plant.clear()
        if float(init_noise) > 0.0:
            apply_seeded_state_noise(
                self.model, self.data, 0 if seed is None else int(seed), init_noise
            )
        apply_loka_mutations(
            self.belief_model, self.loka_state, self.loka_state["nominal_gears"]
        )
        self.assert_mpc_unaware_of_plant()
        self.last_planner_time = 0.0
        self.last_dispatch_time = -COOLDOWN_PERIOD
        self.anomaly_collect_since = None
        self.llm_is_busy = False
        self.failure_episode = None
        self.pending_user_turn = None
        self.pending_session = None
        self.nominal_buffer.clear()
        self.anomaly_buffer.clear()

    def activate_fault(self, spec: dict | ResolvedPerturbation, sim_time: float) -> ResolvedPerturbation:
        if self.plant.model is self.belief_model or self.plant.model is getattr(
            self.agent, "model", None
        ):
            raise RuntimeError("PlantFaults is bound to the MPC planner model")
        resolved = (
            spec
            if isinstance(spec, ResolvedPerturbation)
            else resolve_perturbation(spec, self.model)
        )
        self.plant.activate(resolved, sim_time)
        self.assert_mpc_unaware_of_plant()
        return resolved

    def toggle_right_hip_fault(self) -> bool:
        if (
            self.plant.is_active
            and self.plant.active is not None
            and self.plant.active.kind == "actuator_dead"
            and self.plant.active.actuator_name == "right_hip"
        ):
            self.plant.clear()
            return False
        try:
            resolved = resolve_perturbation(
                {"kind": "actuator_dead", "actuator": "right_hip"}, self.model
            )
        except ValueError:
            print("\n\n[KEYBOARD] No right_hip actuator; fault inject skipped.")
            return False
        self.plant.activate(resolved, float(self.data.time))
        self.assert_mpc_unaware_of_plant()
        return True

    def close(self) -> None:
        try:
            self.agent.close()
        except Exception:
            pass

    def _drain_llm(self) -> None:
        if self.llm_queue.empty():
            return
        worker_result = self.llm_queue.get()
        completed_session = self.pending_session
        if worker_result and self.pending_user_turn is not None and completed_session is not None:
            completed_session.append_exchange(
                self.pending_user_turn, worker_result["raw_yaml"]
            )
            self.agent = apply_scratchpad(
                self.agent,
                worker_result["scratchpad"],
                self.loka_state,
                self.belief_model,
                episode=completed_session,
                data=self.data,
                task_id=self.task_id,
                plant_model=self.model,
            )
            self.assert_mpc_unaware_of_plant()
            self.system_prompt = rebuild_system_prompt(
                self.loka_state, self.robot_context, self.agent, self.belief_model
            )
            self.loka_turns.append(
                {
                    "time": float(self.data.time),
                    "raw_yaml": worker_result.get("raw_yaml"),
                    "scratchpad": worker_result.get("scratchpad"),
                }
            )
            self._record_mpc_snapshot(self.data.time)
        self.pending_user_turn = None
        self.pending_session = None
        self.llm_is_busy = False
        # Cooldown starts when the fix is on the planner, not when we asked the LLM.
        self.last_dispatch_time = float(self.data.time)
        self.anomaly_collect_since = None
        if isinstance(completed_session, FailureEpisode):
            self.anomaly_buffer.clear()

    def _maybe_operator_request(self) -> None:
        if self.llm_is_busy or self.operator_request_queue.empty():
            return
        operator_request = self.operator_request_queue.get()
        self.loka_state["primary_objective"] = operator_request
        self.system_prompt = rebuild_system_prompt(
            self.loka_state, self.robot_context, self.agent, self.belief_model
        )
        telemetry = build_snapshot_telemetry(
            self.nominal_buffer,
            self.anomaly_buffer,
            self.belief_model,
            self.telemetry_sections,
            self.loka_state["error_spec"],
        )
        mpc_config = format_current_mpc_configuration(self.agent, self.belief_model)
        model_parameters = format_live_model_parameters(
            self.belief_model, self.loka_state
        )
        user_turn = self.operator_session.build_request_turn(
            operator_request,
            telemetry,
            self.loka_state,
            self.data.time,
            mpc_config,
            model_parameters=model_parameters,
        )
        self.pending_user_turn, self.pending_session = dispatch_to_orchestrator(
            user_turn,
            self.operator_session,
            self.system_prompt,
            self.llm_queue,
            f"operator request, {self.operator_session.prior_turn_count} prior turn(s)",
        )
        self.llm_is_busy = True

    def _maybe_failure_dispatch(self, current_error: float, trigger_threshold: float) -> bool:
        in_failure = current_error > trigger_threshold
        if not in_failure:
            self.anomaly_collect_since = None
            return False

        if self.llm_is_busy:
            return True

        cooldown_ready = (self.data.time - self.last_dispatch_time) >= COOLDOWN_PERIOD
        if not cooldown_ready:
            self.anomaly_collect_since = None
            return True

        if self.anomaly_collect_since is None:
            self.anomaly_collect_since = self.data.time

        collection_ready = (
            self.data.time - self.anomaly_collect_since
        ) >= ANOMALY_COLLECTION_S
        if not collection_ready:
            return True

        if self.failure_episode is None:
            self.failure_episode = FailureEpisode(
                self.data.time, nominal_baseline=list(self.nominal_buffer)
            )

        telemetry = synthesize_generalized_telemetry(
            self.failure_episode.nominal_baseline,
            self.anomaly_buffer,
            self.belief_model,
            sections=self.telemetry_sections,
            error_spec=self.loka_state["error_spec"],
        )
        mpc_config = format_current_mpc_configuration(self.agent, self.belief_model)
        model_parameters = format_live_model_parameters(
            self.belief_model, self.loka_state
        )
        if self.failure_episode.round_number == 1:
            user_turn = self.failure_episode.build_initial_user_turn(
                telemetry,
                self.loka_state,
                self.data.time,
                mpc_config,
                model_parameters=model_parameters,
            )
        else:
            user_turn = self.failure_episode.build_user_turn(
                telemetry,
                self.loka_state,
                self.data.time,
                mpc_config,
                model_parameters=model_parameters,
            )

        self.pending_user_turn, self.pending_session = dispatch_to_orchestrator(
            user_turn,
            self.failure_episode,
            self.system_prompt,
            self.llm_queue,
            (
                f"failure episode round {self.failure_episode.round_number}, "
                f"{self.failure_episode.prior_turn_count} prior turn(s)"
            ),
        )
        self.llm_is_busy = True
        self.anomaly_collect_since = None
        return True

    def _status_line(self, in_failure: bool, queued_operator_requests: int) -> str:
        if self.llm_is_busy:
            return "[THINKING]"
        if queued_operator_requests:
            return f"[REQ Q={queued_operator_requests}]"
        if in_failure:
            cooldown_left = COOLDOWN_PERIOD - (self.data.time - self.last_dispatch_time)
            if cooldown_left > 0:
                return f"[COOLDOWN {cooldown_left:.1f}s]"
            if self.anomaly_collect_since is not None:
                collected = self.data.time - self.anomaly_collect_since
                if collected < ANOMALY_COLLECTION_S:
                    return f"[COLLECT {collected:.1f}/{ANOMALY_COLLECTION_S:.0f}s]"
            return "[FAIL]"
        return "[NOMINAL]"

    def assert_mpc_unaware_of_plant(self) -> None:
        """Suite faults live on the plant; MJPC may only see nominal + LOKA edits."""
        if self.model is self.belief_model:
            raise RuntimeError("Plant and MPC belief share the same MjModel")
        if getattr(self.agent, "model", None) is self.model:
            raise RuntimeError(
                "MJPC Agent.model aliases the plant; perturbations would leak "
                "into planning."
            )
        assert_belief_isolated(self.belief_model, self.loka_state)
        oid = self._belief_obstacle_id
        if oid >= 0 and self._belief_obstacle_pos is not None:
            if not np.allclose(
                self.belief_model.geom_pos[oid], self._belief_obstacle_pos, atol=1e-9
            ):
                raise RuntimeError("Obstacle pose leaked into the MPC belief model")
            if self._belief_obstacle_body_id > 0 and self._belief_obstacle_body_pos is not None:
                if not np.allclose(
                    self.belief_model.body_pos[self._belief_obstacle_body_id],
                    self._belief_obstacle_body_pos,
                    atol=1e-9,
                ):
                    raise RuntimeError(
                        "Obstacle body pose leaked into the MPC belief model"
                    )
            if (
                self._belief_obstacle_size is not None
                and not np.allclose(
                    self.belief_model.geom_size[oid],
                    self._belief_obstacle_size,
                    atol=1e-9,
                )
            ):
                raise RuntimeError("Obstacle size leaked into the MPC belief model")
            if int(self.belief_model.geom_contype[oid]) != self._belief_obstacle_contype:
                raise RuntimeError("Obstacle collision leaked into the MPC belief model")
            if (
                int(self.belief_model.geom_conaffinity[oid])
                != self._belief_obstacle_conaffinity
            ):
                raise RuntimeError("Obstacle collision leaked into the MPC belief model")
        bid = self._belief_backpack_id
        if bid >= 0 and self._belief_backpack_rgba is not None:
            if not np.allclose(
                self.belief_model.geom_rgba[bid], self._belief_backpack_rgba, atol=1e-9
            ):
                raise RuntimeError("Backpack appearance leaked into the MPC belief model")
            if float(self.belief_model.geom_rgba[bid, 3]) != 0.0:
                raise RuntimeError("Backpack became visible in the MPC belief model")
        agent_model = getattr(self.agent, "model", None)
        if agent_model is not None and agent_model is not self.belief_model:
            assert_belief_isolated(agent_model, self.loka_state)
        if (
            self.plant.is_active
            and self.plant.active is not None
            and self.plant.active.kind == "mass"
            and self.plant.active.body_id >= 0
        ):
            body_id = int(self.plant.active.body_id)
            if np.isclose(
                float(self.model.body_mass[body_id]),
                float(self.belief_model.body_mass[body_id]),
                atol=1e-9,
                rtol=0.0,
            ):
                raise RuntimeError(
                    "Backpack mass is on the plant and the MPC belief; "
                    "the planner is not using a nominal worldview."
                )

    def belief_isolation_report(self) -> dict:
        """Plant vs belief numbers for suite metadata (ice must differ on the plant)."""
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        hip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_hip")
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        report = {
            "plant_is_belief": self.model is self.belief_model,
            "agent_model_is_plant": getattr(self.agent, "model", None) is self.model,
            "agent_model_is_belief": getattr(self.agent, "model", None)
            is self.belief_model,
        }
        if floor_id >= 0:
            report["floor_mu_plant"] = float(self.model.geom_friction[floor_id, 0])
            report["floor_mu_belief"] = float(self.belief_model.geom_friction[floor_id, 0])
            foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot")
            if foot_id >= 0:
                report["right_foot_mu_plant"] = float(self.model.geom_friction[foot_id, 0])
                report["right_foot_mu_belief"] = float(
                    self.belief_model.geom_friction[foot_id, 0]
                )
        if hip_id >= 0:
            report["right_hip_gear_plant"] = float(self.model.actuator_gear[hip_id, 0])
            report["right_hip_gear_belief"] = float(
                self.belief_model.actuator_gear[hip_id, 0]
            )
        if torso_id >= 0:
            report["torso_mass_plant"] = float(self.model.body_mass[torso_id])
            report["torso_mass_belief"] = float(self.belief_model.body_mass[torso_id])
            report["torso_inertia_plant"] = self.model.body_inertia[torso_id].tolist()
            report["torso_inertia_belief"] = self.belief_model.body_inertia[
                torso_id
            ].tolist()
        oid = self._belief_obstacle_id
        if oid >= 0:
            pbody = int(self.model.geom_bodyid[oid])
            bbody = int(self.belief_model.geom_bodyid[oid])
            report["obstacle_pos_plant"] = (
                self.model.body_pos[pbody].tolist()
                if pbody > 0
                else self.model.geom_pos[oid].tolist()
            )
            report["obstacle_pos_belief"] = (
                self.belief_model.body_pos[bbody].tolist()
                if bbody > 0
                else self.belief_model.geom_pos[oid].tolist()
            )
            report["obstacle_size_plant"] = self.model.geom_size[oid].tolist()
            report["obstacle_size_belief"] = self.belief_model.geom_size[oid].tolist()
            report["obstacle_contype_plant"] = int(self.model.geom_contype[oid])
            report["obstacle_contype_belief"] = int(self.belief_model.geom_contype[oid])
            report["obstacle_conaffinity_plant"] = int(self.model.geom_conaffinity[oid])
            report["obstacle_conaffinity_belief"] = int(
                self.belief_model.geom_conaffinity[oid]
            )
        pack = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "backpack")
        if pack >= 0:
            report["backpack_alpha_plant"] = float(self.model.geom_rgba[pack, 3])
            report["backpack_alpha_belief"] = float(
                self.belief_model.geom_rgba[pack, 3]
            )
        return report

    def step(self) -> WalkerStep:
        error_spec = self.loka_state["error_spec"]
        trigger_threshold = error_spec.trigger_threshold

        if self.enable_llm:
            self._drain_llm()
            self._maybe_operator_request()

        queued_operator_requests = self.operator_request_queue.qsize()

        qpos_before = self.data.qpos.copy()
        # Plan against belief (nominal XML + LOKA mutations). The C++ agent
        # holds a serialized copy of belief_model; plant faults never enter it.
        new_command = self.data.time >= self.last_planner_time
        if new_command:
            self.agent.set_state(
                time=self.data.time,
                qpos=self.data.qpos,
                qvel=self.data.qvel,
                act=self.data.act,
            )
            self.agent.planner_step()
            self.last_planner_time += PLANNER_TIMESTEP

        planner_cmd = self.agent.get_action().copy()
        raw_actions = planner_cmd.copy()
        # Zero ctrl only if LOKA declared a virtual amputation in belief.
        # Plant faults must not mask commands — MPC still thinks the hip works.
        raw_actions = zero_dead_actuator_commands(raw_actions, self.loka_state)

        # Physics overlay lives only on the plant model / data.
        self.plant.apply_physics(self.data.time)
        self.assert_mpc_unaware_of_plant()
        # Command latency lags the plant input. The planner still sees its
        # own latest action; only data.ctrl is delayed.
        self.data.ctrl[:] = self.plant.delay_command(
            raw_actions, new_command=new_command
        )
        mujoco.mj_step(self.model, self.data)
        joint_delta = np.abs(self.data.qpos - qpos_before)
        # MuJoCo actuator_force is pre-gear (equals ctrl for these motors).
        # Delivered joint torque is gear_plant * force; plant gear=0 => 0 Nm.
        delivered_torque = self.data.actuator_force * self.model.actuator_gear[:, 0]

        current_error = get_tracking_error(self.data, error_spec)
        frame = {
            "time": self.data.time,
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "planner_cmd": planner_cmd,
            "joint_delta": joint_delta,
            "actuator_force": self.data.actuator_force.copy(),
            "actuator_torque": delivered_torque.copy(),
        }

        self.anomaly_buffer.append(frame)
        if current_error == 0.0 and not self.llm_is_busy:
            self.nominal_buffer.append(frame)

        if (
            self.failure_episode is not None
            and current_error == 0.0
            and not self.llm_is_busy
        ):
            print(
                f"\n[INFO] Failure episode recovered at t={self.data.time:.2f}s. "
                "Closing LLM conversation thread."
            )
            self.failure_episode = None
            self.anomaly_collect_since = None

        in_failure = current_error > trigger_threshold
        if self.enable_llm:
            in_failure = self._maybe_failure_dispatch(current_error, trigger_threshold)
        elif not in_failure:
            self.anomaly_collect_since = None

        status = self._status_line(in_failure, queued_operator_requests)
        return WalkerStep(
            time=float(self.data.time),
            error=float(current_error),
            in_failure=bool(in_failure),
            llm_is_busy=self.llm_is_busy,
            status=status,
            frame=frame,
            failure_round=(
                self.failure_episode.round_number if self.failure_episode else None
            ),
            queued_operator_requests=queued_operator_requests,
            fault_active=self.plant.is_active,
        )
