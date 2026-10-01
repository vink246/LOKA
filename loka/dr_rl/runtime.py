"""Walker-suite runtime that applies a frozen PPO policy instead of MJPC."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from loka.dr_rl.config import load_dr_config, walker_xml_path
from loka.dr_rl.observation import build_observation, gait_phase_features
from loka.dr_rl.policy import FrozenPPOPolicy
from loka.error_spec import ErrorSpec, default_walker_error_spec, get_tracking_error
from loka.walker_suite.faults import PlantFaults, ResolvedPerturbation, resolve_perturbation
from loka.walker_suite.stochastic import apply_seeded_state_noise

ACTUATOR_NAMES = (
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
)


@dataclass
class DrRlStep:
    time: float
    error: float
    in_failure: bool
    llm_is_busy: bool
    status: str
    frame: dict
    failure_round: int | None
    queued_operator_requests: int
    fault_active: bool


class DrRlRuntime:
    """One physics step of the Walker plant under a frozen DR-PPO policy.

    Duck-typed for `EpisodeLogger` / `run_episode`: same plant overlay and
    qpos layout as `WalkerRuntime`, but `data.ctrl` comes from PPO.
    """

    def __init__(
        self,
        *,
        checkpoint: str | Path | None = None,
        xml_path: str | Path | None = None,
        speed_goal: float | None = 1.0,
        config: dict[str, Any] | None = None,
    ):
        self.config = config if config is not None else load_dr_config()
        self.xml_path = str(xml_path) if xml_path else str(walker_xml_path(self.config))
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.plant = PlantFaults(self.model, self.data)
        self.policy = FrozenPPOPolicy(checkpoint, config=self.config)
        self.frame_skip = int(self.config.get("frame_skip", 4))
        self.include_last_action = bool(
            (self.config.get("observation") or {}).get("include_last_action", True)
        )
        spec = default_walker_error_spec()
        if speed_goal is not None:
            spec = ErrorSpec(
                trigger_threshold=spec.trigger_threshold,
                terms=[
                    replace(term, target=float(speed_goal))
                    if term.name == "speed"
                    else term
                    for term in spec.terms
                ],
            )
        self.error_spec = spec

        self.actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            or ACTUATOR_NAMES[i]
            for i in range(self.model.nu)
        ]
        self._last_action = np.zeros(self.model.nu, dtype=np.float32)
        self._policy_hold = 0
        obs_cfg = self.config.get("observation") or {}
        self.include_gait_phase = bool(obs_cfg.get("include_gait_phase", True))
        self.gait_period = float((self.config.get("reward") or {}).get("gait_period", 0.8))
        self.mpc_snapshots: list[dict] = []
        self.loka_turns: list[dict] = []

    def reset(self, *, seed: int | None = None, init_noise: float = 0.0) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.plant.clear()
        self._last_action[:] = 0.0
        self._policy_hold = 0
        if float(init_noise) > 0.0:
            apply_seeded_state_noise(
                self.model, self.data, 0 if seed is None else int(seed), init_noise
            )
        else:
            mujoco.mj_forward(self.model, self.data)

    def activate_fault(
        self, spec: dict | ResolvedPerturbation, sim_time: float
    ) -> ResolvedPerturbation:
        resolved = (
            spec
            if isinstance(spec, ResolvedPerturbation)
            else resolve_perturbation(spec, self.model)
        )
        self.plant.activate(resolved, sim_time)
        return resolved

    def close(self) -> None:
        return None

    def _obs(self) -> np.ndarray:
        phase = None
        if self.include_gait_phase:
            phase = gait_phase_features(float(self.data.time), self.gait_period)
        return build_observation(
            self.data,
            self._last_action,
            include_last_action=self.include_last_action,
            gait_phase=phase,
        )

    def step(self) -> DrRlStep:
        new_command = self._policy_hold <= 0
        if new_command:
            self._last_action = self.policy.predict(self._obs())
            self._policy_hold = self.frame_skip
        self._policy_hold -= 1

        qpos_before = self.data.qpos.copy()
        self.plant.apply_physics(float(self.data.time))
        # Observation still uses the policy's latest action. Latency only
        # delays the torque command the plant receives.
        self.data.ctrl[:] = self.plant.delay_command(
            self._last_action, new_command=new_command
        )
        mujoco.mj_step(self.model, self.data)

        delivered_torque = self.data.actuator_force * self.model.actuator_gear[:, 0]
        current_error = get_tracking_error(self.data, self.error_spec)
        frame = {
            "time": self.data.time,
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "planner_cmd": self._last_action.copy(),
            "joint_delta": np.abs(self.data.qpos - qpos_before),
            "actuator_force": self.data.actuator_force.copy(),
            "actuator_torque": delivered_torque.copy(),
        }
        in_failure = current_error > self.error_spec.trigger_threshold
        return DrRlStep(
            time=float(self.data.time),
            error=float(current_error),
            in_failure=bool(in_failure),
            llm_is_busy=False,
            status="[DR_RL]",
            frame=frame,
            failure_round=None,
            queued_operator_requests=0,
            fault_active=self.plant.is_active,
        )
