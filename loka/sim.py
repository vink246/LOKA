"""MuJoCo simulation harness for the standing controller.

The controller reads ``qpos`` / ``qvel`` straight out of ``MjData`` and writes
torques into ``ctrl`` -- one process, no state estimator, no network. That
keeps the control law the only thing under test. ``UnitreeRLSimulation`` puts
the unitree_rl_lab velocity policy on the same plant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from loka.control.locomotion import LocomotionConfig, LocomotionController

#: Pelvis height below which the run is called a fall.
FALL_HEIGHT = 0.45


@dataclass
class Push:
    """A horizontal impulse delivered to the pelvis."""

    impulse: float  # N.s
    direction: np.ndarray  # unit vector in the xy plane
    time: float  # when it starts [s]
    duration: float = 0.05

    @property
    def force(self) -> np.ndarray:
        return self.direction * (self.impulse / self.duration)


@dataclass
class RunStats:
    duration: float
    steps: int
    fell: bool
    pelvis_height: float
    com_error_mean: float
    com_error_max: float
    tilt_mean: float  # rad
    tilt_max: float  # rad
    torque_peak: float
    solve_ms_mean: float
    solve_ms_p95: float
    realtime_factor: float = 0.0

    def report(self) -> str:
        return "\n".join(
            [
                f"simulated       : {self.duration:.2f} s over {self.steps} control steps",
                f"fell            : {self.fell}",
                f"pelvis height   : {self.pelvis_height:.4f} m",
                f"CoM error       : mean {self.com_error_mean * 1e3:.2f} mm, "
                f"max {self.com_error_max * 1e3:.2f} mm",
                f"torso tilt      : mean {np.degrees(self.tilt_mean):.3f} deg, "
                f"max {np.degrees(self.tilt_max):.3f} deg",
                f"peak |torque|   : {self.torque_peak:.1f} Nm",
                f"controller time : mean {self.solve_ms_mean:.2f} ms, "
                f"p95 {self.solve_ms_p95:.2f} ms",
            ]
        )


class Simulation:
    """One controller driving one MuJoCo model.

    ``props`` (slot count per kind) and ``camera`` (an ``EgoCameraSpec``
    recipe) build the plant with :mod:`loka.scene_props`. Free-joint props sit
    after the robot in ``qpos``, so anything that hands state to the robot
    reads :attr:`robot_qpos` / :attr:`robot_qvel`, never the full vectors.
    """

    def __init__(
        self,
        config: LocomotionConfig | None = None,
        model_path: str | Path | None = None,
        pushes: list[Push] | None = None,
        *,
        props: dict[str, int] | None = None,
        static_props=None,
        camera: dict | None = None,
    ) -> None:
        self.config = config or LocomotionConfig()
        if model_path is not None:
            self.config.model_path = str(model_path)
        from loka.control.stacks import make as make_stack

        self.controller = make_stack(self.config)
        self._load_plant(props, static_props, camera)
        self.steps_per_control = max(
            1, int(round(self.config.control_dt / self.model.opt.timestep))
        )
        self.pushes = list(pushes or [])
        self.history: list[dict] = []
        self.reset()

    def _load_plant(self, props, static_props, camera) -> None:
        from loka.scene_props import (
            DEFAULT_STATIC_KINDS,
            PropManager,
            build_plant_model,
            plant_recipe,
        )

        self.recipe = plant_recipe(
            self.config.model_path,
            props,
            DEFAULT_STATIC_KINDS if static_props is None else static_props,
            camera,
        )
        self.model, self.prop_layout = build_plant_model(self.recipe)
        self.data = mujoco.MjData(self.model)
        self.pelvis_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )
        self.robot_nq = self.prop_layout.robot_nq
        self.robot_nv = self.prop_layout.robot_nv
        self.props = PropManager(self) if self.prop_layout.slots else None

    @property
    def robot_qpos(self) -> np.ndarray:
        return self.data.qpos[: self.robot_nq]

    @property
    def robot_qvel(self) -> np.ndarray:
        return self.data.qvel[: self.robot_nv]

    @property
    def has_camera(self) -> bool:
        return bool(self.recipe.get("camera"))

    def snapshot(self) -> dict:
        """Copy of the plant state for the ego renderer (cheap; no rendering)."""
        from loka.scene_props import plant_snapshot

        return plant_snapshot(self.model, self.data, self.prop_layout)

    def apply_prop_requests(self) -> None:
        if self.props is not None:
            for message in self.props.apply_pending():
                print(message)

    def reset(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        if getattr(self, "props", None) is not None:
            self.props.reset()
        mujoco.mj_forward(self.model, self.data)
        self.controller.reset()
        self.history.clear()
        self.torque = np.zeros(self.model.nu)

    # -- stepping ---------------------------------------------------------

    def _applied_force(self) -> np.ndarray:
        now = self.data.time
        total = np.zeros(3)
        for push in self.pushes:
            if push.time <= now < push.time + push.duration:
                total[:2] += push.force[:2]
        return total

    def step(self) -> None:
        """Advance one control period, holding the torque across substeps."""
        self.apply_prop_requests()
        self.torque = self.controller.compute_torque(self.robot_qpos, self.robot_qvel)
        force = self._applied_force()
        for _ in range(self.steps_per_control):
            self.data.ctrl[:] = self.torque
            self.data.xfrc_applied[self.pelvis_id, :3] = force
            mujoco.mj_step(self.model, self.data)
        self.data.xfrc_applied[self.pelvis_id, :3] = 0.0

        telemetry = self.controller.telemetry
        self.history.append(
            {
                "t": self.data.time,
                "com_error": float(np.linalg.norm(telemetry.com_error)),
                "tilt": float(np.linalg.norm(telemetry.rpy[:2])),
                "torque": float(np.abs(telemetry.torque).max()),
                "solve_ms": telemetry.solve_ms,
                "pelvis_z": float(self.data.qpos[2]),
            }
        )

    @property
    def fell(self) -> bool:
        return bool(self.data.qpos[2] < FALL_HEIGHT)

    def run(self, duration: float, stop_on_fall: bool = True) -> RunStats:
        while self.data.time < duration:
            self.step()
            if stop_on_fall and self.fell:
                break
        return self.stats()

    # -- results ----------------------------------------------------------

    def stats(self) -> RunStats:
        if not self.history:
            raise RuntimeError("no steps recorded")
        columns = {
            key: np.array([row[key] for row in self.history])
            for key in self.history[0]
        }
        # Ignore the first second so start-up transients do not dominate.
        settled = columns["t"] > min(1.0, columns["t"][-1] / 2.0)
        return RunStats(
            duration=float(columns["t"][-1]),
            steps=len(self.history),
            fell=self.fell,
            pelvis_height=float(columns["pelvis_z"][settled].mean()),
            com_error_mean=float(columns["com_error"][settled].mean()),
            com_error_max=float(columns["com_error"][settled].max()),
            tilt_mean=float(columns["tilt"][settled].mean()),
            tilt_max=float(columns["tilt"][settled].max()),
            torque_peak=float(columns["torque"].max()),
            solve_ms_mean=float(columns["solve_ms"].mean()),
            solve_ms_p95=float(np.percentile(columns["solve_ms"], 95)),
        )


class UnitreeRLSimulation(Simulation):
    """The unitree_rl_lab velocity policy driving the same MuJoCo scene.

    One :meth:`step` is still ``config.control_dt``. The policy runs every
    ``step_dt`` (50 Hz for v0) and the PD law on every physics substep, as the
    motor drivers do on the robot.
    """

    def __init__(
        self,
        config: LocomotionConfig | None = None,
        model_path: str | Path | None = None,
        pushes: list[Push] | None = None,
        policy_dir: str | Path | None = None,
        min_speed: float | None = None,
        *,
        props: dict[str, int] | None = None,
        static_props=None,
        camera: dict | None = None,
    ) -> None:
        from loka.control.unitree_rl import MIN_WALK_SPEED, UnitreeRLController

        self.config = config or LocomotionConfig()
        if model_path is not None:
            self.config.model_path = str(model_path)
        self._load_plant(props, static_props, camera)
        self.controller = UnitreeRLController(
            self.model,
            policy_dir,
            min_speed=MIN_WALK_SPEED if min_speed is None else float(min_speed),
        )
        self.steps_per_control = max(
            1, int(round(self.config.control_dt / self.model.opt.timestep))
        )
        self.policy_decimation = max(
            1, int(round(self.controller.policy.step_dt / self.config.control_dt))
        )
        self.pushes = list(pushes or [])
        self.history: list[dict] = []
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        if getattr(self, "props", None) is not None:
            self.props.reset()
        self.controller.reset()
        self.controller.place_default_pose(self.data)
        self.history.clear()
        self.torque = np.zeros(self.model.nu)
        self._ticks = 0

    def step(self) -> None:
        self.apply_prop_requests()
        if self._ticks % self.policy_decimation == 0:
            self.controller.policy_step(self.robot_qpos, self.robot_qvel)
        self._ticks += 1
        force = self._applied_force()
        for _ in range(self.steps_per_control):
            self.torque = self.controller.compute_torque(self.robot_qpos, self.robot_qvel)
            self.data.ctrl[:] = self.torque
            self.data.xfrc_applied[self.pelvis_id, :3] = force
            mujoco.mj_step(self.model, self.data)
        self.data.xfrc_applied[self.pelvis_id, :3] = 0.0

        telemetry = self.controller.telemetry
        self.history.append(
            {
                "t": self.data.time,
                "com_error": 0.0,
                "tilt": float(np.linalg.norm(telemetry.rpy[:2])),
                "torque": float(np.abs(self.torque).max()),
                "solve_ms": telemetry.solve_ms,
                "pelvis_z": float(self.data.qpos[2]),
            }
        )


def push_sequence(
    impulse: float, period: float, count: int, seed: int = 0, start: float = 2.0
) -> list[Push]:
    """Evenly spaced pushes in random horizontal directions."""
    rng = np.random.default_rng(seed)
    pushes = []
    for i in range(count):
        angle = rng.uniform(0.0, 2.0 * np.pi)
        pushes.append(
            Push(
                impulse=impulse,
                direction=np.array([np.cos(angle), np.sin(angle), 0.0]),
                time=start + i * period,
            )
        )
    return pushes
