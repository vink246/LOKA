"""MuJoCo simulation harness for the standing controller.

The controller reads ``qpos`` / ``qvel`` straight out of ``MjData`` and writes
torques into ``ctrl`` -- one process, no state estimator, no network. That
keeps the control law the only thing under test.
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
    """One controller driving one MuJoCo model."""

    def __init__(
        self,
        config: LocomotionConfig | None = None,
        model_path: str | Path | None = None,
        pushes: list[Push] | None = None,
    ) -> None:
        self.config = config or LocomotionConfig()
        if model_path is not None:
            self.config.model_path = str(model_path)
        from loka.control.stacks import make as make_stack

        self.controller = make_stack(self.config)

        self.model = mujoco.MjModel.from_xml_path(self.config.model_path)
        self.data = mujoco.MjData(self.model)
        self.pelvis_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )
        self.steps_per_control = max(
            1, int(round(self.config.control_dt / self.model.opt.timestep))
        )
        self.pushes = list(pushes or [])
        self.history: list[dict] = []
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
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
        self.torque = self.controller.compute_torque(self.data.qpos, self.data.qvel)
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
