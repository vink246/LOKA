"""Write per-episode time series, metadata, and video for the Walker suite."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from loka.recording import QposClipSampler, render_qpos_clip
from loka.walker_suite.forces import (
    ground_reaction_channels,
    nonfoot_geom_names,
    sample_floor_contacts,
)
from loka.walker_suite.outcomes import has_fallen, pitch, pos_x, world_height

ACTUATOR_FALLBACK = (
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


class EpisodeLogger:
    def __init__(
        self,
        directory: Path,
        *,
        log_hz: float,
        record: bool,
        record_fps: float,
        record_width: int,
        record_height: int,
        record_camera: str,
        model_path: str,
        actuator_names: list[str] | None = None,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log_interval = 1.0 / float(log_hz)
        self.next_log_time = 0.0
        self.record = bool(record)
        self.record_fps = float(record_fps)
        self.record_width = int(record_width)
        self.record_height = int(record_height)
        self.record_camera = record_camera
        self.model_path = model_path
        self.actuator_names = list(actuator_names or ACTUATOR_FALLBACK)
        self.sampler = QposClipSampler(self.record_fps) if self.record else None
        self.rows: list[dict[str, Any]] = []
        self._nonfoot_names: tuple[str, ...] | None = None

    def maybe_log(self, runtime, step) -> None:
        data = runtime.data
        if self.sampler is not None:
            self.sampler.maybe_capture(data)
        if data.time + 1e-9 < self.next_log_time:
            return

        qpos = np.asarray(data.qpos, dtype=float)
        qvel = np.asarray(data.qvel, dtype=float)
        ctrl = np.asarray(data.ctrl, dtype=float)
        planner_cmd = np.asarray(step.frame.get("planner_cmd", ctrl), dtype=float)
        torque = step.frame.get("actuator_torque")
        if torque is None:
            torque = data.actuator_force * runtime.model.actuator_gear[:, 0]
        torque = np.asarray(torque, dtype=float)
        if self._nonfoot_names is None:
            self._nonfoot_names = nonfoot_geom_names(runtime.model)
        force, touching = sample_floor_contacts(runtime.model, data)
        touching_set = set(touching)
        grf = ground_reaction_channels(force)
        row: dict[str, Any] = {
            "t": float(data.time),
            "pitch": pitch(data),
            "height": world_height(data),
            "pos_x": pos_x(data),
            "vel_x": float(qvel[1]) if qvel.size > 1 else 0.0,
            "fault_active": int(step.fault_active),
            "tracking_error": float(step.error),
            "in_failure": int(step.in_failure),
            "fallen": int(has_fallen(data)),
            "grf_mag": grf["grf_mag"],
            "grf_vertical": grf["grf_vertical"],
            "grf_horizontal": grf["grf_horizontal"],
            "nonfoot_floor": int(bool(touching_set)),
        }
        for name in self._nonfoot_names:
            row[f"floor_{name}"] = int(name in touching_set)
        n_act = min(
            len(self.actuator_names),
            qpos.size - 3,
            torque.size,
            ctrl.size,
            planner_cmd.size,
        )
        for i in range(n_act):
            name = self.actuator_names[i]
            row[f"pos_{name}"] = float(qpos[3 + i])
            row[f"cmd_{name}"] = float(planner_cmd[i])
            row[f"ctrl_{name}"] = float(ctrl[i])
            row[f"torque_{name}"] = float(torque[i])
        self.rows.append(row)
        while self.next_log_time <= data.time + 1e-9:
            self.next_log_time += self.log_interval

    def write(
        self,
        *,
        metadata: dict[str, Any],
        mpc_snapshots: list[dict[str, Any]],
        loka_turns: list[dict[str, Any]],
        obstacle_overlay: dict[str, Any] | None = None,
        visual_overlay: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        meta_path = self.directory / "metadata.json"
        meta_path.write_text(
            json.dumps(_json_ready(metadata), indent=2), encoding="utf-8"
        )

        if self.rows:
            csv_path = self.directory / "timeseries.csv"
            fieldnames = list(self.rows[0].keys())
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.rows)
            arrays = {key: np.array([row[key] for row in self.rows]) for key in fieldnames}
            np.savez_compressed(self.directory / "timeseries.npz", **arrays)

        mpc_path = self.directory / "mpc_params.jsonl"
        with mpc_path.open("w", encoding="utf-8") as handle:
            for snap in mpc_snapshots:
                handle.write(json.dumps(_json_ready(snap)) + "\n")

        turns_path = self.directory / "loka_turns.jsonl"
        with turns_path.open("w", encoding="utf-8") as handle:
            for turn in loka_turns:
                handle.write(json.dumps(_json_ready(turn)) + "\n")

        video_path = None
        if self.sampler is not None and self.sampler.frame_count:
            video_path = self.directory / "episode.mp4"
            times, qpos = self.sampler.stacked()
            try:
                render_qpos_clip(
                    model_path=self.model_path,
                    qpos_frames=qpos,
                    times=times,
                    out_path=video_path,
                    obstacle=obstacle_overlay,
                    visual=visual_overlay if visual_overlay is not None else obstacle_overlay,
                    camera=self.record_camera,
                    fps=self.record_fps,
                    width=self.record_width,
                    height=self.record_height,
                )
            except Exception as exc:
                print(f"[walker_suite] video encode failed ({exc}); timeseries still saved")
                video_path = None

        return {"video_path": str(video_path) if video_path else None}
