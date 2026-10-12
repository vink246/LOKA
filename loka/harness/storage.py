"""Per-episode files. Metrics read these back, so a new metric does not need a rerun."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


@dataclass
class EpisodeData:
    directory: Path
    metadata: dict[str, Any]
    timeseries: dict[str, np.ndarray] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    turns: list[dict[str, Any]] = field(default_factory=list)

    def column(self, name: str) -> np.ndarray:
        if name not in self.timeseries:
            return np.zeros(0)
        return np.asarray(self.timeseries[name])


class EpisodeStore:
    """Buffers one rollout and writes it on :meth:`write`."""

    def __init__(self, directory: Path, *, log_hz: float, record_fps: float) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log_interval = 1.0 / float(log_hz)
        self.next_log_time = 0.0
        self.frame_interval = 1.0 / float(record_fps)
        self.next_frame_time = 0.0
        self.rows: list[dict[str, np.ndarray | float]] = []
        self.render_times: list[float] = []
        self.render_qpos: list[np.ndarray] = []
        self.events: list[dict[str, Any]] = []
        self.overlays: list[dict[str, Any]] = []

    def event(self, record: dict[str, Any]) -> None:
        self.events.append(dict(record))

    def overlay(self, record: dict[str, Any]) -> None:
        self.overlays.append(dict(record))

    def maybe_log(self, agent) -> None:
        data = agent.sim.data
        if data.time + 1e-9 >= self.next_frame_time:
            self.render_times.append(float(data.time))
            self.render_qpos.append(np.array(data.qpos, dtype=np.float64, copy=True))
            while self.next_frame_time <= data.time + 1e-9:
                self.next_frame_time += self.frame_interval
        if data.time + 1e-9 < self.next_log_time:
            return
        self.rows.append(_sample(agent))
        while self.next_log_time <= data.time + 1e-9:
            self.next_log_time += self.log_interval

    def write(
        self,
        *,
        metadata: dict[str, Any],
        turns: list[dict[str, Any]],
    ) -> None:
        meta_path = self.directory / "metadata.json"
        meta_path.write_text(
            json.dumps(json_ready(metadata), indent=2), encoding="utf-8"
        )
        _write_events(self.directory / "events.jsonl", self.events)
        _write_events(self.directory / "loka_turns.jsonl", turns)
        if self.rows:
            _write_timeseries(self.directory, self.rows)
        qpos = (
            np.stack(self.render_qpos, axis=0)
            if self.render_qpos
            else np.zeros((0, 1))
        )
        times = np.asarray(self.render_times, dtype=np.float64)
        np.savez_compressed(
            self.directory / "render.npz",
            times=times,
            qpos=qpos,
            overlays_json=np.array(json.dumps(json_ready(self.overlays))),
        )


def _sample(agent) -> dict[str, Any]:
    sim = agent.sim
    data = sim.data
    tel = sim.controller.telemetry
    nu = int(sim.model.nu)
    torque = np.zeros(nu) if tel is None else np.asarray(tel.torque, dtype=float)
    rpy = np.zeros(3) if tel is None else np.asarray(tel.rpy, dtype=float)
    com_error = np.zeros(3) if tel is None else np.asarray(tel.com_error, dtype=float)
    if tel is None or tel.contact_forces is None:
        contact = np.zeros((8, 3))
    else:
        contact = np.asarray(tel.contact_forces, dtype=float)
    goal = agent.goal_xy
    pelvis = np.asarray(data.qpos[:2], dtype=float)
    goal_dist = float(np.linalg.norm(pelvis - goal)) if goal is not None else float("nan")
    bias = agent.bias_nm()
    return {
        "t": float(data.time),
        "qpos": np.array(data.qpos, dtype=np.float64, copy=True),
        "qvel": np.array(data.qvel, dtype=np.float64, copy=True),
        "ctrl": np.array(data.ctrl, dtype=np.float64, copy=True),
        "torque": np.asarray(torque, dtype=np.float64),
        "rpy": np.asarray(rpy, dtype=np.float64),
        "com_error": np.asarray(com_error, dtype=np.float64),
        "pelvis_xyz": np.array(data.qpos[:3], dtype=np.float64, copy=True),
        "fallen": float(agent.fell),
        "contact_forces": np.asarray(contact, dtype=np.float64),
        "goal_distance": goal_dist,
        "gait_mode": float(sim.controller.gait.config.mode),
        "bias_nm": float("nan") if bias is None else float(bias),
        "mutations": float(agent.mutation_count()),
    }


def _write_timeseries(directory: Path, rows: list[dict[str, Any]]) -> None:
    stacked: dict[str, np.ndarray] = {}
    for key in rows[0]:
        stacked[key] = np.stack([np.asarray(row[key]) for row in rows], axis=0)
    np.savez_compressed(directory / "timeseries.npz", **stacked)
    scalar_keys = [
        key
        for key, value in stacked.items()
        if value.ndim == 1
    ]
    csv_path = directory / "timeseries.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        for index in range(stacked["t"].shape[0]):
            writer.writerow({key: float(stacked[key][index]) for key in scalar_keys})


def _write_events(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_ready(row)) + "\n")


def load_episode(directory: str | Path) -> EpisodeData:
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    series: dict[str, np.ndarray] = {}
    npz = directory / "timeseries.npz"
    if npz.is_file():
        with np.load(npz) as blob:
            series = {key: blob[key] for key in blob.files}
    return EpisodeData(
        directory=directory,
        metadata=metadata,
        timeseries=series,
        events=_read_jsonl(directory / "events.jsonl"),
        turns=_read_jsonl(directory / "loka_turns.jsonl"),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_unit_episodes(unit_dir: Path) -> list[EpisodeData]:
    """Probe subdirectories, or the unit directory itself for a single rollout."""
    unit_dir = Path(unit_dir)
    probes = sorted(path for path in unit_dir.glob("probe_*") if path.is_dir())
    if probes:
        return [load_episode(path) for path in probes if (path / "metadata.json").is_file()]
    if (unit_dir / "metadata.json").is_file():
        return [load_episode(unit_dir)]
    episode = unit_dir / "episode"
    if (episode / "metadata.json").is_file():
        return [load_episode(episode)]
    return []


def primary_episode(episodes: list[EpisodeData]) -> EpisodeData:
    """The rollout an episode-scoped metric should read.

    A search trial has many probes. The representative one is the strongest
    push that was survived, or the last probe when none were.
    """
    if len(episodes) == 1:
        return episodes[0]
    survived = [ep for ep in episodes if not ep.metadata.get("fell")]
    if not survived:
        return episodes[-1]
    return max(survived, key=lambda ep: float(ep.metadata.get("push_force_n") or 0.0))
