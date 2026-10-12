"""One offscreen rollout. No viewer and no realtime throttle."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from loka.harness.storage import EpisodeStore


@dataclass
class Event:
    time: float
    kind: str
    payload: Any = None


def run_episode(
    agent,
    *,
    directory: Path,
    events: list[Event],
    timeout_s: float,
    log_hz: float,
    record_fps: float,
    metadata: dict[str, Any],
    success_fn: Callable[[Any], bool],
    encode_video: bool,
    recording,
    model_path: str,
) -> dict[str, Any]:
    """Step ``agent`` until success, a fall, or ``timeout_s``.

    Scheduled events (a goal command, a perturbation) fire when sim time
    reaches them, before that step's control update.
    """
    directory = Path(directory)
    store = EpisodeStore(directory, log_hz=log_hz, record_fps=record_fps)
    pending = sorted(events, key=lambda event: event.time)
    for event in pending:
        if event.kind == "perturbation" and event.payload is not None:
            event.payload.prepare_model(agent.sim.model)

    outcome = "timeout"
    t_first_fall = None
    wall0 = time.perf_counter()
    store.maybe_log(agent)
    try:
        while True:
            now = float(agent.sim.data.time)
            while pending and pending[0].time <= now + 1e-9:
                _fire(agent, pending.pop(0), store)
            if now + 1e-9 >= float(timeout_s):
                outcome = "timeout"
                break
            agent.step()
            store.maybe_log(agent)
            now = float(agent.sim.data.time)
            if agent.fell and t_first_fall is None:
                t_first_fall = now
                outcome = "fell"
                store.event({"t": now, "kind": "fell"})
                break
            if success_fn(agent):
                outcome = "success"
                store.event({"t": now, "kind": "arrived"})
                break
            if now + 1e-9 >= float(timeout_s):
                outcome = "timeout"
                break
    finally:
        wall = time.perf_counter() - wall0
        meta = {
            **metadata,
            "outcome": outcome,
            "t_end": float(agent.sim.data.time),
            "fell": bool(t_first_fall is not None or agent.fell),
            "t_first_fall": t_first_fall,
            "wall_s": wall,
            "llm_turn_count": len(agent.turns),
            "episode_dir": str(directory),
        }
        if agent.goal_xy is not None:
            pelvis = agent.sim.data.qpos[:2]
            meta["final_goal_error_m"] = float(
                ((pelvis[0] - agent.goal_xy[0]) ** 2 + (pelvis[1] - agent.goal_xy[1]) ** 2)
                ** 0.5
            )
        store.write(metadata=meta, turns=list(agent.turns))
        if encode_video:
            _encode(directory, recording, model_path)
        agent.close()
    print(
        f"  [episode] {meta['outcome']} t={meta['t_end']:.2f}s "
        f"wall={meta['wall_s']:.1f}s -> {directory}"
    )
    return meta


def _fire(agent, event: Event, store: EpisodeStore) -> None:
    now = float(agent.sim.data.time)
    if event.kind == "goal":
        xy, speed, radius = event.payload
        agent.command_goal(xy, speed, radius)
        store.event(
            {
                "t": now,
                "kind": "goal",
                "xy": [float(xy[0]), float(xy[1])],
                "speed": float(speed),
                "radius": float(radius),
            }
        )
        return
    if event.kind == "directive":
        text, xy, speed, radius = event.payload
        agent.submit_directive(text, xy)
        store.event(
            {
                "t": now,
                "kind": "directive",
                "text": text,
                "xy": [float(xy[0]), float(xy[1])],
                "speed": float(speed),
                "radius": float(radius),
            }
        )
        return
    if event.kind == "perturbation":
        record = event.payload.apply(agent)
        store.event({"t": now, "kind": "perturbation", "overlay": record})
        if record:
            store.overlay({"t": now, **record})
        return
    raise ValueError(f"Unknown episode event {event.kind!r}")


def _encode(directory: Path, recording, model_path: str) -> None:
    try:
        from loka.harness.render import render_episode_dir

        render_episode_dir(directory, recording=recording, model_path=model_path)
    except Exception as exc:
        print(f"[eval] video encode failed ({exc}); timeseries still saved")
