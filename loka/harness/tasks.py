"""Tasks. Each one turns a work unit into one or more episodes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from loka.harness.baselines import build_baseline
from loka.harness.episode import Event, run_episode
from loka.harness.perturbations import make_perturbation, make_push
from loka.harness.registry import register_task
from loka.harness.search import FellBeforePush, run_force_search
from loka.harness.storage import json_ready


def _rewrite(directory: Path, metadata: dict) -> None:
    (Path(directory) / "metadata.json").write_text(
        json.dumps(json_ready(metadata), indent=2), encoding="utf-8"
    )


def _run(agent, suite, directory: Path, events, timeout_s, metadata, success_fn, encode: bool):
    return run_episode(
        agent,
        directory=directory,
        events=events,
        timeout_s=timeout_s,
        log_hz=suite.log_hz,
        record_fps=suite.recording.fps,
        metadata=metadata,
        success_fn=success_fn,
        encode_video=encode,
        recording=suite.recording,
        model_path=str(agent.sim.config.model_path),
    )


@register_task("goto_goal")
class GotoGoalTask:
    """Walk to a world point. Success is arriving inside the radius and holding."""

    def __init__(self, spec) -> None:
        self.spec = spec

    def run(self, suite, unit, unit_dir: Path) -> dict:
        raw = self.spec.raw
        goal = raw["goal"]
        xy = (float(goal["xy"][0]), float(goal["xy"][1]))
        speed = float(goal.get("speed", 0.2))
        radius = float(goal.get("radius", 0.25))
        hold_s = float(goal.get("hold_s", 0.0))
        command_at = float(raw.get("command_time_s", 0.5))
        perturb_at = float(raw.get("perturbation_time_s", command_at))
        timeout_s = float(raw["timeout_s"])
        goal_via = str(raw.get("goal_via", "task_targets"))
        perturbation = make_perturbation(self.spec.perturbation(unit.perturbation))
        log_path = unit_dir / "session.log"
        agent = build_baseline(suite.baseline(unit.baseline), suite, log_path=log_path)
        agent.reset(unit.seed, suite.init_noise)
        if goal_via == "directive" and agent.supports_directive:
            text = (
                f"Walk to the world coordinate ({xy[0]:g}, {xy[1]:g}) "
                f"at {speed:g} m/s and stop within {radius:g} m."
            )
            goal_event = Event(command_at, "directive", (text, xy, speed, radius))
            used_via = "directive"
        else:
            goal_event = Event(command_at, "goal", (xy, speed, radius))
            used_via = "task_targets"
        held = {"since": None}

        def arrived(agent) -> bool:
            if agent.goal_xy is None:
                return False
            dist = float(np.linalg.norm(agent.sim.data.qpos[:2] - agent.goal_xy))
            now = float(agent.sim.data.time)
            if dist <= radius:
                if held["since"] is None:
                    held["since"] = now
                return now - held["since"] + 1e-9 >= hold_s
            held["since"] = None
            return False

        meta = _run(
            agent,
            suite,
            unit_dir,
            [goal_event, Event(perturb_at, "perturbation", perturbation)],
            timeout_s,
            {
                "task": unit.task,
                "perturbation": unit.perturbation,
                "baseline": unit.baseline,
                "trial": unit.trial,
                "seed": unit.seed,
                "init_noise": suite.init_noise,
                "goal_xy": [xy[0], xy[1]],
                "goal_speed": speed,
                "goal_radius": radius,
                "goal_via": used_via,
            },
            arrived,
            encode=bool(suite.recording.enabled),
        )
        if meta.get("outcome") == "success":
            meta["time_to_goal_s"] = meta["t_end"]
        else:
            meta["time_to_goal_s"] = None
        _rewrite(unit_dir, meta)
        return meta


@register_task("stand_push")
class StandPushTask:
    """Asymmetric mass, then the largest push the robot survives.

    Every force is a fresh episode from the same seed. The search doubles the
    upper bound until the robot falls, then bisects to ``resolution_n``.
    """

    def __init__(self, spec) -> None:
        self.spec = spec

    def run(self, suite, unit, unit_dir: Path) -> dict:
        raw = self.spec.raw
        mass_time = float(raw.get("mass_time_s", 2.0))
        push_delay = float(raw.get("push_delay_s", 20.0))
        observe_s = float(raw.get("observe_s", 10.0))
        push_cfg = dict(raw.get("push") or {})
        direction = list(push_cfg.get("direction", [1.0, 0.0]))
        duration_s = float(push_cfg.get("duration_s", 0.05))
        search = dict(raw.get("search") or {})
        lo = float(search.get("lo_n", 0.0))
        hi = float(search.get("hi_n", 200.0))
        max_hi = float(search.get("max_hi_n", 2000.0))
        resolution = float(search.get("resolution_n", 0.2))
        mass = make_perturbation(self.spec.perturbation(unit.perturbation))
        encode_each = bool(
            suite.recording.enabled and suite.recording.stand_push_probes == "all"
        )
        timeout_s = mass_time + push_delay + observe_s
        push_time = mass_time + push_delay
        records: list[dict] = []
        index = {"n": 0}

        def probe(force: float) -> bool:
            probe_dir = unit_dir / f"probe_{index['n']:02d}"
            index["n"] += 1
            agent = build_baseline(
                suite.baseline(unit.baseline),
                suite,
                log_path=probe_dir / "session.log",
            )
            agent.reset(unit.seed, suite.init_noise)
            events = [
                Event(mass_time, "perturbation", mass),
                Event(push_time, "perturbation", make_push(force, direction, duration_s)),
            ]
            meta = _run(
                agent,
                suite,
                probe_dir,
                events,
                timeout_s,
                {
                    "task": unit.task,
                    "perturbation": unit.perturbation,
                    "baseline": unit.baseline,
                    "trial": unit.trial,
                    "seed": unit.seed,
                    "init_noise": suite.init_noise,
                    "push_force_n": float(force),
                    "push_time_s": push_time,
                    "mass_time_s": mass_time,
                },
                lambda _agent: False,
                encode=encode_each,
            )
            fell_before = (
                meta.get("t_first_fall") is not None
                and float(meta["t_first_fall"]) + 1e-9 < push_time
            )
            meta["fell_before_push"] = bool(fell_before)
            meta["push_force_n"] = float(force)
            _rewrite(probe_dir, meta)
            records.append(meta)
            if fell_before:
                raise FellBeforePush()
            return bool(meta["fell"])

        print(f"[eval] {unit.id} push search {lo:.2f}..{hi:.2f} N")
        threshold = run_force_search(
            probe, lo=lo, hi=hi, resolution=resolution, max_hi=max_hi
        )
        if suite.recording.enabled and suite.recording.stand_push_probes == "boundary":
            _encode_boundary(records, suite)
        wall = float(sum(float(row.get("wall_s") or 0.0) for row in records))
        return {
            "task": unit.task,
            "perturbation": unit.perturbation,
            "baseline": unit.baseline,
            "trial": int(unit.trial),
            "seed": int(unit.seed),
            "init_noise": float(suite.init_noise),
            "outcome": "measured",
            "max_push_force_n": float(threshold),
            "n_probes": len(records),
            "wall_s": wall,
            "probes": [
                {
                    "push_force_n": row["push_force_n"],
                    "fell": row["fell"],
                    "fell_before_push": row["fell_before_push"],
                    "t_end": row["t_end"],
                    "episode_dir": row["episode_dir"],
                }
                for row in records
            ],
        }


def _encode_boundary(records: list[dict], suite) -> None:
    from loka.harness.render import render_episode_dir

    survived = [row for row in records if not row["fell"]]
    fallen = [
        row for row in records if row["fell"] and not row.get("fell_before_push")
    ]
    chosen: list[dict] = []
    if survived:
        chosen.append(max(survived, key=lambda row: float(row["push_force_n"])))
    if fallen:
        chosen.append(min(fallen, key=lambda row: float(row["push_force_n"])))
    if not chosen and records:
        chosen.append(records[0])
    seen: set[str] = set()
    for row in chosen:
        directory = row["episode_dir"]
        if directory in seen:
            continue
        seen.add(directory)
        try:
            render_episode_dir(
                directory,
                recording=suite.recording,
                model_path=_model_path(suite),
            )
        except Exception as exc:
            print(f"[eval] video encode failed ({exc}); render.npz still saved")


def _model_path(suite) -> str:
    from loka.harness.baselines import locomotion_config

    return str(locomotion_config(suite).model_path)
