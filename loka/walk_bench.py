"""Closed-loop robustness scenarios for turning, gait changes and pushes.

    python -m loka.walk_bench
    python -m loka.walk_bench --only T1
    python -m loka.walk_bench --json-out logs/bench/baseline.json

Each scenario is a timed list of ``set_task_targets`` calls plus optional
pelvis impulses. Pass criteria live in :func:`evaluate`; the JSON dump is
the before/after record the tuning phases compare against.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from loka.control.gait import TURN_CAP_STAND, heading_frame, wrap_angle
from loka.control.robot import quat_to_rpy
from loka.sim import Push, Simulation

TURN_RATE = 0.4  # rad/s, the governor default the pass window is sized to


@dataclass
class Scenario:
    name: str
    duration: float
    script: list[tuple[float, dict]]
    pushes: list[Push] = field(default_factory=list)
    #: Heading the pelvis should reach, and the sim time the command is issued.
    heading_goal: float | None = None
    heading_from: float = 0.0
    #: Unwrapped rotation the deadline is sized to. None uses the short arc.
    heading_delta: float | None = None
    heading_at: float = 0.0
    #: Forward speed the settled walk should track [m/s]. None skips the check.
    speed_goal: float | None = None
    #: If set, the robot must be standing (gait idle, all sites planted) by then.
    stand_by: float | None = None
    group: str = "turn"


def _walk(speed: float, **extra) -> dict:
    return {"gait.mode": "walk", "gait.speed": speed, **extra}


def scenarios() -> list[Scenario]:
    """The Phase 0 matrix. Names are stable; tests and the CLI select by them."""
    out = [
        Scenario("T1", 12.0, [(0.5, _walk(0.20)), (3.0, {"gait.heading": 0.3})],
                 heading_goal=0.3, heading_at=3.0, speed_goal=0.20),
        Scenario("T2", 16.0, [(0.5, _walk(0.20)), (3.0, {"gait.heading": np.pi / 2})],
                 heading_goal=np.pi / 2, heading_at=3.0, speed_goal=0.20),
        Scenario("T3", 20.0, [(0.5, _walk(0.20)), (3.0, {"gait.heading": np.pi})],
                 heading_goal=np.pi, heading_at=3.0, speed_goal=0.20),
        Scenario("T4", 16.0, [(0.5, _walk(0.40)), (3.0, {"gait.heading": -np.pi / 2})],
                 heading_goal=-np.pi / 2, heading_at=3.0, speed_goal=0.40),
        Scenario("T5", 16.0, [(0.5, _walk(0.0, **{"gait.heading": np.pi / 2}))],
                 heading_goal=np.pi / 2, heading_at=0.5, speed_goal=0.0),
        Scenario(
            "T6", 24.0,
            [(0.5, _walk(0.20))]
            + [(0.5 + i * 0.5, {"gait.heading": 0.3 * (0.5 + i * 0.5)}) for i in range(40)],
            heading_goal=0.3 * 20.0, heading_delta=6.0, heading_at=0.5, speed_goal=0.20,
        ),
        Scenario(
            "T7", 14.0,
            [(0.5, _walk(0.20, **{"gait.heading": 3.0})), (6.0, {"gait.heading": -3.0})],
            heading_goal=-3.0, heading_from=3.0, heading_at=6.0, speed_goal=0.20,
        ),
        Scenario(
            "G1", 17.0,
            [(0.5, _walk(0.10)), (5.5, {"gait.speed": 0.50}), (10.5, {"gait.speed": 0.10})],
            speed_goal=0.10, group="gait",
        ),
        Scenario(
            "G2", 18.0,
            [
                (0.5, _walk(0.20)),
                (4.0, {"gait.step_period": 0.80}),
                (7.0, {"gait.step_period": 0.60}),
                (10.0, {"gait.duty_factor": 0.76}),
                (13.0, {"gait.stance_width": 0.28}),
                (15.5, {"gait.swing_height": 0.08, "gait.stance_width": 0.22}),
            ],
            speed_goal=0.20, group="gait",
        ),
        Scenario(
            "G3", 14.0,
            [(0.5, _walk(0.30)), (4.15, {"gait.mode": "stand"}), (9.0, _walk(0.20))],
            stand_by=7.15, speed_goal=0.20, group="gait",
        ),
        Scenario(
            "G4", 14.0,
            [(0.5, _walk(0.20)), (4.0, {"height": 0.62}), (9.0, {"height": 0.70})],
            speed_goal=0.20, group="gait",
        ),
        Scenario(
            "G5", 16.0,
            [(0.5, {"gait.mode": "tread", "gait.speed": 0.0}), (10.5, _walk(0.20))],
            speed_goal=0.20, group="gait",
        ),
        Scenario(
            "G5b", 12.0,
            [(0.5, {"gait.mode": "limp", "gait.speed": 0.40})],
            speed_goal=0.15, group="gait",
        ),
        Scenario(
            "G6", 14.0,
            [(0.5, _walk(0.20)), (4.0, {
                "gait.speed": 0.40, "gait.heading": 1.0, "gait.step_period": 0.70,
            })],
            heading_goal=1.0, heading_at=4.0, speed_goal=0.40, group="gait",
        ),
        Scenario("S0", 30.0, [(0.5, _walk(0.20))], speed_goal=0.20, group="regression"),
    ]
    out.extend(_push_matrix())
    return out


def _push_matrix() -> list[Scenario]:
    """P1: direction x phase x impulse while walking straight. P2: during a turn."""
    dirs = {
        "fwd": np.array([1.0, 0.0, 0.0]),
        "back": np.array([-1.0, 0.0, 0.0]),
        "left": np.array([0.0, 1.0, 0.0]),
        "right": np.array([0.0, -1.0, 0.0]),
    }
    # Offsets from a 4 s mark, aimed at DS / mid-swing / late swing of a ~0.4 s step.
    phases = {"ds": 0.00, "mid": 0.22, "late": 0.34}
    rows = []
    for name, direction in dirs.items():
        for phase, offset in phases.items():
            for impulse in (2.0, 4.0, 6.0, 8.0, 10.0, 12.0):
                rows.append(Scenario(
                    f"P1_{name}_{phase}_{impulse:.0f}",
                    8.0,
                    [(0.5, _walk(0.20))],
                    pushes=[Push(impulse, direction, time=4.0 + offset)],
                    speed_goal=0.20,
                    group="push",
                ))
    rows.append(Scenario(
        "P2", 12.0,
        [(0.5, _walk(0.20)), (3.0, {"gait.heading": np.pi / 2})],
        pushes=[Push(4.0, np.array([1.0, 0.0, 0.0]), time=5.0)],
        heading_goal=np.pi / 2, heading_at=3.0, speed_goal=0.20, group="push",
    ))
    return rows


@dataclass
class BenchResult:
    name: str
    fell: bool
    elapsed: float
    fall_time: float | None
    yaw_err_final: float | None
    yaw_err_time: float | None
    speed_achieved: float | None
    peak_roll: float
    peak_pitch: float
    knee_min: float
    stood: bool | None
    passed: bool
    reasons: list[str]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "fell": self.fell,
            "elapsed": self.elapsed,
            "fall_time": self.fall_time,
            "yaw_err_final": self.yaw_err_final,
            "yaw_err_time": self.yaw_err_time,
            "speed_achieved": self.speed_achieved,
            "peak_roll": self.peak_roll,
            "peak_pitch": self.peak_pitch,
            "knee_min": self.knee_min,
            "stood": self.stood,
            "passed": self.passed,
            "reasons": self.reasons,
        }


def _pelvis_yaw(sim: Simulation) -> float:
    return float(quat_to_rpy(sim.data.qpos[3:7])[2])


def _forward_speed(sim: Simulation) -> float:
    """Pelvis linear speed along its own heading [m/s]."""
    yaw = _pelvis_yaw(sim)
    forward, _ = heading_frame(yaw)
    vel = np.asarray(sim.data.qvel[:2], dtype=float)
    return float(vel @ forward)


def run_scenario(scenario: Scenario) -> BenchResult:
    sim = Simulation(pushes=list(scenario.pushes))
    pending = list(scenario.script)
    yaw_err_time = None
    speed_samples: list[float] = []
    peak_roll = peak_pitch = 0.0
    knee_min = np.inf
    stood = None
    fall_time = None
    per_leg = 6

    while sim.data.time < scenario.duration and not sim.fell:
        now = float(sim.data.time)
        while pending and pending[0][0] <= now:
            _, updates = pending.pop(0)
            sim.controller.set_task_targets(updates)
        sim.step()
        rpy = sim.controller.telemetry.rpy
        peak_roll = max(peak_roll, abs(float(rpy[0])))
        peak_pitch = max(peak_pitch, abs(float(rpy[1])))
        qj = np.asarray(sim.data.qpos[7:7 + 12], dtype=float)
        knee_min = min(knee_min, float(qj[3]), float(qj[per_leg + 3]))

        if scenario.heading_goal is not None and now >= scenario.heading_at:
            err = abs(wrap_angle(_pelvis_yaw(sim) - scenario.heading_goal))
            if yaw_err_time is None and err < 0.1:
                yaw_err_time = now

        if scenario.stand_by is not None and stood is None and now >= scenario.stand_by:
            gait = sim.controller.gait
            mask = sim.controller.telemetry.contact_mask
            stood = (not gait.walking) and bool(np.all(mask))

        if now > scenario.duration - 3.0:
            speed_samples.append(_forward_speed(sim))

        if sim.fell and fall_time is None:
            fall_time = now

    yaw_final = None
    if scenario.heading_goal is not None:
        yaw_final = abs(wrap_angle(_pelvis_yaw(sim) - scenario.heading_goal))
    speed = float(np.mean(speed_samples)) if speed_samples else None
    result = BenchResult(
        name=scenario.name,
        fell=bool(sim.fell),
        elapsed=float(sim.data.time),
        fall_time=fall_time,
        yaw_err_final=yaw_final,
        yaw_err_time=yaw_err_time,
        speed_achieved=speed,
        peak_roll=peak_roll,
        peak_pitch=peak_pitch,
        knee_min=float(knee_min) if np.isfinite(knee_min) else float("nan"),
        stood=stood,
        passed=False,
        reasons=[],
    )
    result.passed, result.reasons = evaluate(scenario, result)
    return result


def evaluate(scenario: Scenario, result: BenchResult) -> tuple[bool, list[str]]:
    """Pass: no fall, heading within the rate budget, speed, and a timely stop."""
    reasons = []
    if result.fell:
        reasons.append(f"fell at {result.fall_time}")
    if scenario.heading_goal is not None:
        if scenario.heading_delta is None:
            delta = abs(wrap_angle(scenario.heading_goal - scenario.heading_from))
        else:
            delta = abs(scenario.heading_delta)
        # The per-step cap can bind below turn_rate. Size the deadline to
        # whichever is slower, using a 0.35 s step as the crawl.
        rate = min(TURN_RATE, TURN_CAP_STAND / 0.35)
        budget = delta / rate + 2.0
        # The clock starts at the command, not at t = 0.
        deadline = scenario.heading_at + budget
        if result.yaw_err_final is None or result.yaw_err_final > 0.1:
            reasons.append(f"heading error {result.yaw_err_final}")
        elif result.yaw_err_time is None or result.yaw_err_time > deadline:
            reasons.append(
                f"heading settled at {result.yaw_err_time}, deadline {deadline:.2f}"
            )
    if scenario.speed_goal is not None and scenario.speed_goal > 1e-3:
        if result.speed_achieved is None or result.speed_achieved < 0.6 * scenario.speed_goal:
            reasons.append(
                f"speed {result.speed_achieved} < 0.6*{scenario.speed_goal}"
            )
    if scenario.stand_by is not None and result.stood is not True:
        reasons.append("did not stand by the deadline")
    return (not reasons), reasons


def select(names: list[str] | None) -> list[Scenario]:
    rows = scenarios()
    if not names:
        return rows
    wanted = {name.strip() for name in names}
    return [row for row in rows if row.name in wanted or row.group in wanted]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", default=None,
                        help="Scenario name or group (turn, gait, push, regression)")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)
    chosen = select(args.only)
    results = []
    for scenario in chosen:
        result = run_scenario(scenario)
        results.append(result.to_dict())
        flag = "PASS" if result.passed else "FAIL"
        why = "; ".join(result.reasons) if result.reasons else "ok"
        print(f"{flag} {scenario.name:22s} {why}")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0 if all(row["passed"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
