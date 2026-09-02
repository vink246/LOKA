#!/usr/bin/env python3
"""Phase A evaluation harness skeleton for standing LOKA.

    python -m loka.evaluate_loka
    python -m loka.evaluate_loka --scenario lean_operator --no-llm

Scenarios run headless. LLM scenarios need OPENAI_API_KEY unless --no-llm.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from loka.control.locomotion import LocomotionConfig
from loka.sim import Simulation
from loka.agent.faults import FaultSpec
from loka.agent.runtime import LokaConfig, LokaRuntime


@dataclass
class ScenarioResult:
    name: str
    fell: bool
    duration: float
    com_error_max: float
    tracking_error_max: float
    final_height: float
    final_lean_x: float
    final_lean_y: float
    llm_enabled: bool
    notes: str = ""


def _run(
    name: str,
    *,
    duration: float,
    faults: list[FaultSpec] | None = None,
    operator: str | None = None,
    operator_at: float = 3.0,
    enable_llm: bool = False,
    local_operator_fn=None,
) -> ScenarioResult:
    sim = Simulation(LocomotionConfig())
    runtime = LokaRuntime(
        sim,
        LokaConfig(enable_llm=enable_llm),
        faults=faults or [],
    )
    queued = False
    t0 = time.perf_counter()
    while sim.data.time < duration:
        if not queued and sim.data.time >= operator_at:
            if enable_llm and operator:
                queued = True
                runtime.submit_operator(operator)
            elif local_operator_fn is not None:
                queued = True
                local_operator_fn(runtime)
        runtime.step()
        if sim.fell:
            break
    wall = time.perf_counter() - t0
    hist = runtime.history
    tasks = runtime.controller.task_snapshot()
    return ScenarioResult(
        name=name,
        fell=sim.fell,
        duration=float(sim.data.time),
        com_error_max=float(max((h["com_error"] for h in hist), default=0.0)),
        tracking_error_max=float(max((h["tracking_error"] for h in hist), default=0.0)),
        final_height=tasks["height"],
        final_lean_x=tasks["lean_x"],
        final_lean_y=tasks["lean_y"],
        llm_enabled=enable_llm,
        notes=f"wall={wall:.1f}s steps={len(hist)}",
    )


def scenario_hold_baseline() -> ScenarioResult:
    return _run("hold_baseline_no_loka", duration=6.0, enable_llm=False)


def scenario_lean_operator_ablation() -> ScenarioResult:
    """No-LLM ablation: apply structured Task_Targets directly (not language)."""

    def apply_local(runtime: LokaRuntime) -> None:
        runtime.controller.set_task_targets(
            {"lean_x": 0.025, "lean_y": -0.02, "height": 0.65}
        )

    return _run(
        "lean_task_ablation",
        duration=8.0,
        operator_at=2.5,
        enable_llm=False,
        local_operator_fn=apply_local,
    )


def scenario_mass_fault_no_loka() -> ScenarioResult:
    return _run(
        "mass_fault_no_loka",
        duration=10.0,
        faults=[FaultSpec("mass", 3.0, {"delta_kg": 5.0, "body": "torso_link"})],
        enable_llm=False,
    )


def scenario_friction_fault_no_loka() -> ScenarioResult:
    return _run(
        "friction_fault_no_loka",
        duration=10.0,
        faults=[FaultSpec("friction", 3.0, {"mu": 0.25})],
        enable_llm=False,
    )


def scenario_mass_fault_scripted_recovery() -> ScenarioResult:
    """Ablation stand-in for multi-turn LLM recovery: two scripted scratchpads.

    Turn 1 softens CoM gains / friction belief; turn 2 crouches slightly.
    Exercises the apply path without an API key.
    """
    from loka.agent.apply import apply_stand_scratchpad

    sim = Simulation(LocomotionConfig())
    runtime = LokaRuntime(
        sim,
        LokaConfig(enable_llm=False),
        faults=[FaultSpec("mass", 2.5, {"delta_kg": 6.0, "body": "torso_link"})],
    )
    turn1_at, turn2_at = 3.5, 5.5
    applied = {1: False, 2: False}
    while sim.data.time < 12.0:
        t = sim.data.time
        if t >= turn1_at and not applied[1]:
            applied[1] = True
            apply_stand_scratchpad(
                runtime.controller,
                {
                    "Semantic_State": {
                        "Hypothesis": "added torso mass",
                        "Analysis": "scripted turn 1",
                    },
                    "Controller_Targets": {
                        "wbc.kp_base_position": 35.0,
                        "mpc.friction_mu": 0.45,
                        "wbc.friction_mu": 0.45,
                    },
                },
                runtime.loka_state,
                plant_model=sim.model,
                sim_time=t,
            )
        if t >= turn2_at and not applied[2]:
            applied[2] = True
            apply_stand_scratchpad(
                runtime.controller,
                {
                    "Semantic_State": {
                        "Hypothesis": "still elevated CoM error",
                        "Analysis": "scripted turn 2 crouch",
                    },
                    "Task_Targets": {"height": 0.64, "lean_x": 0.0, "lean_y": 0.0},
                },
                runtime.loka_state,
                plant_model=sim.model,
                sim_time=t,
            )
        runtime.step()
        if sim.fell:
            break
    tasks = runtime.controller.task_snapshot()
    hist = runtime.history
    return ScenarioResult(
        name="mass_fault_scripted_recovery",
        fell=sim.fell,
        duration=float(sim.data.time),
        com_error_max=float(max((h["com_error"] for h in hist), default=0.0)),
        tracking_error_max=float(max((h["tracking_error"] for h in hist), default=0.0)),
        final_height=tasks["height"],
        final_lean_x=tasks["lean_x"],
        final_lean_y=tasks["lean_y"],
        llm_enabled=False,
        notes=f"turns={sum(applied.values())} steps={len(hist)}",
    )


SCENARIOS = {
    "hold": scenario_hold_baseline,
    "lean_operator": scenario_lean_operator_ablation,
    "mass": scenario_mass_fault_no_loka,
    "friction": scenario_friction_fault_no_loka,
    "mass_recovery": scenario_mass_fault_scripted_recovery,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=["all", *SCENARIOS.keys()],
        default="all",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    results = []
    for name in names:
        print(f"\n=== scenario: {name} ===")
        result = SCENARIOS[name]()
        results.append(result)
        print(
            f"  fell={result.fell}  t={result.duration:.2f}s  "
            f"com_err_max={result.com_error_max*1e3:.1f}mm  "
            f"height={result.final_height:.3f}  lean=({result.final_lean_x:.3f},"
            f"{result.final_lean_y:.3f})  {result.notes}"
        )

    if args.json_out:
        args.json_out.write_text(
            json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.json_out}")

    return 0 if not any(r.fell for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
