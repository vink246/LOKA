"""Run a single Walker distance episode with an optional plant perturbation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from loka.walker_runtime import WalkerRuntime, distance_objective
from loka.walker_suite.config import EpisodeDefaults, SuiteConfig, TestCase
from loka.walker_suite.faults import perturbation_metadata, resolve_perturbation
from loka.walker_suite.logging import EpisodeLogger
from loka.walker_suite.outcomes import classify_outcome, has_fallen, pos_x, world_height


def run_episode(
    test: TestCase,
    baseline: str,
    suite: SuiteConfig,
    out_dir: Path,
) -> dict[str, Any]:
    params: EpisodeDefaults = test.episode_params(suite.defaults)
    enable_llm = baseline == "loka"
    objective = distance_objective(params.goal_distance_m)

    runtime = WalkerRuntime(
        objective=objective,
        enable_llm=enable_llm,
        speed_goal=params.speed_goal,
    )
    runtime.reset()
    resolved = resolve_perturbation(test.perturbation, runtime.model)

    logger = EpisodeLogger(
        out_dir,
        log_hz=suite.log_hz,
        record=suite.record,
        record_fps=suite.record_fps,
        record_width=suite.record_width,
        record_height=suite.record_height,
        record_camera=suite.record_camera,
        model_path=runtime.xml_path,
        actuator_names=runtime.actuator_names,
    )

    injected = False
    outcome = None
    fell_ever = False
    t_first_fall = None
    wall0 = time.perf_counter()
    last_print = -1.0
    try:
        while True:
            t = float(runtime.data.time)
            if not injected and t + 1e-9 >= params.perturbation_time_s:
                runtime.activate_fault(resolved, t)
                injected = True
                print(
                    f"  [{test.name}/{baseline}] perturbation {resolved.kind} "
                    f"at t={t:.2f}s"
                )

            step = runtime.step()
            logger.maybe_log(runtime, step)

            if has_fallen(runtime.data) and not fell_ever:
                fell_ever = True
                t_first_fall = float(runtime.data.time)

            if t - last_print >= 0.5:
                fallen_tag = " [FALLEN]" if has_fallen(runtime.data) else ""
                print(
                    f"\r  [{test.name}/{baseline}] t={step.time:.2f}s "
                    f"x={pos_x(runtime.data):.2f}m err={step.error:.3f} "
                    f"{step.status}{fallen_tag} ",
                    end="",
                    flush=True,
                )
                last_print = t

            outcome = classify_outcome(
                runtime.data,
                goal_distance_m=params.goal_distance_m,
                timeout_s=params.timeout_s,
            )
            if outcome is not None:
                break
    finally:
        print()
        wall = time.perf_counter() - wall0
        metadata = {
            "test": test.name,
            "baseline": baseline,
            "enable_llm": enable_llm,
            "goal_distance_m": params.goal_distance_m,
            "perturbation_time_s": params.perturbation_time_s,
            "timeout_s": params.timeout_s,
            "speed_goal": params.speed_goal,
            "perturbation": perturbation_metadata(resolved),
            "outcome": outcome or "timeout",
            "t_end": float(runtime.data.time),
            "pos_x_final": pos_x(runtime.data),
            "height_final": world_height(runtime.data),
            "fell": fell_ever,
            "fallen_at_end": has_fallen(runtime.data),
            "t_first_fall": t_first_fall,
            "llm_turn_count": len(runtime.loka_turns),
            "wall_s": wall,
            "fault_injected": injected,
        }
        logger.write(
            metadata=metadata,
            mpc_snapshots=runtime.mpc_snapshots,
            loka_turns=runtime.loka_turns,
            obstacle_overlay=runtime.plant.obstacle_overlay(),
        )
        runtime.close()

    print(
        f"  [{test.name}/{baseline}] {metadata['outcome']} "
        f"t={metadata['t_end']:.2f}s x={metadata['pos_x_final']:.2f}m "
        f"turns={metadata['llm_turn_count']}"
        f"{' fell@' + f'{t_first_fall:.2f}s' if t_first_fall is not None else ''}"
    )
    return metadata
