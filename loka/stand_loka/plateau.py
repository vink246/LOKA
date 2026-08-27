"""Plateau / intervention-cap gate for standing failure episodes.

When early-anomaly residuals stop improving across rounds (or the episode
hits ``max_interventions``), accept the residual as the new operating point:
lean Task_Targets toward the bias, update Model_Mutations mass belief from
telemetry, re-baseline MissionNominal, and close the episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from loka.control.stand import StandController

# Approximate lateral lever from pelvis to shoulder roll link [m].
_SHOULDER_LEVER_M = 0.20
_MAX_INFERRED_MASS_KG = 12.0
_MIN_INFERRED_MASS_KG = 1.0
_MAX_LEAN_ABS = 0.04


@dataclass(frozen=True)
class EpisodeMetrics:
    """Compact plant-health snapshot for round-to-round comparison."""

    anomaly_score: float
    com_xy_m: float
    com_lat_m: float
    com_error: tuple[float, float, float]
    force_mismatch_n: float
    tilt_rad: float
    mpc_cost: float

    def severity(self) -> float:
        """Scalar used for improvement checks (lower is better)."""
        return (
            float(self.anomaly_score)
            + 0.015 * float(self.force_mismatch_n)
            + 8.0 * float(self.com_xy_m)
            + 4.0 * float(self.tilt_rad)
        )


def snapshot_metrics(frames: Sequence[dict]) -> EpisodeMetrics | None:
    """Mean metrics over an anomaly collection window."""
    if not frames:
        return None
    scores = []
    com_xy = []
    com_lat = []
    com_errs = []
    force_errs = []
    tilts = []
    costs = []
    for frame in frames:
        com = np.asarray(frame.get("com_error", np.zeros(3)), dtype=float)
        com_errs.append(com)
        com_xy.append(float(np.linalg.norm(com[:2])))
        com_lat.append(float(abs(com[1])))
        scores.append(float(frame.get("anomaly_score", 0.0)))
        des = np.asarray(frame.get("desired_forces", np.zeros((1, 3))), dtype=float)
        got = np.asarray(frame.get("contact_forces", np.zeros((1, 3))), dtype=float)
        if des.shape == got.shape:
            force_errs.append(float(np.linalg.norm(des - got)))
        else:
            force_errs.append(0.0)
        rpy = np.asarray(frame.get("rpy", np.zeros(3)), dtype=float)
        tilts.append(float(np.linalg.norm(rpy[:2])))
        costs.append(float(frame.get("mpc_cost", 0.0)))
    mean_com = np.mean(np.stack(com_errs, axis=0), axis=0)
    return EpisodeMetrics(
        anomaly_score=float(np.mean(scores)),
        com_xy_m=float(np.mean(com_xy)),
        com_lat_m=float(np.mean(com_lat)),
        com_error=(float(mean_com[0]), float(mean_com[1]), float(mean_com[2])),
        force_mismatch_n=float(np.mean(force_errs)),
        tilt_rad=float(np.mean(tilts)),
        mpc_cost=float(np.mean(costs)),
    )


def is_improved(
    current: EpisodeMetrics,
    previous: EpisodeMetrics,
    *,
    rel_improve: float = 0.08,
    abs_score_eps: float = 0.03,
) -> bool:
    """True if ``current`` is meaningfully healthier than ``previous``."""
    prev_s = previous.severity()
    cur_s = current.severity()
    if prev_s <= 1e-9:
        return cur_s < prev_s
    rel = (prev_s - cur_s) / prev_s
    score_drop = previous.anomaly_score - current.anomaly_score
    return rel >= rel_improve or score_drop >= abs_score_eps


def should_stop_episode(
    *,
    n_interventions: int,
    stall_count: int,
    max_interventions: int,
    plateau_min_interventions: int,
    plateau_stall_rounds: int,
) -> str | None:
    """Return stop reason, or None to keep dispatching."""
    if max_interventions > 0 and n_interventions >= max_interventions:
        return "max_interventions"
    if (
        n_interventions >= plateau_min_interventions
        and stall_count >= plateau_stall_rounds
    ):
        return "plateau"
    return None


def _infer_mass_belief(
    metrics: EpisodeMetrics,
    controller: StandController,
) -> dict[str, Any] | None:
    """Estimate an absolute body-mass belief update from CoM lateral residual."""
    import mujoco

    com_y = float(metrics.com_error[1])
    com_lat = abs(com_y)
    if com_lat < 0.008 and metrics.force_mismatch_n < 45.0:
        return None

    total_mass = float(controller.robot.total_mass)
    # Δm ≈ |Δy| * M / lever — shoulder lever for lateral bias; torso for vertical load.
    if com_lat >= 0.008:
        delta_kg = com_lat * total_mass / _SHOULDER_LEVER_M
        body = (
            "left_shoulder_roll_link" if com_y > 0.0 else "right_shoulder_roll_link"
        )
    else:
        # Persistent force mismatch without clear lateral bias → torso mass.
        delta_kg = max(0.0, (metrics.force_mismatch_n - 40.0) / 9.81)
        body = "torso_link"

    delta_kg = float(np.clip(delta_kg, _MIN_INFERRED_MASS_KG, _MAX_INFERRED_MASS_KG))
    model = controller.robot.model
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if body_id < 0:
        body = "torso_link"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if body_id < 0:
        return None
    absolute = float(model.body_mass[body_id]) + delta_kg
    return {
        "object_type": "body",
        "name": body,
        "attribute": "mass",
        "value": absolute,
    }


def build_accept_residual_scratchpad(
    metrics: EpisodeMetrics,
    controller: StandController,
    *,
    reason: str,
) -> dict[str, Any]:
    """Scratchpad that absorbs the residual into mission + mass belief."""
    cmd = controller.task_snapshot()
    # Lean toward the heavy / biased side (prompt convention: +lean_y = left).
    lean_x = float(np.clip(metrics.com_error[0], -_MAX_LEAN_ABS, _MAX_LEAN_ABS))
    lean_y = float(np.clip(metrics.com_error[1], -_MAX_LEAN_ABS, _MAX_LEAN_ABS))
    # Keep current height command — do not thrash crouch further.
    height = float(cmd["height"])
    yaw = float(cmd["yaw"])

    pelvis_z = float(cmd.get("height", height))  # fallback; caller may override
    # Error_Tracking uses pelvis world-Z; approximate from commanded CoM height.
    # Stand tall: height~0.70 → pelvis~0.78. Use a soft floor under current pose.
    pelvis_target = max(0.55, height + 0.06)

    mutation = _infer_mass_belief(metrics, controller)
    mutations = [mutation] if mutation is not None else []

    hypothesis = (
        "Intervention plateau — accept residual as new operating point"
        if reason == "plateau"
        else "Intervention cap reached — accept residual as new operating point"
    )
    analysis = (
        f"Stop reason={reason}. Metrics: score={metrics.anomaly_score:.2f}, "
        f"com_xy={1000 * metrics.com_xy_m:.1f} mm, "
        f"force_mismatch={metrics.force_mismatch_n:.1f} N. "
        "Further Controller_Targets churn is unlikely to help; absorb the bias "
        "into Task_Targets lean, update mass Model_Mutations from the residual, "
        "and re-baseline MissionNominal."
    )

    scratchpad: dict[str, Any] = {
        "Semantic_State": {
            "Hypothesis": hypothesis,
            "Analysis": analysis,
        },
        "Task_Targets": {
            "height": height,
            "lean_x": lean_x,
            "lean_y": lean_y,
            "yaw": yaw,
        },
        "Error_Tracking": {
            "trigger_threshold": 0.25,
            "terms": [
                {
                    "name": "pelvis_height",
                    "signal": "qpos",
                    "index": 2,
                    "mode": "below_target",
                    "target": pelvis_target,
                    "tolerance": 0.10,
                    "weight": 2.0,
                },
                {
                    "name": "lateral_drift",
                    "signal": "qpos",
                    "index": 1,
                    "mode": "abs_deviation",
                    "target": 0.0,
                    "tolerance": 0.18,
                    "weight": 1.0,
                },
                {
                    "name": "forward_drift",
                    "signal": "qpos",
                    "index": 0,
                    "mode": "abs_deviation",
                    "target": 0.0,
                    "tolerance": 0.20,
                    "weight": 1.0,
                },
                {
                    "name": "tip_rate",
                    "signal": "qvel",
                    "index": 4,
                    "mode": "abs_above",
                    "target": 0.0,
                    "tolerance": 0.8,
                    "weight": 1.5,
                },
                {
                    "name": "roll_rate",
                    "signal": "qvel",
                    "index": 3,
                    "mode": "abs_above",
                    "target": 0.0,
                    "tolerance": 0.8,
                    "weight": 1.5,
                },
            ],
        },
    }
    if mutations:
        scratchpad["Model_Mutations"] = mutations
    # pelvis_z placeholder kept for callers that want to retarget from live qpos.
    scratchpad["_suggested_pelvis_target"] = pelvis_z
    return scratchpad


def exceeds_residual_floor(
    metrics: EpisodeMetrics,
    floor: EpisodeMetrics,
    *,
    score_margin: float = 0.12,
    force_margin_n: float = 15.0,
    com_margin_m: float = 0.012,
) -> bool:
    """True if plant health is meaningfully worse than an accepted residual."""
    return (
        metrics.anomaly_score > floor.anomaly_score + score_margin
        or metrics.force_mismatch_n > floor.force_mismatch_n + force_margin_n
        or metrics.com_xy_m > floor.com_xy_m + com_margin_m
    )
