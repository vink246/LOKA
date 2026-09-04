"""Success / fall / timeout predicates for a Walker distance episode."""

from __future__ import annotations

# Matches torso mjcf pos z in models/walker/walker_modified.xml.
TORSO_Z0 = 1.3
FALL_HEIGHT_M = 0.45
FALL_PITCH_RAD = 1.2


def world_height(data) -> float:
    return TORSO_Z0 + float(data.qpos[0])


def pos_x(data) -> float:
    return float(data.qpos[1])


def pitch(data) -> float:
    return float(data.qpos[2])


def has_fallen(data) -> bool:
    return world_height(data) < FALL_HEIGHT_M or abs(pitch(data)) > FALL_PITCH_RAD


def reached_goal(data, goal_distance_m: float) -> bool:
    return pos_x(data) >= float(goal_distance_m)


def classify_outcome(
    data,
    *,
    goal_distance_m: float,
    timeout_s: float,
) -> str | None:
    """Return 'success' or 'timeout', or None if the episode should continue.

    A fall does not end the episode. LOKA needs the remaining time to diagnose
    and recover; only the distance goal or the timeout stops the trial.
    """
    if reached_goal(data, goal_distance_m):
        return "success"
    if float(data.time) >= float(timeout_s):
        return "timeout"
    return None
