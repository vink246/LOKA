"""Metrics. Each one is a pure function of stored episode data.

Episode-scoped metrics read one rollout. Trial-scoped metrics read every probe
of a search. Both are recomputed from disk by ``run_eval metrics``.
"""

from __future__ import annotations

import numpy as np

from loka.harness.registry import register_metric
from loka.harness.storage import EpisodeData


def _finite(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


@register_metric("success", scope="episode")
def success(episode: EpisodeData) -> float:
    return 1.0 if episode.metadata.get("outcome") == "success" else 0.0


@register_metric("num_successes", scope="episode")
def num_successes(episode: EpisodeData) -> float:
    """1 when the episode arrived, else 0. Plots sum this across trials."""
    return success(episode)


@register_metric("time_to_goal_s", scope="episode")
def time_to_goal_s(episode: EpisodeData):
    """Sim time to arrival. Failures are omitted, not scored as the timeout."""
    if episode.metadata.get("outcome") != "success":
        return None
    return _finite(episode.metadata.get("time_to_goal_s", episode.metadata.get("t_end")))


@register_metric("wall_time_to_goal_s", scope="episode")
def wall_time_to_goal_s(episode: EpisodeData):
    """Wall-clock seconds until arrival. Failures are omitted."""
    if episode.metadata.get("outcome") != "success":
        return None
    return _finite(episode.metadata.get("wall_s"))


@register_metric("final_goal_error_m", scope="episode")
def final_goal_error_m(episode: EpisodeData):
    stored = _finite(episode.metadata.get("final_goal_error_m"))
    if stored is not None:
        return stored
    distances = episode.column("goal_distance")
    distances = distances[np.isfinite(distances)] if distances.size else distances
    if distances.size == 0:
        return None
    return float(distances[-1])


@register_metric("fell", scope="episode")
def fell(episode: EpisodeData) -> float:
    return 1.0 if episode.metadata.get("fell") else 0.0


@register_metric("num_falls", scope="episode")
def num_falls(episode: EpisodeData) -> float:
    """1 when the episode fell, else 0. Plots sum this across trials."""
    return fell(episode)


@register_metric("t_first_fall_s", scope="episode")
def t_first_fall_s(episode: EpisodeData):
    return _finite(episode.metadata.get("t_first_fall"))


@register_metric("path_length_m", scope="episode")
def path_length_m(episode: EpisodeData):
    pelvis = episode.column("pelvis_xyz")
    if pelvis.ndim != 2 or pelvis.shape[0] < 2:
        return None
    steps = np.linalg.norm(np.diff(pelvis[:, :2], axis=0), axis=1)
    return float(np.sum(steps))


@register_metric("path_efficiency", scope="episode")
def path_efficiency(episode: EpisodeData):
    """Straight-line distance from the start to the goal, over the path taken.

    Failures are omitted. A short path that never arrives is not a high score.
    """
    if episode.metadata.get("outcome") != "success":
        return None
    length = path_length_m(episode)
    if length is None:
        return None
    goal = episode.metadata.get("goal_xy")
    pelvis = episode.column("pelvis_xyz")
    if not goal or pelvis.ndim != 2 or pelvis.shape[0] == 0:
        return None
    optimal = float(np.linalg.norm(np.asarray(goal, dtype=float)[:2] - pelvis[0, :2]))
    if optimal < 1e-6:
        return 1.0
    if length < 1e-9:
        return 0.0
    return float(optimal / length)


@register_metric("rms_tilt_rad", scope="episode")
def rms_tilt_rad(episode: EpisodeData):
    rpy = episode.column("rpy")
    if rpy.ndim != 2 or rpy.shape[0] == 0 or rpy.shape[1] < 2:
        return None
    tilt = np.linalg.norm(rpy[:, :2], axis=1)
    return float(np.sqrt(np.mean(np.square(tilt))))


@register_metric("llm_turn_count", scope="episode")
def llm_turn_count(episode: EpisodeData) -> float:
    if episode.turns:
        return float(len(episode.turns))
    return float(episode.metadata.get("llm_turn_count") or 0)


@register_metric("wall_s", scope="episode")
def wall_s(episode: EpisodeData):
    return _finite(episode.metadata.get("wall_s"))


@register_metric("max_push_force_n", scope="trial")
def max_push_force_n(episodes: list[EpisodeData]) -> float:
    """Largest push that was survived. 0 if the mass alone caused a fall."""
    if not episodes:
        return 0.0
    if any(ep.metadata.get("fell_before_push") for ep in episodes):
        survived = [ep for ep in episodes if not ep.metadata.get("fell")]
        if not survived:
            return 0.0
    survived = [
        ep
        for ep in episodes
        if not ep.metadata.get("fell") and not ep.metadata.get("fell_before_push")
    ]
    if not survived:
        return 0.0
    return float(max(float(ep.metadata.get("push_force_n") or 0.0) for ep in survived))
