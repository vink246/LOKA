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


@register_metric("highest_step_m", scope="episode")
def highest_step_m(episode: EpisodeData):
    """Highest ground both soles stood on at once [m], stairs tasks only."""
    return _finite(episode.metadata.get("highest_step_m"))


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


#: Centred moving-average window on the pelvis track before its length is
#: taken [s], applied ``PATH_SMOOTH_PASSES`` times. It is longer than one gait
#: cycle (0.64-0.80 s on both stacks), so the side-to-side sway of each step is
#: not counted as distance walked, and gaits that sway more are not penalised
#: for it. One pass leaves about a fifth of the sway, which at 0.15 m/s (12 cm
#: per cycle) still adds ~4 % to the length; three leave under 1 %.
PATH_SMOOTH_S = 1.0
PATH_SMOOTH_PASSES = 3


def _smoothed_xy(episode: EpisodeData):
    """Pelvis xy with step sway averaged out. The end points are kept exactly."""
    pelvis = episode.column("pelvis_xyz")
    if pelvis.ndim != 2 or pelvis.shape[0] < 2:
        return None
    xy = np.asarray(pelvis[:, :2], dtype=float)
    t = episode.column("t")
    if t.ndim != 1 or t.shape[0] != xy.shape[0]:
        return xy
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0.0:
        return xy
    half = int(round(0.5 * PATH_SMOOTH_S / dt))
    if half < 1:
        return xy
    n = xy.shape[0]
    index = np.arange(n)
    # The window shrinks toward each end, so the first and last samples stay put.
    reach = np.minimum(np.minimum(index, n - 1 - index), half)
    lo, hi = index - reach, index + reach + 1
    for _ in range(PATH_SMOOTH_PASSES):
        cumsum = np.vstack([np.zeros((1, 2)), np.cumsum(xy, axis=0)])
        xy = (cumsum[hi] - cumsum[lo]) / (hi - lo)[:, None]
    return xy


@register_metric("path_length_m", scope="episode")
def path_length_m(episode: EpisodeData):
    """Length of the pelvis path in the ground plane, after ``PATH_SMOOTH_S`` smoothing."""
    xy = _smoothed_xy(episode)
    if xy is None:
        return None
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return float(np.sum(steps))


@register_metric("path_efficiency", scope="episode")
def path_efficiency(episode: EpisodeData):
    """Straight-line distance from the start to the goal region, over the path taken.

    The optimum is measured to the boundary of the goal region (``goal_radius``),
    where arrival is declared, so stopping at its edge cannot score above 1.
    The path is the smoothed one of ``path_length_m``, which keeps its end
    points, so it is never shorter than that optimum.
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
    radius = _finite(episode.metadata.get("goal_radius")) or 0.0
    to_centre = float(np.linalg.norm(np.asarray(goal, dtype=float)[:2] - pelvis[0, :2]))
    optimal = max(to_centre - max(radius, 0.0), 0.0)
    if optimal < 1e-6:
        return 1.0
    if length < 1e-9:
        return 0.0
    return float(optimal / length)


@register_metric("prop_collisions", scope="episode")
def prop_collisions(episode: EpisodeData):
    """Contact onsets between the robot and any prop, over the whole episode."""
    return _finite(episode.metadata.get("prop_collisions"))


#: Half-window either side of the closest approach used for the path tangent [samples].
_ROUTE_TANGENT_SAMPLES = 25


@register_metric("route_followed", scope="episode")
def route_followed(episode: EpisodeData):
    """1 when every ``route`` landmark was passed on its ``keep_on`` side, else 0.

    The side is the sign of the cross product of the smoothed path's tangent
    at its closest approach to the landmark and the direction to it. A path
    that is closest to the landmark at its start or end never passed it.
    Failures and tasks without a route are omitted.
    """
    route = episode.metadata.get("route") or []
    if not route or episode.metadata.get("outcome") != "success":
        return None
    xy = _smoothed_xy(episode)
    if xy is None:
        return None
    n = xy.shape[0]
    for leg in route:
        point = np.asarray(leg["xy"], dtype=float)[:2]
        index = int(np.argmin(np.linalg.norm(xy - point, axis=1)))
        if index == 0 or index == n - 1:
            return 0.0
        k = _ROUTE_TANGENT_SAMPLES
        tangent = xy[min(index + k, n - 1)] - xy[max(index - k, 0)]
        to_point = point - xy[index]
        cross = float(tangent[0] * to_point[1] - tangent[1] * to_point[0])
        side = "left" if cross > 0.0 else "right"
        if side != str(leg.get("keep_on", "")).lower():
            return 0.0
    return 1.0


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


@register_metric("model_invocations", scope="episode")
def model_invocations(episode: EpisodeData):
    """Calls sent to the baseline's model (LOKA's orchestrator, NaVILA's VLA).

    Unlike ``llm_turn_count``, a call that timed out or was still in flight at
    the end counts. 0 for a baseline without a model; None for runs recorded
    before the count was stored.
    """
    return _finite(episode.metadata.get("model_invocations"))


@register_metric("camera_frames", scope="episode")
def camera_frames(episode: EpisodeData):
    """Answered model calls that carried a rendered head-camera frame.

    A vision baseline whose renderer failed to start falls back to text and
    scores 0 here.
    """
    return _finite(episode.metadata.get("camera_frames"))


@register_metric("wall_s", scope="episode")
def wall_s(episode: EpisodeData):
    return _finite(episode.metadata.get("wall_s"))


def _usage_total(episode: EpisodeData, key: str):
    turns = [turn for turn in episode.turns or () if isinstance(turn.get("usage"), dict)]
    if not turns:
        return None
    return float(sum(float(turn["usage"].get(key) or 0) for turn in turns))


@register_metric("llm_input_tokens", scope="episode")
def llm_input_tokens(episode: EpisodeData):
    """Prompt tokens over every orchestrator call, as the provider reported them."""
    return _usage_total(episode, "input_tokens")


@register_metric("llm_output_tokens", scope="episode")
def llm_output_tokens(episode: EpisodeData):
    return _usage_total(episode, "output_tokens")


@register_metric("telemetry_tokens_mean", scope="episode")
def telemetry_tokens_mean(episode: EpisodeData):
    """Mean tokens of the raw-telemetry section per turn (raw ablation only)."""
    values = [
        float(turn["telemetry"]["telemetry_tokens"])
        for turn in episode.turns or ()
        if isinstance(turn.get("telemetry"), dict) and turn["telemetry"].get("telemetry_tokens") is not None
    ]
    return float(np.mean(values)) if values else None


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
