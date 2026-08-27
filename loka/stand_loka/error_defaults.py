"""Standing-mission Error_Tracking defaults and G1 index helpers."""

from __future__ import annotations

from loka.error_spec import ErrorSpec, ErrorTerm

# Floating-base layout in models/g1/scene.xml:
#   qpos: [x, y, z, qw, qx, qy, qz, joints...]
#   qvel: [vx, vy, vz, wx, wy, wz, joint rates...]
# Pitch is not a qpos Euler angle (quaternion). We track pelvis height and
# linear/angular rates that are directly readable; lean is judged via CoM
# error terms the compressor adds as semantic tags, and via torso angular
# velocity as a cheap proxy for tipping.


def default_stand_error_spec(*, nominal_height: float = 0.778) -> ErrorSpec:
    """Mission criteria for an undisturbed stand.

    Trigger is intentionally loose so normal QP noise does not spam the LLM;
    a real lean-off or height collapse still crosses it.
    """
    return ErrorSpec(
        trigger_threshold=0.20,
        terms=[
            ErrorTerm(
                name="pelvis_height",
                signal="qpos",
                index=2,
                mode="below_target",
                target=nominal_height,
                tolerance=0.08,
                weight=2.0,
            ),
            ErrorTerm(
                name="lateral_drift",
                signal="qpos",
                index=1,
                mode="abs_deviation",
                target=0.0,
                tolerance=0.12,
                weight=1.0,
            ),
            ErrorTerm(
                name="forward_drift",
                signal="qpos",
                index=0,
                mode="abs_deviation",
                target=0.0,
                tolerance=0.15,
                weight=1.0,
            ),
            ErrorTerm(
                name="tip_rate",
                signal="qvel",
                index=4,  # wy (pitch rate)
                mode="abs_above",
                target=0.0,
                tolerance=0.8,
                weight=1.5,
            ),
            ErrorTerm(
                name="roll_rate",
                signal="qvel",
                index=3,  # wx
                mode="abs_above",
                target=0.0,
                tolerance=0.8,
                weight=1.5,
            ),
        ],
    )


def default_walk_error_spec(*, nominal_height: float = 0.778) -> ErrorSpec:
    """Mission criteria while ``gait.mode=walk``.

    Absolute forward world-x drift is *not* a failure — walking is supposed
    to advance. Track tip-over proxies and lateral wander only.
    """
    return ErrorSpec(
        trigger_threshold=0.35,
        terms=[
            ErrorTerm(
                name="pelvis_height",
                signal="qpos",
                index=2,
                mode="below_target",
                target=nominal_height,
                tolerance=0.12,
                weight=2.0,
            ),
            ErrorTerm(
                name="lateral_drift",
                signal="qpos",
                index=1,
                mode="abs_deviation",
                target=0.0,
                tolerance=0.25,
                weight=1.5,
            ),
            ErrorTerm(
                name="tip_rate",
                signal="qvel",
                index=4,
                mode="abs_above",
                target=0.0,
                tolerance=1.2,
                weight=1.5,
            ),
            ErrorTerm(
                name="roll_rate",
                signal="qvel",
                index=3,
                mode="abs_above",
                target=0.0,
                tolerance=1.2,
                weight=1.5,
            ),
        ],
    )
