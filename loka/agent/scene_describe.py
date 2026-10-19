"""Describe-first orchestration: the model says what the camera shows, then replies.

Small vision-language models act better when they first describe the image and
then decide (perception, then action). Before each orchestrator turn that
carries a camera frame, the model gets the image alone with
:func:`build_describe_prompt` and names each object with its direction and a
rough distance. :func:`format_scene_section` appends that description to the
turn verbatim. Nothing is computed from it: turning a direction into a goal is
left to the model, as it is for every other orchestrator.
"""

from __future__ import annotations

import math

#: Prefill for the description call, so the reply starts in the required shape.
DESCRIBE_PREFILL = "SCENE:"
DESCRIBE_MAX_TOKENS = 350


def _floor_rows(spec, cam_height: float, pitch: float) -> list[tuple[int, float]]:
    """(image row, floor distance ahead) for a few rows below the horizon."""
    horizon = spec.cy - spec.fy * math.tan(pitch)
    rows = []
    for v in (int(math.ceil(horizon)) + 20, (spec.height * 5) // 8, (spec.height * 13) // 16, spec.height - 1):
        yn = (v - spec.cy) / spec.fy
        denom = math.sin(pitch) + yn * math.cos(pitch)
        if denom > 1e-6 and 0 <= v < spec.height:
            rows.append((v, cam_height * (math.cos(pitch) - yn * math.sin(pitch)) / denom))
    return rows


def build_describe_prompt(directive: str, spec, pose) -> str:
    """Text that follows the image in the description call. Camera facts only."""
    pitch = float(pose.pitch_down)
    rows = _floor_rows(spec, float(pose.xyz[2]), pitch)
    horizon = spec.cy - spec.fy * math.tan(pitch)
    half_fov = spec.fovx_deg / 2.0
    row_text = "; ".join(f"row {v} of {spec.height}: {d:.1f} m ahead" for v, d in rows)
    return (
        "You are the eyes of a walking humanoid robot. This image is from its head camera, "
        f"{pose.xyz[2]:.2f} m above the floor, looking forward and {math.degrees(pitch):.0f} degrees down. "
        f"The image's left is the robot's left. The horizon is near row {horizon:.0f}; lower in the "
        f"image is closer. Where an object touches the floor tells you how far ahead it is: {row_text}. "
        f"The left and right image edges are {half_fov:.0f} degrees off straight ahead.\n\n"
        f'The robot\'s task: "{directive}"\n\n'
        "Describe the image for the robot. Write only these lines:\n"
        "SCENE: <one sentence about the room and the floor>\n"
        "OBJECT: <name> | <left, ahead, or right>, about <degrees> degrees | about <metres> m away "
        "| blocks path: <yes or no>\n"
        "TARGET: <the object the task refers to and its direction, or none>\n"
        "Write one OBJECT line for every object you see, closest first. blocks path means it is "
        "on or near the straight line ahead of the robot. If you are unsure of a distance, give "
        "your best guess; the direction matters most."
    )


def format_scene_section(description: str) -> str:
    """The section appended to the orchestrator turn after the description call."""
    return "\n".join(
        [
            "## WHAT THE CAMERA SHOWS",
            "Your own description of this frame, written just before this reply. Directions are "
            "relative to where the robot faces now (its yaw in STATE).",
            (description or "").strip() or "(no description)",
        ]
    )
