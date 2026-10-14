"""LLM worker and system-prompt assembly for the stand plant (no MJPC import)."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

from loka.agent.robot_context import format_primary_objective_block

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "system_prompt.txt"


_VISION_LINE = (
    "- an attached egocentric RGB image from the head camera, captured when this "
    "message was sent, and a CAMERA FRAME section with the camera's world pose at "
    "that instant (see EGOCENTRIC CAMERA below)"
)


def load_stand_system_prompt(
    *,
    robot_context: str | None = None,
    objective: str | None = None,
    vision: str | None = None,
) -> str:
    """``vision`` is the EGOCENTRIC CAMERA block; None leaves the prompt text-only."""
    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8").rstrip()
    plant = (
        "The robot is a Unitree G1 controlled by a centroidal convex MPC and a whole-body QP."
    )
    system_prompt = system_prompt.replace("{{PLANT}}", plant)
    if vision:
        system_prompt = system_prompt.replace("{{VISION}}", _VISION_LINE)
    else:
        system_prompt = system_prompt.replace("{{VISION}}\n", "").replace("{{VISION}}", "")
    blocks = [system_prompt]
    if objective:
        blocks.append(format_primary_objective_block(objective))
    if vision:
        blocks.append(vision)
    if robot_context:
        blocks.append(robot_context)
    return "\n\n".join(blocks)


def _attach_camera_frame(api_messages, vision: dict) -> dict:
    """Render the dispatch snapshot and attach it. Returns what to report back."""
    from loka.agent.session import append_text, attach_image

    try:
        png = vision["render"](vision["snapshot"])
    except Exception as exc:
        append_text(
            api_messages,
            f"(The camera frame for this turn is unavailable: {exc}. Reason from telemetry only.)",
        )
        return {"image_error": str(exc)}
    attach_image(api_messages, png, detail=vision.get("detail", "high"))
    frame_path = vision.get("frame_path")
    if frame_path is not None:
        try:
            Path(frame_path).parent.mkdir(parents=True, exist_ok=True)
            Path(frame_path).write_bytes(png)
        except OSError as exc:
            return {"image_bytes": len(png), "image_error": f"frame not saved: {exc}"}
    return {"image_bytes": len(png), "frame_path": frame_path}


def stand_llm_worker(api_messages, result_queue, model=None, vision=None) -> None:
    """Background thread: never call from the control loop.

    ``model`` overrides the ``LOKA_STAND_MODEL`` environment variable. When
    both are unset the worker uses ``gpt-5.4-mini``. ``vision`` holds
    ``render`` (snapshot -> PNG), the dispatch ``snapshot``, ``detail``, and an
    optional ``frame_path``; the frame is rendered here, off the control loop.
    """
    image = {}
    try:
        if vision is not None:
            image = _attach_camera_frame(api_messages, vision)
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        chosen = model or os.environ.get("LOKA_STAND_MODEL", "gpt-6-luna")
        response = client.chat.completions.create(
            model=chosen,
            messages=api_messages,
            # temperature=0.1,
        )
        yaml_text = response.choices[0].message.content or ""
        yaml_text = yaml_text.replace("```yaml", "").replace("```", "").strip()
        scratchpad = yaml.safe_load(yaml_text)
        result_queue.put({"scratchpad": scratchpad, "raw_yaml": yaml_text, **image})
    except Exception as exc:
        print(f"\n[LLM Worker Error] {exc}")
        result_queue.put(None)
