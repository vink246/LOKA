"""LLM worker and system-prompt assembly for the stand plant (no MJPC import)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from loka.agent.reply_parse import parse_orchestrator_reply
from loka.agent.scene_describe import DESCRIBE_MAX_TOKENS, DESCRIBE_PREFILL, format_scene_section
from loka.agent.robot_context import format_primary_objective_block

load_dotenv()
_client: OpenAI | None = None
#: A local backend with ``chat(messages) -> str`` replaces the OpenAI call.
_chat_backend = None

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "system_prompt.txt"
#: The same rules and reply schema in about a fifth of the tokens, for small local models.
_COMPACT_PROMPT_PATH = _PROMPT_PATH.with_name("system_prompt_compact.txt")
PROMPT_STYLES = ("full", "compact")


def set_chat_backend(backend) -> None:
    """Route orchestrator turns to ``backend.chat`` (None restores OpenAI)."""
    global _chat_backend
    _chat_backend = backend


def _openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


_VISION_LINE = (
    "- an attached egocentric RGB image from the head camera, captured when this "
    "message was sent, and a CAMERA FRAME section with the camera's world pose at "
    "that instant (see EGOCENTRIC CAMERA below)"
)
_COMPACT_VISION_LINE = (
    "Each message also carries one head-camera image and its CAMERA FRAME pose (see CAMERA below)."
)


def load_stand_system_prompt(
    *,
    robot_context: str | None = None,
    objective: str | None = None,
    vision: str | None = None,
    style: str = "full",
) -> str:
    """``vision`` is the EGOCENTRIC CAMERA block; None leaves the prompt text-only.

    ``style`` ``compact`` uses ``system_prompt_compact.txt``.
    """
    if style not in PROMPT_STYLES:
        raise ValueError(f"prompt style must be one of {PROMPT_STYLES}, got {style!r}")
    path = _COMPACT_PROMPT_PATH if style == "compact" else _PROMPT_PATH
    system_prompt = path.read_text(encoding="utf-8").rstrip()
    plant = (
        "The robot is a Unitree G1 controlled by a centroidal convex MPC and a whole-body QP."
    )
    system_prompt = system_prompt.replace("{{PLANT}}", plant)
    if vision:
        line = _COMPACT_VISION_LINE if style == "compact" else _VISION_LINE
        system_prompt = system_prompt.replace("{{VISION}}", line)
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
            return {"image_bytes": len(png), "image_error": f"frame not saved: {exc}", "_png": png}
    return {"image_bytes": len(png), "frame_path": frame_path, "_png": png}


def _complete(messages, model=None, *, prefill: str | None = None, max_tokens: int | None = None) -> str:
    """One chat completion from the local backend, or OpenAI when none is set.

    ``prefill`` starts the local model's reply (OpenAI has no prefill and ignores
    it); None keeps the backend's default.
    """
    if _chat_backend is not None:
        kwargs = {}
        if prefill is not None:
            kwargs["prefill"] = prefill
        if max_tokens is not None:
            kwargs["max_new_tokens"] = max_tokens
        return _chat_backend.chat(messages, **kwargs)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    chosen = model or os.environ.get("LOKA_STAND_MODEL", "gpt-6-luna")
    response = _openai_client().chat.completions.create(
        model=chosen,
        messages=messages,
        # temperature=0.1,
    )
    return response.choices[0].message.content or ""


def _describe_scene(png: bytes, describe: dict, model=None, detail: str = "high") -> str:
    """The description call: the frame alone, then the description prompt."""
    import base64

    url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url, "detail": detail}},
                {"type": "text", "text": describe["prompt"]},
            ],
        }
    ]
    text = _complete(messages, model, prefill=DESCRIBE_PREFILL, max_tokens=DESCRIBE_MAX_TOKENS)
    return text.strip()


def stand_llm_worker(api_messages, result_queue, model=None, vision=None) -> None:
    """Background thread: never call from the control loop.

    ``model`` overrides the ``LOKA_STAND_MODEL`` environment variable. When
    both are unset the worker uses ``gpt-6-luna``. A backend set with
    :func:`set_chat_backend` takes the turn instead and ignores ``model``.
    ``vision`` holds ``render`` (snapshot -> PNG), the dispatch ``snapshot``,
    ``detail``, and an optional ``frame_path``; the frame is rendered here, off
    the control loop. With ``vision["describe"]`` (a ``prompt``), the model
    first describes the frame and that description is appended to the turn
    as written.
    """
    image = {}
    try:
        if vision is not None:
            image = _attach_camera_frame(api_messages, vision)
            png = image.pop("_png", None)
            describe = vision.get("describe")
            if describe and png is not None:
                from loka.agent.session import append_text

                description = _describe_scene(png, describe, model, vision.get("detail", "high"))
                print(f"\n[LLM] scene description:\n{description}")
                append_text(api_messages, format_scene_section(description))
                image["scene_description"] = description
        yaml_text = _complete(api_messages, model)
    except Exception as exc:
        print(f"\n[LLM Worker Error] {exc}")
        result_queue.put(None)
        return
    try:
        scratchpad, cleaned, notes = parse_orchestrator_reply(yaml_text)
    except ValueError as exc:
        print(f"\n[LLM Worker Error] unparseable reply: {exc}\n--- reply ---\n{yaml_text[:1500]}\n---")
        # Keep the raw text so the turn record and session log show what came back.
        result_queue.put({"scratchpad": None, "raw_yaml": yaml_text, "parse_error": str(exc), **image})
        return
    result = {"scratchpad": scratchpad, "raw_yaml": cleaned, "parse_notes": notes, **image}
    if notes:
        print(f"\n[LLM] reply repaired: {'; '.join(notes)}\n--- reply before repair ---\n{yaml_text[:1500]}\n---")
        result["raw_reply"] = yaml_text
    result_queue.put(result)
