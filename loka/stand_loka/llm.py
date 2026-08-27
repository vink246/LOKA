"""LLM worker and system-prompt assembly for the stand plant (no MJPC import)."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

from loka.error_spec import format_error_spec
from loka.robot_context import format_primary_objective_block
from loka.stand_loka.context import format_stand_capabilities_block

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "system_prompt_stand.txt"


def load_stand_system_prompt(
    *,
    robot_context: str | None = None,
    objective: str | None = None,
    capabilities: dict | None = None,
    error_spec=None,
) -> str:
    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8").rstrip()
    blocks = [system_prompt]
    if objective:
        blocks.append(format_primary_objective_block(objective))
    if capabilities:
        blocks.append(format_stand_capabilities_block(capabilities))
    if error_spec is not None:
        blocks.append("## CURRENT ERROR TRACKING\n" + format_error_spec(error_spec))
    if robot_context:
        blocks.append(robot_context)
    return "\n\n".join(blocks)


def stand_llm_worker(api_messages, result_queue) -> None:
    """Background thread: never call from the control loop."""
    try:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        response = client.chat.completions.create(
            model=os.environ.get("LOKA_STAND_MODEL", "gpt-5.4-mini"),
            messages=api_messages,
            temperature=0.1,
        )
        yaml_text = response.choices[0].message.content or ""
        yaml_text = yaml_text.replace("```yaml", "").replace("```", "").strip()
        scratchpad = yaml.safe_load(yaml_text)
        result_queue.put({"scratchpad": scratchpad, "raw_yaml": yaml_text})
    except Exception as exc:
        print(f"\n[LLM Worker Error] {exc}")
        result_queue.put(None)
