"""Anchored edits of the system prompt for ablation runs.

Each ablation has one patch file per prompt style in
``loka/agent/prompts/ablations/<ablation>.<style>.yaml``: a list of
``{find, replace}`` edits, optionally gated with ``when`` / ``unless`` on
active flags (the applied ablation names, plus ``hide_residual``). Every
active ``find`` must occur exactly once in the text it is applied to, so an
edit to the base prompt that moves an anchor fails loudly instead of
silently leaving the ablated run with the wrong instructions.

Patches apply in ``KNOWN_ABLATIONS`` order. ``{{NAME}}`` placeholders in a
replacement are filled from ``fills``.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from loka.agent.ablations import KNOWN_ABLATIONS, AblationSet

PATCH_DIR = Path(__file__).resolve().parent / "prompts" / "ablations"
_PLACEHOLDER = re.compile(r"\{\{([A-Z_]+)\}\}")


def active_flags(ablations: AblationSet) -> set[str]:
    flags = set(ablations.names)
    if ablations.hide_residual:
        flags.add("hide_residual")
    return flags


def load_patch(ablation: str, style: str) -> list[dict]:
    if ablation not in KNOWN_ABLATIONS:
        raise ValueError(f"no prompt patch for unknown ablation {ablation!r}")
    path = PATCH_DIR / f"{ablation}.{style}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise ValueError(f"{path} must be a list of {{find, replace}} edits")
    edits = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or "find" not in item or "replace" not in item:
            raise ValueError(f"{path} edit {index} needs find and replace")
        unknown = set(item) - {"find", "replace", "when", "unless"}
        if unknown:
            raise ValueError(f"{path} edit {index} has unknown keys {sorted(unknown)}")
        edits.append(
            {
                "find": str(item["find"]),
                "replace": str(item["replace"]),
                "when": [str(f) for f in item.get("when") or ()],
                "unless": [str(f) for f in item.get("unless") or ()],
                "source": f"{path.name}#{index}",
            }
        )
    return edits


def apply_prompt_ablations(
    text: str,
    style: str,
    ablations: AblationSet | None,
    fills: dict[str, str] | None = None,
) -> str:
    """``text`` with every active ablation's edits applied."""
    if ablations is None or not ablations.names:
        return text
    flags = active_flags(ablations)
    for ablation in ablations.names:
        for edit in load_patch(ablation, style):
            if any(flag not in flags for flag in edit["when"]):
                continue
            if any(flag in flags for flag in edit["unless"]):
                continue
            count = text.count(edit["find"])
            if count != 1:
                raise ValueError(
                    f"prompt patch {edit['source']} anchor occurs {count} times (needs exactly 1): "
                    f"{edit['find'][:80]!r}"
                )
            text = text.replace(edit["find"], edit["replace"])
    fills = dict(fills or {})

    def _fill(match: re.Match) -> str:
        key = match.group(1)
        if key in ("VISION", "PLANT"):
            return match.group(0)
        if key not in fills:
            raise ValueError(f"prompt patch placeholder {{{{{key}}}}} has no value")
        return str(fills[key])

    return _PLACEHOLDER.sub(_fill, text)
