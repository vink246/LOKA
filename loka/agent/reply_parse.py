"""Recover the orchestrator's YAML scratchpad from a model reply.

Smaller local models wrap the YAML in markdown (``**Semantic_State**``,
``### Task_Targets``, code fences), add prose before or after it, or leave
values unquoted that YAML cannot read (``Analysis: The robot: stands``).
:func:`parse_orchestrator_reply` undoes those in order and only falls back to
loading one top-level section at a time when the whole reply still fails.
"""

from __future__ import annotations

import re

import yaml

#: Top-level sections the runtime reads (apply.py, session.py, belief_search.py).
TOP_LEVEL_KEYS = (
    "Semantic_State",
    "Task_Targets",
    "Listeners",
    "Plan",
    "Belief_Searches",
    "Belief_Search",
    "Model_Mutations",
    "Planner_Targets",
    "Mission_Directives",
    "Primary_Objective",
)
_CANONICAL = {key.lower(): key for key in TOP_LEVEL_KEYS}

_FENCE = re.compile(r"```[ \t]*([A-Za-z]*)[ \t]*\n(.*?)```", re.DOTALL)
#: ``**Key**:``, ``**Key:**``, ``__Key__:``, ``*Key*:``, ``` `Key`: ```, optional list dash.
_EMPHASIS_KEY = re.compile(
    r"^(?P<indent>\s*)(?P<dash>-\s+)?(?P<mark>\*\*|__|\*|`)(?P<key>[A-Za-z_][\w. -]*?)"
    r"(?P<inner>:?)(?P=mark)(?P<outer>:?)(?P<rest>.*)$"
)
#: ``### Key`` or ``## Key:`` headings.
_HEADING_KEY = re.compile(r"^#{1,6}\s*(?P<key>[A-Za-z_][\w .-]*?)\s*:?\s*$")
_KEY_LINE = re.compile(r"^(?P<indent>\s*)(?P<dash>-\s+)?(?P<key>[A-Za-z_][\w.]*)\s*:(?P<rest>.*)$")
_PLAIN_SCALAR = re.compile(
    r"^(-?\d+(\.\d+)?([eE][-+]?\d+)?|true|false|yes|no|null|~|walk|stand|tread|limp|clear)$",
    re.IGNORECASE,
)


def _canonical_top_key(key: str) -> str | None:
    return _CANONICAL.get(key.strip().lower().replace(" ", "_").replace("-", "_"))


def _pick_fenced(text: str) -> tuple[str, bool]:
    """The fenced block that holds the scratchpad, else the whole text."""
    blocks = [body for _lang, body in _FENCE.findall(text)]
    for body in blocks:
        if any(re.search(rf"(?im)^\W*{key}\W*:?", body) for key in TOP_LEVEL_KEYS):
            return body, True
    if blocks:
        return blocks[0], True
    # A lone opening fence with no close: drop the fence markers.
    if "```" in text:
        return re.sub(r"```[A-Za-z]*", "", text), True
    return text, False


def _strip_markdown(lines: list[str]) -> tuple[list[str], bool]:
    out, changed = [], False
    for line in lines:
        heading = _HEADING_KEY.match(line)
        if heading and _canonical_top_key(heading.group("key")):
            out.append(f"{_canonical_top_key(heading.group('key'))}:")
            changed = True
            continue
        emph = _EMPHASIS_KEY.match(line)
        if emph and (emph.group("inner") or emph.group("outer") or not emph.group("rest").strip()):
            key = emph.group("key").strip()
            top = _canonical_top_key(key) if not emph.group("indent") and not emph.group("dash") else None
            rest = emph.group("rest").lstrip(":").strip()
            out.append(
                f"{emph.group('indent')}{emph.group('dash') or ''}{top or key.replace(' ', '_')}:"
                + (f" {rest}" if rest else "")
            )
            changed = True
            continue
        out.append(line)
    return out, changed


def _normalize_top_keys(lines: list[str]) -> tuple[list[str], bool]:
    """``semantic_state:`` / ``Task targets:`` at column 0 -> the canonical name."""
    out, changed = [], False
    for line in lines:
        m = re.match(r"^(?P<key>[A-Za-z_][\w -]*?)\s*:(?P<rest>.*)$", line)
        if m and not line[:1].isspace():
            top = _canonical_top_key(m.group("key"))
            if top and top != m.group("key"):
                line = f"{top}:{m.group('rest')}"
                changed = True
        out.append(line)
    return out, changed


def _keep_yaml_lines(lines: list[str]) -> tuple[list[str], bool]:
    """Drop prose before the first section and unindented prose between sections."""
    start = next(
        (i for i, line in enumerate(lines) if _canonical_top_key(line.split(":", 1)[0]) and ":" in line),
        None,
    )
    if start is None:
        return lines, False
    kept = []
    for line in lines[start:]:
        if not line.strip() or line[:1].isspace() or line.lstrip().startswith("- "):
            kept.append(line)
        elif _KEY_LINE.match(line):
            kept.append(line)
    dropped = start > 0 or len(kept) != len(lines) - start
    return kept, dropped


def _quote_unsafe_values(lines: list[str]) -> tuple[list[str], bool]:
    """Quote ``key: value`` scalars YAML would misread (colons, #, leading * & ! % @ `)."""
    out, changed = [], False
    for line in lines:
        m = _KEY_LINE.match(line)
        if m:
            value = m.group("rest").strip()
            risky = value and value[0] not in "\"'[{|>" and not _PLAIN_SCALAR.match(value) and (
                ": " in value or " #" in value or value[0] in "*&!%@`"
            )
            if risky:
                escaped = value.replace("\\", "\\\\").replace('"', '\\"')
                line = f'{m.group("indent")}{m.group("dash") or ""}{m.group("key")}: "{escaped}"'
                changed = True
        out.append(line)
    return out, changed


def _load_by_section(text: str) -> tuple[dict, list[str]]:
    """Load each top-level section on its own; keep the ones that parse."""
    sections: list[list[str]] = []
    for line in text.splitlines():
        if not line[:1].isspace() and _canonical_top_key(line.split(":", 1)[0]) and ":" in line:
            sections.append([line])
        elif sections:
            sections[-1].append(line)
    merged: dict = {}
    failed: list[str] = []
    for block in sections:
        name = block[0].split(":", 1)[0]
        try:
            loaded = yaml.safe_load("\n".join(block))
        except yaml.YAMLError:
            failed.append(name)
            continue
        if isinstance(loaded, dict):
            merged.update(loaded)
    return merged, failed


def parse_orchestrator_reply(text: str) -> tuple[dict, str, list[str]]:
    """Return ``(scratchpad, cleaned_yaml, notes)``; raise ``ValueError`` if nothing parses.

    ``notes`` says which repairs ran, for the session log.
    """
    if not text or not text.strip():
        raise ValueError("empty reply")
    notes: list[str] = []
    body, fenced = _pick_fenced(text.strip())
    if fenced:
        notes.append("code fence removed")
    lines = body.replace("\t", "    ").splitlines()
    lines, changed = _strip_markdown(lines)
    if changed:
        notes.append("markdown removed from keys")
    lines, changed = _normalize_top_keys(lines)
    if changed:
        notes.append("section names normalized")
    lines, changed = _keep_yaml_lines(lines)
    if changed:
        notes.append("prose outside the YAML dropped")
    cleaned = "\n".join(lines).strip()

    attempts = [cleaned]
    quoted_lines, changed = _quote_unsafe_values(lines)
    if changed:
        attempts.append("\n".join(quoted_lines).strip())
    last_error: Exception | None = None
    for index, candidate in enumerate(attempts):
        try:
            loaded = yaml.safe_load(candidate)
        except yaml.YAMLError as exc:
            last_error = exc
            continue
        if isinstance(loaded, dict):
            if index:
                notes.append("unsafe values quoted")
            return loaded, candidate, notes
        last_error = ValueError(f"reply is YAML but not a mapping ({type(loaded).__name__})")

    merged, failed = _load_by_section(attempts[-1])
    if merged:
        notes.append(f"loaded section by section; dropped {', '.join(failed) or 'none'}")
        return merged, yaml.safe_dump(merged, sort_keys=False), notes
    raise ValueError(f"no YAML scratchpad in the reply: {last_error}")
