"""Recover the orchestrator's YAML scratchpad from a model reply.

Smaller local models wrap the YAML in markdown (``**Semantic_State**``,
``### Task_Targets``, code fences), add prose before or after it, or leave
values unquoted that YAML cannot read (``Analysis: The robot: stands``).
:func:`parse_orchestrator_reply` undoes those in order and only falls back to
loading one top-level section at a time when the whole reply still fails.
"""

from __future__ import annotations

import math
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


#: Markdown list markers: ``* x``, ``+ x``, ``• x``, ``1. x``, ``2) x``.
_BULLET = re.compile(r"^(?P<indent>\s*)(?:[*+\u2022\u00b7]|\d+[.)])\s+(?P<rest>\S.*)$")


def _normalize_bullets(lines: list[str]) -> tuple[list[str], bool]:
    """Turn markdown bullets into YAML ``- `` items; ``_sanitize`` merges mapping sections."""
    out, changed = [], False
    for line in lines:
        m = _BULLET.match(line)
        if m:
            out.append(f"{m.group('indent')}- {m.group('rest')}")
            changed = True
        else:
            out.append(line)
    return out, changed


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


#: Fields of a list item (a listener or a search) that may lose their indent.
_LIST_ITEM_FIELDS = frozenset(
    {"kind", "name", "axis", "x", "y", "z", "radius", "tol", "speed", "delay", "message", "when"}
)


def _renest_flat_sections(lines: list[str]) -> tuple[list[str], bool]:
    """Indent a section whose body lost its indentation.

    ``Semantic_State:`` followed by ``Hypothesis: ...`` at column 0 loads as
    ``Semantic_State: null`` plus a stray top-level ``Hypothesis``. A section is
    flattened when its header has no value and its first body line starts at
    column 0; its body is then shifted right until the next section. In a list
    section, a column-0 key that is not a list-item field ends the section.
    """
    out: list[str] = []
    changed = False
    header: str | None = None  # waiting for the first body line of this section
    mode: str | None = None  # "map" or "list" while shifting a flattened body
    for line in lines:
        stripped = line.strip()
        col0 = bool(line) and not line[:1].isspace()
        key = line.split(":", 1)[0] if ":" in line else ""
        if col0 and ":" in line and _canonical_top_key(key):
            header = key if not line.split(":", 1)[1].strip() else None
            mode = None
            out.append(line)
            continue
        if not stripped:
            out.append(line)
            continue
        if header is not None:
            header = None
            if col0:
                mode = "list" if stripped.startswith("- ") else "map"
        if mode is None:
            out.append(line)
            continue
        if mode == "list" and col0 and not stripped.startswith("- "):
            field = key.strip().lower()
            if field in _LIST_ITEM_FIELDS:
                out.append("    " + line)
                changed = True
                continue
            mode = None
            out.append(line)
            continue
        out.append("  " + line)
        changed = True
    return out, changed


#: Where a stray top-level field belongs when its section came back empty.
_STRAY_FIELDS = {
    "Semantic_State": ("hypothesis", "analysis", "hypotheses"),
    "Plan": ("goal", "rationale", "steps", "current_step", "notes"),
}
_TARGET_PREFIXES = ("gait.", "mpc.")
_TARGET_FIELDS = frozenset({"height", "lean_x", "lean_y", "yaw", "com_offset_x", "com_offset_y"})


def _home_section(field: str) -> str | None:
    """The section a field belongs in, for the fields that only live in one."""
    name = str(field)
    if name.startswith(_TARGET_PREFIXES) or name in _TARGET_FIELDS:
        return "Task_Targets"
    for section, fields in _STRAY_FIELDS.items():
        if name.lower() in fields:
            return section
    return None


def _sanitize(pad: dict) -> tuple[dict, list[str]]:
    """Make each known section the type the runtime expects; salvage stray fields.

    A field found at the top level, or inside the wrong one of Semantic_State,
    Task_Targets and Plan, is moved into its own section.
    """
    notes: list[str] = []
    # A mapping section written as bullets loads as a list of one-key mappings.
    for section in ("Semantic_State", "Task_Targets", "Plan"):
        body = pad.get(section)
        if isinstance(body, list) and body and all(isinstance(item, dict) for item in body):
            merged: dict = {}
            for item in body:
                merged.update(item)
            pad[section] = merged
            notes.append(f"{section} list merged into a mapping")
        elif section == "Semantic_State" and isinstance(body, list) and body:
            pad[section] = {"Hypothesis": " ".join(str(item) for item in body)}
            notes.append("Semantic_State list joined into Hypothesis")
    # Entries without a kind are unusable; a section of only those must not
    # replace the armed listeners or the search queue.
    for section in ("Listeners", "Belief_Searches", "Belief_Search"):
        body = pad.get(section)
        if isinstance(body, dict):
            body = [body]
        if not isinstance(body, list) or not body:
            continue
        usable = [item for item in body if isinstance(item, dict) and (item.get("kind") or item.get("when"))]
        if not usable:
            del pad[section]
            notes.append(f"{section} dropped: no entry has a kind")
        elif len(usable) != len(body):
            pad[section] = usable
            notes.append(f"{section}: {len(body) - len(usable)} entry(ies) without a kind dropped")
    moved: dict[str, dict] = {}
    for key in list(pad):
        home = _home_section(key)
        if home is not None:
            moved.setdefault(home, {})[key] = pad.pop(key)
    for section in _STRAY_FIELDS.keys() | {"Task_Targets"}:
        body = pad.get(section)
        if not isinstance(body, dict):
            continue
        for key in list(body):
            home = _home_section(key)
            if home is not None and home != section:
                moved.setdefault(home, {})[key] = body.pop(key)
    for section, fields in moved.items():
        if not isinstance(pad.get(section), dict):
            pad[section] = {}
        for key, value in fields.items():
            name = key if section == "Task_Targets" else key[:1].upper() + key[1:]
            pad[section].setdefault(name, value)
    if moved:
        notes.append("stray fields moved back into their sections")
    for section in list(_STRAY_FIELDS) + ["Task_Targets"]:
        if pad.get(section) == {} and section != "Semantic_State":
            pad[section] = None
    empty = [k for k in TOP_LEVEL_KEYS if k in pad and pad[k] is None]
    for k in empty:
        del pad[k]
    if empty:
        notes.append(f"empty sections dropped: {', '.join(empty)}")
    sem = pad.get("Semantic_State")
    if sem is not None and not isinstance(sem, dict):
        pad["Semantic_State"] = {"Hypothesis": str(sem)}
        notes.append("Semantic_State coerced to a mapping")
    elif sem is None and any(k in pad for k in TOP_LEVEL_KEYS):
        # The runtime refuses a reply without it; keep the targets the model did send.
        pad["Semantic_State"] = {}
        notes.append("missing Semantic_State added")
    for section in ("Task_Targets", "Plan"):
        value = pad.get(section)
        if value is not None and not isinstance(value, dict) and not (section == "Plan" and isinstance(value, str)):
            del pad[section]
            notes.append(f"{section} dropped: not a mapping")
    targets = pad.get("Task_Targets")
    if isinstance(targets, dict):
        bad = [k for k, v in targets.items() if not _valid_target_value(str(k), v)]
        for k in bad:
            del targets[k]
        if bad:
            notes.append(f"non-numeric Task_Targets dropped: {', '.join(map(str, bad))}")
        if not targets:
            del pad["Task_Targets"]
    return pad, notes


_GAIT_MODES = frozenset({"stand", "walk", "tread", "limp"})


def _valid_target_value(key: str, value) -> bool:
    """A number, or a mode name for ``gait.mode``; a copied ``<metres>`` is neither."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if key == "gait.mode" and isinstance(value, str):
        return value.strip().lower() in _GAIT_MODES
    if key == "gait.goal_heading" and value is None:
        return True
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


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
    lines, changed = _normalize_bullets(lines)
    if changed:
        notes.append("markdown bullets turned into YAML items")
    lines, changed = _strip_markdown(lines)
    if changed:
        notes.append("markdown removed from keys")
    lines, changed = _normalize_top_keys(lines)
    if changed:
        notes.append("section names normalized")
    lines, changed = _keep_yaml_lines(lines)
    if changed:
        notes.append("prose outside the YAML dropped")
    lines, changed = _renest_flat_sections(lines)
    if changed:
        notes.append("flattened sections re-indented")
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
            pad, fixes = _sanitize(loaded)
            if fixes:
                notes.extend(fixes)
                candidate = yaml.safe_dump(pad, sort_keys=False)
            return pad, candidate, notes
        last_error = ValueError(f"reply is YAML but not a mapping ({type(loaded).__name__})")

    merged, failed = _load_by_section(attempts[-1])
    if merged:
        notes.append(f"loaded section by section; dropped {', '.join(failed) or 'none'}")
        pad, fixes = _sanitize(merged)
        notes.extend(fixes)
        return pad, yaml.safe_dump(pad, sort_keys=False), notes
    raise ValueError(f"no YAML scratchpad in the reply: {last_error}")
