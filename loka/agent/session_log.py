"""Session log for LOKA turns (compressor + LLM).

Fixed paths under ``logs/loka/`` (gitignored). Each new session truncates and
rewrites ``session.log`` / ``session.jsonl``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_LOG_DIR = Path("logs") / "loka"
SESSION_LOG_NAME = "session.log"
SESSION_JSONL_NAME = "session.jsonl"


def _log_message(message: dict[str, Any]) -> dict[str, Any]:
    """Drop the system prompt, which also carries the robot model reference."""
    if message.get("role") == "system":
        size = len(str(message.get("content") or ""))
        return {"role": "system", "content": f"(omitted, {size} chars)"}
    return message


def default_log_dir(log_dir: Path | None = None) -> Path:
    root = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def default_log_path(log_dir: Path | None = None) -> Path:
    """Human-readable session log path (rewritten each session)."""
    return default_log_dir(log_dir) / SESSION_LOG_NAME


class StandSessionLog:
    """Human-readable + JSONL sidecar for every orchestrator exchange.

    Opening a new instance truncates both files so only the current session
    remains on disk.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_log_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.name == SESSION_LOG_NAME:
            self.jsonl_path = self.path.with_name(SESSION_JSONL_NAME)
        else:
            self.jsonl_path = self.path.with_suffix(".jsonl")
        self._write_header()

    def _write_header(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.path.write_text(
            f"# LOKA stand session log (rewritten each run)\n"
            f"# started_utc: {now}\n"
            f"# path: {self.path.resolve()}\n\n",
            encoding="utf-8",
        )
        self.jsonl_path.write_text("", encoding="utf-8")

    def _append(self, text: str) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(text)
            if not text.endswith("\n"):
                fh.write("\n")

    def _jsonl(self, record: dict[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def event(self, title: str, body: str = "", *, sim_time: float | None = None) -> None:
        t = f" t={sim_time:.3f}s" if sim_time is not None else ""
        banner = f"\n{'=' * 72}\n[{title}]{t}\n{'=' * 72}\n"
        block = banner + (body.rstrip() + "\n" if body else "")
        self._append(block)
        self._jsonl(
            {
                "kind": "event",
                "title": title,
                "sim_time": sim_time,
                "body": body,
                "wall_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    def dispatch(
        self,
        *,
        sim_time: float,
        reason: str,
        user_content: str,
        system_prompt: str | None = None,
        messages: list | None = None,
    ) -> None:
        """Log the user turn. The system prompt and robot model reference are not written."""
        sent = list(messages) if messages else []
        if not sent:
            if system_prompt:
                sent.append({"role": "system", "content": system_prompt})
            sent.append({"role": "user", "content": user_content})
        logged = [_log_message(message) for message in sent]
        parts = [f"reason: {reason}", ""]
        for index, message in enumerate(logged):
            role = message.get("role", "?")
            content = str(message.get("content") or "")
            parts.append(f"--- MESSAGE {index} role={role} ({len(content)} chars) ---")
            parts.append(content.rstrip())
            parts.append("")
        self.event("DISPATCH", "\n".join(parts), sim_time=sim_time)
        self._jsonl(
            {
                "kind": "dispatch",
                "sim_time": sim_time,
                "reason": reason,
                "messages": logged,
                "wall_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    def response(
        self,
        *,
        sim_time: float,
        raw_yaml: str,
        scratchpad: dict[str, Any] | None,
    ) -> None:
        body = f"--- RAW YAML ---\n{raw_yaml.rstrip()}\n"
        self.event("LLM_RESPONSE", body, sim_time=sim_time)
        self._jsonl(
            {
                "kind": "llm_response",
                "sim_time": sim_time,
                "raw_yaml": raw_yaml,
                "scratchpad": scratchpad,
                "wall_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    def apply_summary(self, *, sim_time: float, summary: dict[str, Any]) -> None:
        self.event("APPLY", json.dumps(summary, indent=2, default=str), sim_time=sim_time)

    def note(self, message: str, *, sim_time: float | None = None) -> None:
        self.event("NOTE", message, sim_time=sim_time)
