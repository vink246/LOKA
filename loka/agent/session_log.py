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
    ) -> None:
        body = (
            f"reason: {reason}\n\n"
            f"--- USER TURN (compressor + context sent to LLM) ---\n"
            f"{user_content.rstrip()}\n"
        )
        if system_prompt:
            body += (
                f"\n--- SYSTEM PROMPT ({len(system_prompt)} chars; truncated in .log) ---\n"
                f"{system_prompt[:2000]}{'…' if len(system_prompt) > 2000 else ''}\n"
            )
        self.event("DISPATCH", body, sim_time=sim_time)
        self._jsonl(
            {
                "kind": "dispatch",
                "sim_time": sim_time,
                "reason": reason,
                "user_content": user_content,
                "system_prompt": system_prompt,
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
