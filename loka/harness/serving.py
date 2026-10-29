"""Start an OpenAI-compatible server for a local orchestrator model (PACE only).

The GPU eval job and ``scripts/serve_local_model.sh`` both use the shell block
from :func:`render_server_block`: pick a free port, start the server from the
model's server env, wait for ``/health``, and export ``LOKA_LOCAL_BASE_URL``.

    python -m loka.harness.serving gemma-3-27b-it --print-command --port 8000
    python -m loka.harness.serving gemma-3-27b-it --print-block --log /tmp/server.log
    python -m loka.harness.serving gemma-3-27b-it --print-fetch
"""

from __future__ import annotations

import argparse
import shlex
import sys

from loka.agent.model_registry import expand, resolve_model
from loka.agent.providers import LOCAL_BASE_URL_ENV

#: Loading a 27B-120B checkpoint from scratch storage can take many minutes.
DEFAULT_READY_TIMEOUT_S = 1800


def local_entry(key: str) -> dict:
    entry = resolve_model(key)
    if entry["provider"] != "local":
        raise ValueError(f"{key!r} is a {entry['provider']} model, not a local one")
    return entry


def server_command(entry: dict, port: int | str, host: str = "127.0.0.1") -> list[str]:
    """argv that serves ``entry`` on ``host:port``."""
    env = expand(entry["server_env"])
    path = str(expand(entry["path"]))
    key = str(entry["key"])
    if entry["engine"] == "vllm":
        return [
            str(env / "bin" / "vllm"),
            "serve",
            path,
            "--served-model-name",
            key,
            "--host",
            host,
            "--port",
            str(port),
            "--max-model-len",
            str(int(entry.get("max_model_len", 32768))),
            *[str(a) for a in entry.get("vllm_args") or ()],
        ]
    if entry["engine"] == "hf":
        return [
            str(env / "bin" / "python"),
            "-m",
            "loka.vla.serve_hf",
            "--model-path",
            path,
            "--served-model-name",
            key,
            "--host",
            host,
            "--port",
            str(port),
            "--max-new-tokens",
            str(int(entry.get("max_tokens", 2048))),
        ]
    raise ValueError(f"unknown engine {entry['engine']!r}")


def render_server_block(
    entry: dict,
    *,
    log_path: str,
    ready_timeout_s: int = DEFAULT_READY_TIMEOUT_S,
) -> list[str]:
    """Bash lines that start the server, wait until it answers, and export its URL.

    The server is killed when the calling script exits.
    """
    env = expand(entry["server_env"])
    path = expand(entry["path"])
    port_cmd = (
        'python3 -c "import socket; s = socket.socket(); s.bind((\'127.0.0.1\', 0)); '
        'print(s.getsockname()[1])"'
    )
    argv = server_command(entry, "$LOKA_SERVER_PORT")
    command = " ".join(a if a == "$LOKA_SERVER_PORT" else shlex.quote(a) for a in argv)
    # Double quotes: the GPU job's log path holds ${SLURM_ARRAY_*} to expand.
    if any(ch in str(log_path) for ch in '"`\\'):
        raise ValueError(f"server log path {log_path!r} has a character that cannot be double-quoted")
    log_q = f'"{log_path}"'
    return [
        f"# Local orchestrator {entry['key']} ({entry['engine']}) from {env}",
        f"if [ ! -d {shlex.quote(str(path))} ]; then",
        f'  echo "checkpoint {path} is missing; run scripts/fetch_local_models.sh {entry["key"]}" >&2',
        "  exit 1",
        "fi",
        f"if [ ! -d {shlex.quote(str(env))} ]; then",
        f'  echo "server env {env} is missing; run the setup script for engine {entry["engine"]}" >&2',
        "  exit 1",
        "fi",
        'export HF_HOME="${HF_HOME:-$HOME/scratch/huggingface}"',
        'export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/scratch/.cache}"',
        "export HF_HUB_OFFLINE=1",
        "export VLLM_NO_USAGE_STATS=1",
        f"export LOKA_SERVER_PORT=$({port_cmd})",
        f"( {command} ) > {log_q} 2>&1 &",
        "LOKA_SERVER_PID=$!",
        'trap \'kill "$LOKA_SERVER_PID" 2>/dev/null || true; wait "$LOKA_SERVER_PID" 2>/dev/null || true\' EXIT',
        "loka_ready=0",
        f"for _ in $(seq 1 {int(ready_timeout_s) // 5}); do",
        '  if ! kill -0 "$LOKA_SERVER_PID" 2>/dev/null; then',
        f'    echo "model server exited; see "{log_q} >&2',
        f"    tail -n 50 {log_q} >&2 || true",
        "    exit 1",
        "  fi",
        '  if curl -sf "http://127.0.0.1:$LOKA_SERVER_PORT/health" >/dev/null; then loka_ready=1; break; fi',
        "  sleep 5",
        "done",
        'if [ "$loka_ready" != 1 ]; then',
        f'  echo "model server not ready after {int(ready_timeout_s)} s; see "{log_q} >&2',
        "  exit 1",
        "fi",
        f'export {LOCAL_BASE_URL_ENV}="http://127.0.0.1:$LOKA_SERVER_PORT/v1"',
        f'echo "[serve] {entry["key"]} ready at ${LOCAL_BASE_URL_ENV}"',
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local orchestrator server commands.")
    parser.add_argument("key", help="Model key in loka/config/orchestrator_models.yaml")
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--print-command", action="store_true", help="The server argv, shell-quoted.")
    what.add_argument("--print-block", action="store_true", help="The start-and-wait bash block.")
    what.add_argument(
        "--print-fetch",
        action="store_true",
        help="hf_repo, path and comma-separated fetch_exclude globs, tab-separated.",
    )
    what.add_argument("--print-env", action="store_true", help="The server env prefix.")
    parser.add_argument("--port", default="8000")
    parser.add_argument("--log", default="local_model_server.log")
    parser.add_argument("--ready-timeout-s", type=int, default=DEFAULT_READY_TIMEOUT_S)
    args = parser.parse_args(argv)
    try:
        entry = local_entry(args.key)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.print_command:
        print(" ".join(shlex.quote(a) for a in server_command(entry, args.port)))
    elif args.print_block:
        print("\n".join(render_server_block(entry, log_path=args.log, ready_timeout_s=args.ready_timeout_s)))
    elif args.print_fetch:
        excludes = ",".join(str(p) for p in entry.get("fetch_exclude") or ())
        print(f"{entry.get('hf_repo') or ''}\t{expand(entry['path'])}\t{excludes}")
    else:
        print(expand(entry["server_env"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
