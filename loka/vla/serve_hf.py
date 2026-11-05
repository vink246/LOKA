"""A minimal OpenAI-compatible server for Llama-3.2-11B-Vision (PACE GPU only).

It wraps :class:`loka.vla.mllama_chat.MllamaChatBackend` so the 11B model runs
through the same eval workflow as the vLLM-served models: the GPU job starts
this server, and the orchestrator reaches it with the ``local`` provider.

    python -m loka.vla.serve_hf --model-path ~/scratch/models/llama-3.2-11b-vision-instruct \\
        --served-model-name llama-3.2-11b-vision --port 8000

Endpoints: ``GET /health``, ``GET /v1/models``, ``POST /v1/chat/completions``
(``messages``, ``max_tokens``; a final assistant message with
``continue_final_message: true`` is the reply prefill). One request is decoded
at a time, greedily: a ``temperature`` above 0 is refused with a 400. Only the
standard library is used for HTTP.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def split_prefill(body: dict) -> tuple[list[dict], str]:
    """``(messages, prefill)``: the last assistant message is the prefill when continued."""
    messages = list(body.get("messages") or [])
    if not messages:
        raise ValueError("messages is empty")
    if body.get("continue_final_message") and messages[-1].get("role") == "assistant":
        content = messages[-1].get("content")
        if not isinstance(content, str):
            content = "".join(p.get("text", "") for p in content or () if p.get("type") == "text")
        return messages[:-1], content
    return messages, ""


def completion_payload(model: str, text: str, usage: dict | None) -> dict:
    usage = dict(usage or {})
    prompt = int(usage.get("prompt_tokens", 0))
    completion = int(usage.get("completion_tokens", 0))
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


def make_handler(backend, served_name: str, default_max_tokens: int):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") == "/health":
                self._send(200, {"status": "ok"})
            elif self.path.rstrip("/") == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": served_name, "object": "model"}]})
            else:
                self._send(404, {"error": {"message": f"no route {self.path}"}})

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send(404, {"error": {"message": f"no route {self.path}"}})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                temperature = body.get("temperature")
                if temperature is not None and float(temperature) > 0.0:
                    raise ValueError(
                        f"temperature={temperature} is not supported: this server decodes "
                        "greedily (temperature 0 only)"
                    )
                messages, prefill = split_prefill(body)
                max_tokens = int(
                    body.get("max_tokens") or body.get("max_completion_tokens") or default_max_tokens
                )
                with lock:
                    text = backend.chat(messages, prefill=prefill, max_new_tokens=max_tokens)
                    usage = dict(getattr(backend, "last_usage", {}) or {})
            except ValueError as exc:
                self._send(400, {"error": {"message": str(exc)}})
                return
            except Exception as exc:  # report, keep serving
                self._send(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
                return
            self._send(200, completion_payload(served_name, text, usage))

        def log_message(self, fmt, *args):
            print(f"[serve_hf] {self.address_string()} {fmt % args}", flush=True)

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", default="llama-3.2-11b-vision")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args(argv)

    from loka.vla.mllama_chat import MllamaChatBackend

    print(f"[serve_hf] loading {args.model_path} on {args.device}", flush=True)
    backend = MllamaChatBackend(args.model_path, device=args.device, max_new_tokens=args.max_new_tokens)
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(backend, args.served_model_name, args.max_new_tokens)
    )
    print(f"[serve_hf] {args.served_model_name} on http://{args.host}:{args.port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
