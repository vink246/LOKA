"""Chat providers for the orchestrator: OpenAI, Anthropic, and local servers.

Every provider takes the OpenAI-style messages the runtime builds and returns
``(text, usage)``. ``usage`` holds ``input_tokens`` and ``output_tokens`` as
reported by the API or server.

``local`` talks to an OpenAI-compatible server (``vllm serve`` or
``loka.vla.serve_hf``) at ``LOKA_LOCAL_BASE_URL``; the PACE GPU job starts the
server and exports the URL. Local models run only on PACE.

Smoke test against a running server (one real LOKA turn, parsed):

    python -m loka.agent.providers --smoke gemma-3-27b-it --base-url http://127.0.0.1:8000/v1
"""

from __future__ import annotations

import argparse
import base64
import os
import sys

LOCAL_BASE_URL_ENV = "LOKA_LOCAL_BASE_URL"
DEFAULT_MAX_RETRIES = 4


def _empty_usage() -> dict:
    return {"input_tokens": 0, "output_tokens": 0, "calls": 0}


def add_usage(total: dict | None, part: dict | None) -> None:
    """Accumulate ``part`` into ``total`` in place."""
    if total is None or not part:
        return
    for key, value in part.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total[key] = total.get(key, 0) + value
    total["calls"] = total.get("calls", 0) + 1


def join_prefill(prefill: str, text: str) -> str:
    """The prefill plus its continuation, without doubling a restated prefix."""
    if not prefill:
        return text.strip()
    if text.lstrip().startswith(prefill.split(":", 1)[0]):
        return text.strip()
    return prefill + (text if text[:1].isspace() else " " + text)


def _accepts_keyword(fn, name: str) -> bool:
    import inspect

    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(p.name == name or p.kind is inspect.Parameter.VAR_KEYWORD for p in params)


def _rejects_temperature(exc: Exception) -> bool:
    """A 400 from a model that only accepts its default temperature."""
    return type(exc).__name__ == "BadRequestError" and "temperature" in str(exc).lower()


class ChatProvider:
    name = "base"

    def __init__(
        self,
        model: str,
        *,
        max_tokens: int | None = None,
        prefill: str | None = None,
        temperature: float | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        #: The reply start used when a call does not pass its own prefill.
        self.default_prefill = prefill or ""
        #: None sends no temperature: the vendor or server default applies.
        self.temperature = None if temperature is None else float(temperature)
        self.requested_temperature = self.temperature
        self.temperature_rejected = False

    def decoding(self) -> dict:
        """The decoding settings actually sent, for the turn record."""
        return {
            "temperature": "default" if self.temperature is None else self.temperature,
            "requested_temperature": self.requested_temperature,
            "temperature_rejected": self.temperature_rejected,
        }

    def complete(
        self,
        messages: list[dict],
        *,
        prefill: str | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, dict]:
        raise NotImplementedError

    @property
    def label(self) -> str:
        return f"{self.name}:{self.model}"


class OpenAIProvider(ChatProvider):
    """The OpenAI chat-completions API. A prefill is ignored (the API has none)."""

    name = "openai"

    def __init__(
        self,
        model: str,
        *,
        max_tokens: int | None = None,
        prefill: str | None = None,
        temperature: float | None = None,
        client=None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__(model, max_tokens=max_tokens, prefill=prefill, temperature=temperature)
        if client is None:
            from openai import OpenAI

            kwargs = {"max_retries": DEFAULT_MAX_RETRIES}
            if base_url:
                kwargs["base_url"] = base_url
            if timeout:
                kwargs["timeout"] = timeout
            key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY is not set")
            client = OpenAI(api_key=key, **kwargs)
        self.client = client

    def _request(self, messages, prefill, max_tokens) -> dict:
        request = {"model": self.model, "messages": messages}
        if max_tokens:
            request["max_completion_tokens"] = int(max_tokens)
        return request

    def _finish(self, text: str, prefill: str) -> str:
        return text

    def complete(self, messages, *, prefill=None, max_tokens=None):
        prefill = self.default_prefill if prefill is None else prefill
        request = self._request(list(messages), prefill, max_tokens or self.max_tokens)
        if self.temperature is not None:
            request["temperature"] = self.temperature
        try:
            response = self.client.chat.completions.create(**request)
        except Exception as exc:
            # Some reasoning models accept only their default temperature. Drop it
            # for the rest of the run; decoding() records that it was rejected.
            if self.temperature is None or not _rejects_temperature(exc):
                raise
            print(f"[LLM] {self.model} rejected temperature={self.temperature}; using its default ({exc})")
            self.temperature = None
            self.temperature_rejected = True
            request.pop("temperature", None)
            response = self.client.chat.completions.create(**request)
        text = response.choices[0].message.content or ""
        usage = _empty_usage()
        raw = getattr(response, "usage", None)
        if raw is not None:
            usage["input_tokens"] = int(getattr(raw, "prompt_tokens", 0) or 0)
            usage["output_tokens"] = int(getattr(raw, "completion_tokens", 0) or 0)
            details = getattr(raw, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", None) if details is not None else None
            if cached:
                usage["cached_input_tokens"] = int(cached)
        return self._finish(text, prefill), usage


class LocalProvider(OpenAIProvider):
    """An OpenAI-compatible server on a PACE GPU node (vLLM or serve_hf).

    A prefill is sent as a final assistant message that the server continues
    (``continue_final_message``), and is kept at the start of the returned text.
    """

    name = "local"

    def __init__(
        self,
        model: str,
        *,
        max_tokens: int | None = None,
        prefill: str | None = None,
        temperature: float | None = None,
        client=None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        if client is None:
            base_url = base_url or os.environ.get(LOCAL_BASE_URL_ENV)
            if not base_url:
                raise RuntimeError(
                    f"local orchestrator {model!r} needs {LOCAL_BASE_URL_ENV} (the PACE GPU job "
                    "sets it after starting the model server). Local models run only on PACE."
                )
        super().__init__(
            model,
            max_tokens=max_tokens,
            prefill=prefill,
            temperature=temperature,
            client=client,
            base_url=base_url,
            api_key="EMPTY",
            timeout=timeout,
        )

    def _request(self, messages, prefill, max_tokens) -> dict:
        request = {"model": self.model, "messages": messages}
        if max_tokens:
            request["max_tokens"] = int(max_tokens)
        if prefill:
            request["messages"] = messages + [{"role": "assistant", "content": prefill}]
            request["extra_body"] = {"continue_final_message": True, "add_generation_prompt": False}
        return request

    def _finish(self, text: str, prefill: str) -> str:
        return join_prefill(prefill, text)


def openai_messages_to_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    """``(system, messages)`` for the Anthropic Messages API.

    System messages become the ``system`` text. Data-URL images become base64
    image blocks. Consecutive messages of one role are merged, and the list
    starts with a user message, as the API requires.
    """
    system_parts: list[str] = []
    out: list[dict] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            else:
                system_parts.extend(p.get("text", "") for p in content or () if p.get("type") == "text")
            continue
        if role not in ("user", "assistant"):
            raise ValueError(f"unsupported message role {role!r}")
        blocks: list[dict] = []
        if isinstance(content, str):
            blocks.append({"type": "text", "text": content})
        else:
            for part in content or ():
                kind = part.get("type")
                if kind == "text":
                    blocks.append({"type": "text", "text": part.get("text", "")})
                elif kind == "image_url":
                    url = part["image_url"]["url"]
                    if not url.startswith("data:"):
                        blocks.append({"type": "image", "source": {"type": "url", "url": url}})
                        continue
                    header, data = url.split(",", 1)
                    media = header[len("data:"):].split(";", 1)[0] or "image/png"
                    blocks.append(
                        {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}
                    )
                else:
                    raise ValueError(f"unsupported content part {kind!r}")
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})
    if out and out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": [{"type": "text", "text": "(start)"}]})
    return "\n\n".join(p for p in system_parts if p), out


class AnthropicProvider(ChatProvider):
    """The Anthropic Messages API (``ANTHROPIC_API_KEY``).

    The system prompt is marked for prompt caching: it is the same on every
    call of a run and is most of the input.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        max_tokens: int | None = 4096,
        prefill: str | None = None,
        temperature: float | None = None,
        cache_ttl: str | None = "1h",
        client=None,
        timeout: float | None = None,
    ) -> None:
        if temperature is not None and not 0.0 <= float(temperature) <= 1.0:
            raise ValueError(f"Anthropic temperature must be in [0, 1], got {temperature}")
        if cache_ttl not in (None, "5m", "1h"):
            raise ValueError(f"Anthropic cache_ttl must be 5m or 1h, got {cache_ttl!r}")
        super().__init__(model, max_tokens=max_tokens or 4096, prefill=prefill, temperature=temperature)
        #: The system prompt is identical across every turn, episode and worker of
        #: a run, so a 1 h cache is written about once an hour instead of once per
        #: episode (a write costs more than a 5 m one, a read costs the same).
        self.cache_control = {"type": "ephemeral"}
        if cache_ttl:
            self.cache_control["ttl"] = cache_ttl
        if client is None:
            import anthropic

            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            kwargs = {"max_retries": DEFAULT_MAX_RETRIES}
            if timeout:
                kwargs["timeout"] = timeout
            client = anthropic.Anthropic(**kwargs)
        self.client = client

    def complete(self, messages, *, prefill=None, max_tokens=None):
        prefill = (self.default_prefill if prefill is None else prefill).rstrip()
        system, converted = openai_messages_to_anthropic(list(messages))
        # A second breakpoint after the earlier exchanges caches the history, so
        # only the new user turn is billed as fresh input on the next call. Turns
        # are seconds apart, so the cheaper 5 m write suffices (a 1 h breakpoint
        # may precede a 5 m one, not follow it).
        if len(converted) >= 2 and converted[-2]["role"] == "assistant" and converted[-2]["content"]:
            converted[-2]["content"][-1] = {**converted[-2]["content"][-1], "cache_control": {"type": "ephemeral", "ttl": "5m"}}
        if prefill:
            converted.append({"role": "assistant", "content": [{"type": "text", "text": prefill}]})
        request = {
            "model": self.model,
            "max_tokens": int(max_tokens or self.max_tokens),
            "messages": converted,
        }
        if system:
            request["system"] = [
                {"type": "text", "text": system, "cache_control": dict(self.cache_control)}
            ]
        create = self.client.messages.create
        if self.temperature is not None:
            if _accepts_keyword(create, "temperature"):
                request["temperature"] = self.temperature
            else:
                # Newer SDKs dropped sampling keywords; the body still carries it.
                request["extra_body"] = {"temperature": self.temperature}
        try:
            response = create(**request)
        except Exception as exc:
            if self.temperature is None or not _rejects_temperature(exc):
                raise
            print(f"[LLM] {self.model} rejected temperature={self.temperature}; using its default ({exc})")
            self.temperature = None
            self.temperature_rejected = True
            request.pop("temperature", None)
            request.pop("extra_body", None)
            response = create(**request)
        text = "".join(
            getattr(block, "text", "") for block in response.content or () if getattr(block, "type", "") == "text"
        )
        usage = _empty_usage()
        raw = getattr(response, "usage", None)
        if raw is not None:
            # Anthropic's input_tokens excludes cached tokens; count the whole prompt,
            # as OpenAI's prompt_tokens does.
            uncached = int(getattr(raw, "input_tokens", 0) or 0)
            usage["input_tokens"] = uncached
            usage["uncached_input_tokens"] = uncached
            usage["output_tokens"] = int(getattr(raw, "output_tokens", 0) or 0)
            for key in ("cache_read_input_tokens", "cache_creation_input_tokens"):
                value = getattr(raw, key, None)
                if value:
                    usage[key] = int(value)
                    usage["input_tokens"] += int(value)
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) or "unspecified"
            print(f"[LLM] {self.model} refused the turn (category {category}); no reply this turn")
            usage["refusals"] = 1
            return "", usage
        if prefill:
            text = prefill + (text if text[:1].isspace() else " " + text)
        return text, usage


def build_provider(provider: str, options: dict | None = None, *, timeout: float | None = None) -> ChatProvider:
    """A provider from a resolved model entry (``model_registry.resolve_model``)."""
    options = dict(options or {})
    model = str(options.get("model") or options.get("key") or "")
    if not model:
        raise ValueError("orchestrator model entry has no model")
    max_tokens = options.get("max_tokens")
    prefill = options.get("prefill")
    temperature = options.get("temperature")
    if provider == "openai":
        return OpenAIProvider(model, max_tokens=max_tokens, temperature=temperature, timeout=timeout)
    if provider == "anthropic":
        return AnthropicProvider(
            model,
            max_tokens=max_tokens or 4096,
            prefill=prefill,
            temperature=temperature,
            cache_ttl=options.get("cache_ttl", "1h"),
            timeout=timeout,
        )
    if provider == "local":
        return LocalProvider(
            model,
            max_tokens=max_tokens,
            prefill=prefill,
            temperature=temperature,
            base_url=options.get("base_url"),
            timeout=timeout,
        )
    raise ValueError(f"unknown orchestrator provider {provider!r}")


def _smoke(key: str, base_url: str | None, directive: str, temperature: float | None = None) -> int:
    """Send one real LOKA operator turn to ``key`` and parse the reply."""
    from loka.agent.model_registry import resolve_model
    from loka.agent.reply_parse import parse_orchestrator_reply
    from loka.agent.runtime import LokaConfig, LokaRuntime
    from loka.sim import Simulation

    entry = resolve_model(key)
    if base_url:
        entry["base_url"] = base_url
    if temperature is not None:
        entry["temperature"] = temperature
    provider = build_provider(entry["provider"], entry)
    sim = Simulation()
    runtime = LokaRuntime(sim, LokaConfig(enable_llm=False, enable_session_log=False))
    for _ in range(50):
        frame = runtime.step()
    telemetry = runtime._telemetry_for(frame)
    user = runtime.operator_session.build_request_turn(
        directive, telemetry, runtime.loka_state, float(frame["time"]), runtime._config_text()
    )
    messages = runtime.operator_session.compose_api_messages(runtime.system_prompt, user)
    import time

    started = time.perf_counter()
    text, usage = provider.complete(messages)
    elapsed = time.perf_counter() - started
    print(f"[smoke] {provider.label} replied in {elapsed:.1f}s, usage {usage}, decoding {provider.decoding()}")
    print(text)
    try:
        scratchpad, _, notes = parse_orchestrator_reply(text)
    except ValueError as exc:
        print(f"[smoke] reply did not parse: {exc}", file=sys.stderr)
        return 1
    print(f"[smoke] parsed sections: {sorted(scratchpad)}{' (repaired: ' + '; '.join(notes) + ')' if notes else ''}")
    return 0


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--smoke", required=True, help="Model key or API model name.")
    parser.add_argument("--base-url", default=None, help=f"Local server URL (default ${LOCAL_BASE_URL_ENV}).")
    parser.add_argument(
        "--directive",
        default="Walk to the world coordinate (3, 0) at 0.15 m/s and stop within 0.2 m.",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Default: the model's own.")
    args = parser.parse_args(argv)
    return _smoke(args.smoke, args.base_url, args.directive, args.temperature)


if __name__ == "__main__":
    raise SystemExit(main())
