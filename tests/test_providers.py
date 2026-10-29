"""Orchestrator providers (OpenAI, Anthropic, local server) with fake clients."""

from __future__ import annotations

import queue
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace as NS

import pytest

from loka.agent.llm import stand_llm_worker
from loka.agent.providers import (
    AnthropicProvider,
    LocalProvider,
    OpenAIProvider,
    build_provider,
    openai_messages_to_anthropic,
)
from loka.vla.serve_hf import make_handler, split_prefill

REPLY = 'Semantic_State:\n  Hypothesis: "walk"\n  Analysis: "go"\nTask_Targets:\n  gait.mode: walk\n'
PNG_URL = "data:image/png;base64,QUJD"


class FakeOpenAI:
    def __init__(self, content: str = REPLY):
        self.calls: list[dict] = []
        self.content = content
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **request):
        self.calls.append(request)
        usage = NS(prompt_tokens=120, completion_tokens=8, prompt_tokens_details=NS(cached_tokens=100))
        return NS(choices=[NS(message=NS(content=self.content))], usage=usage)


class FakeAnthropic:
    def __init__(self, text: str):
        self.calls: list[dict] = []
        self.text = text
        self.messages = NS(create=self._create)

    def _create(self, **request):
        self.calls.append(request)
        usage = NS(input_tokens=900, output_tokens=40, cache_read_input_tokens=800, cache_creation_input_tokens=None)
        return NS(content=[NS(type="text", text=self.text)], usage=usage)


def _messages():
    return [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "reply"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "now"},
                {"type": "image_url", "image_url": {"url": PNG_URL, "detail": "high"}},
            ],
        },
    ]


def test_anthropic_conversion():
    system, messages = openai_messages_to_anthropic(_messages())
    assert system == "SYSTEM"
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    image = messages[-1]["content"][1]
    assert image == {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}
    _, merged = openai_messages_to_anthropic(
        [{"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}, {"role": "user", "content": "c"}]
    )
    assert merged[0]["role"] == "user" and len(merged[-1]["content"]) == 2


def test_anthropic_provider_caches_the_system_prompt_and_keeps_the_prefill():
    fake = FakeAnthropic(' "rest of the reply"')
    provider = AnthropicProvider("claude-x", client=fake)
    text, usage = provider.complete(_messages(), prefill="Semantic_State:")
    request = fake.calls[0]
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert request["max_tokens"] == 4096
    assert request["messages"][-1] == {"role": "assistant", "content": [{"type": "text", "text": "Semantic_State:"}]}
    assert text.startswith("Semantic_State:")
    assert usage["input_tokens"] == 900 and usage["cache_read_input_tokens"] == 800


def test_local_provider_continues_the_prefill():
    fake = FakeOpenAI(content="\n  Hypothesis: x")
    provider = LocalProvider("gemma", prefill="Semantic_State:", max_tokens=512, client=fake)
    text, usage = provider.complete([{"role": "user", "content": "hi"}])
    request = fake.calls[0]
    assert request["extra_body"] == {"continue_final_message": True, "add_generation_prompt": False}
    assert request["messages"][-1] == {"role": "assistant", "content": "Semantic_State:"}
    assert request["max_tokens"] == 512
    assert text == "Semantic_State:\n  Hypothesis: x"
    # A server that restates the prefix is not doubled.
    fake.content = "Semantic_State:\n  Hypothesis: y"
    assert provider.complete([{"role": "user", "content": "hi"}])[0] == "Semantic_State:\n  Hypothesis: y"
    # No prefill: a plain chat request.
    provider.complete([{"role": "user", "content": "hi"}], prefill="")
    assert "extra_body" not in fake.calls[-1]


def test_openai_provider_usage_and_no_prefill():
    fake = FakeOpenAI()
    provider = OpenAIProvider("gpt-x", client=fake)
    text, usage = provider.complete(_messages(), prefill="ignored")
    assert text == REPLY and "extra_body" not in fake.calls[0]
    assert usage["input_tokens"] == 120 and usage["cached_input_tokens"] == 100


class BadRequestError(Exception):
    """Stands in for openai.BadRequestError (matched by class name)."""


def test_temperature_is_sent_and_recorded():
    fake = FakeOpenAI()
    provider = OpenAIProvider("gpt-x", temperature=0.0, client=fake)
    provider.complete(_messages())
    assert fake.calls[0]["temperature"] == 0.0
    assert provider.decoding() == {"temperature": 0.0, "requested_temperature": 0.0, "temperature_rejected": False}
    local = FakeOpenAI(content="x")
    LocalProvider("gemma", temperature=0.2, client=local).complete([{"role": "user", "content": "u"}])
    assert local.calls[0]["temperature"] == 0.2
    anthropic = FakeAnthropic("x")
    AnthropicProvider("claude-x", temperature=0.0, client=anthropic).complete(_messages())
    assert anthropic.calls[0]["temperature"] == 0.0
    # Unset sends nothing: the vendor default applies, as before.
    plain = FakeOpenAI()
    OpenAIProvider("gpt-x", client=plain).complete(_messages())
    assert "temperature" not in plain.calls[0]
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        AnthropicProvider("claude-x", temperature=1.5, client=FakeAnthropic("x"))


def test_rejected_temperature_falls_back_and_is_recorded():
    fake = FakeOpenAI()
    create = fake._create

    def picky(**request):
        if "temperature" in request:
            raise BadRequestError("Unsupported value: 'temperature' does not support 0 with this model")
        return create(**request)

    fake.chat.completions.create = picky
    provider = OpenAIProvider("gpt-reasoning", temperature=0.0, client=fake)
    text, _ = provider.complete(_messages())
    assert text == REPLY and "temperature" not in fake.calls[0]
    assert provider.decoding() == {
        "temperature": "default",
        "requested_temperature": 0.0,
        "temperature_rejected": True,
    }
    # Other bad requests are not swallowed.
    fake.chat.completions.create = lambda **_r: (_ for _ in ()).throw(BadRequestError("context too long"))
    provider.temperature = 0.0
    with pytest.raises(BadRequestError):
        provider.complete(_messages())


def test_temperature_in_suite_config():
    from loka.harness.baselines import loka_runtime_config
    from loka.harness.config import parse_suite

    raw = {
        "num_trials": 1,
        "orchestrator": {"model": "gpt-x", "temperature": 0.0},
        "baselines": [
            "loka",
            {"name": "warm", "type": "loka", "options": {"orchestrator": {"model": "claude-x", "temperature": 0.7}}},
        ],
        "tasks": [
            {
                "name": "goto_goal",
                "type": "goto_goal",
                "goal": {"xy": [1.0, 0.0]},
                "timeout_s": 5.0,
                "perturbations": [{"name": "nominal", "kind": "none"}],
                "metrics": ["success"],
            }
        ],
    }
    suite = parse_suite(raw)
    assert suite.to_dict()["orchestrator"]["temperature"] == 0.0
    assert loka_runtime_config(suite.baseline("loka"), suite).llm_options["temperature"] == 0.0
    assert loka_runtime_config(suite.baseline("warm"), suite).llm_options["temperature"] == 0.7
    raw["orchestrator"]["temperature"] = 3.0
    with pytest.raises(ValueError, match="temperature"):
        parse_suite(raw)


def test_build_provider_needs_credentials(monkeypatch):
    monkeypatch.delenv("LOKA_LOCAL_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="LOKA_LOCAL_BASE_URL"):
        build_provider("local", {"model": "gemma-3-27b-it"})
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        build_provider("anthropic", {"model": "claude-x"})
    with pytest.raises(ValueError, match="provider"):
        build_provider("other", {"model": "m"})


def test_worker_reports_usage_from_the_provider():
    fake = FakeOpenAI()
    provider = OpenAIProvider("gpt-x", client=fake)
    results: queue.Queue = queue.Queue()
    stand_llm_worker([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], results, provider=provider)
    result = results.get_nowait()
    assert result["scratchpad"]["Task_Targets"]["gait.mode"] == "walk"
    assert result["usage"]["input_tokens"] == 120 and result["usage"]["calls"] == 1


def test_serve_hf_speaks_the_openai_protocol():
    class Backend:
        last_usage: dict = {}

        def chat(self, messages, prefill="", max_new_tokens=0):
            self.last_usage = {"prompt_tokens": 11 * len(messages), "completion_tokens": 3}
            return prefill + "\n  Hypothesis: z"

    with pytest.raises(ValueError):
        split_prefill({"messages": []})
    messages, prefill = split_prefill(
        {"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "P:"}], "continue_final_message": True}
    )
    assert prefill == "P:" and len(messages) == 1

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Backend(), "llama-3.2-11b-vision", 64))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/v1"
        provider = LocalProvider("llama-3.2-11b-vision", prefill="Semantic_State:", base_url=url)
        text, usage = provider.complete([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
        # Greedy only: temperature 0 is accepted, above 0 is refused and recorded.
        greedy = LocalProvider("llama-3.2-11b-vision", temperature=0.0, base_url=url)
        greedy.complete([{"role": "user", "content": "u"}])
        assert not greedy.decoding()["temperature_rejected"]
        warm = LocalProvider("llama-3.2-11b-vision", temperature=0.7, base_url=url)
        warm.complete([{"role": "user", "content": "u"}])
        assert warm.decoding()["temperature_rejected"]
    finally:
        server.shutdown()
        server.server_close()
    assert text == "Semantic_State:\n  Hypothesis: z"
    assert usage == {"input_tokens": 22, "output_tokens": 3, "calls": 0}
