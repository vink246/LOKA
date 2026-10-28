"""Local VILA orchestrator backend: message conversion, prompt, LLM worker routing."""

from __future__ import annotations

import base64
import queue
import sys
import types
from unittest import mock

import numpy as np
import pytest

from loka.agent import llm
from loka.vision import encode_png
from loka.vla.vila_chat import LLAMA3_SYSTEM_HEADER, VilaChatBackend, openai_messages_to_vila


def _png_url(value: int = 7) -> str:
    rgb = np.full((8, 8, 3), value, dtype=np.uint8)
    return "data:image/png;base64," + base64.b64encode(encode_png(rgb)).decode("ascii")


def test_openai_messages_to_vila_places_image_tokens():
    messages = [
        {"role": "system", "content": "You are LOKA."},
        {"role": "user", "content": "turn one"},
        {"role": "assistant", "content": "reply: one"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "turn two"},
                {"type": "image_url", "image_url": {"url": _png_url(), "detail": "high"}},
            ],
        },
    ]
    system, turns, images = openai_messages_to_vila(messages)
    assert system == "You are LOKA."
    assert turns == [
        ("user", "turn one"),
        ("assistant", "reply: one"),
        ("user", "<image>\nturn two"),
    ]
    assert len(images) == 1 and images[0].size == (8, 8)


def test_openai_messages_to_vila_rejects_remote_images_and_tool_roles():
    with pytest.raises(ValueError, match="data:"):
        openai_messages_to_vila(
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]}]
        )
    with pytest.raises(ValueError, match="role"):
        openai_messages_to_vila([{"role": "tool", "content": "x"}])


def test_worker_routes_to_chat_backend_without_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen = {}

    class Stub:
        def chat(self, messages):
            seen["messages"] = messages
            return "```yaml\nreasoning: ok\n```"

    llm.set_chat_backend(Stub())
    try:
        out: queue.Queue = queue.Queue()
        llm.stand_llm_worker([{"role": "user", "content": "hi"}], out, model="ignored")
        result = out.get_nowait()
    finally:
        llm.set_chat_backend(None)
    assert result["scratchpad"] == {"reasoning": "ok"}
    assert seen["messages"][0]["content"] == "hi"


class _FakeConv:
    """The slice of llava's llama_3 Conversation the backend uses."""

    def __init__(self):
        self.system = LLAMA3_SYSTEM_HEADER + "default"
        self.roles = ("<U>", "<A>")
        self.messages = []
        self.sep = "<|eot_id|>"
        self.sep2 = "<|end_of_text|>"
        self.sep_style = "LLAMA_3"

    def copy(self):
        return _FakeConv()

    def append_message(self, role, message):
        self.messages.append((role, message))

    def get_prompt(self):
        body = "".join(role + (msg + self.sep if msg else "") for role, msg in self.messages)
        return self.system + self.sep + body


def test_backend_builds_llama3_prompt_with_system_and_image(tmp_path):
    ckpt = tmp_path / "llama-3-vila1.5-8b"
    (ckpt / "llm").mkdir(parents=True)
    (ckpt / "config.json").write_text("{}", encoding="utf-8")
    (ckpt / "llm" / "model-00001-of-00004.safetensors").write_bytes(b"x")

    model = mock.MagicMock()
    model.generate.return_value = "ids"
    tokenizer = mock.MagicMock()
    tokenizer.batch_decode.return_value = ["  reasoning: ok  "]
    captured = {}

    def tokenizer_image_token(prompt, tok, index, return_tensors=None):
        captured["prompt"] = prompt
        ids = mock.MagicMock()
        ids.shape = (1, 120)
        ids.unsqueeze.return_value.to.return_value = ids
        return ids

    conversation = types.SimpleNamespace(
        conv_templates={"llama_3": _FakeConv()},
        SeparatorStyle=types.SimpleNamespace(TWO="TWO"),
    )
    mm_utils = types.SimpleNamespace(
        KeywordsStoppingCriteria=mock.MagicMock(),
        process_images=mock.MagicMock(return_value=mock.MagicMock()),
        tokenizer_image_token=tokenizer_image_token,
    )
    modules = {
        "torch": mock.MagicMock(),
        "llava": mock.MagicMock(),
        "llava.constants": types.SimpleNamespace(IMAGE_TOKEN_INDEX=-200),
        "llava.conversation": conversation,
        "llava.mm_utils": mm_utils,
    }
    with mock.patch.dict(sys.modules, modules), mock.patch(
        "loka.vla.vila_chat.load_vila_checkpoint",
        return_value=(tokenizer, model, "proc", 8192),
    ):
        backend = VilaChatBackend(ckpt, device="cuda:0")
        reply = backend.chat(
            [
                {"role": "system", "content": "LOKA SYSTEM"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "operator: walk to the chair"},
                        {"type": "image_url", "image_url": {"url": _png_url()}},
                    ],
                },
            ]
        )
    assert reply == "reasoning: ok"
    assert backend.label == "vila:llama-3-vila1.5-8b"
    prompt = captured["prompt"]
    assert prompt.startswith(LLAMA3_SYSTEM_HEADER + "LOKA SYSTEM")
    assert prompt.count("<image>") == 1
    assert prompt.endswith("<A>")
    mm_utils.process_images.assert_called_once()
    assert model.generate.call_args.kwargs["images"] is not None


def test_parse_args_orchestrator_vila_needs_the_llm():
    from loka.run_loka import parse_args

    args = parse_args(["--orchestrator-vila", "/tmp/vila", "--headless"])
    assert args.orchestrator_vila is not None and not args.no_llm
    for extra in (["--no-llm"], ["--navila"], ["--unitree-rl"]):
        with pytest.raises(SystemExit):
            parse_args(["--orchestrator-vila", "/tmp/vila", *extra])
