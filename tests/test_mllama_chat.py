"""Llama 3.2 Vision orchestrator backend: message conversion, CLI wiring, tiny-model run.

The end-to-end test needs ``transformers`` and a tiny random Mllama checkpoint:
set ``LOKA_TINY_MLLAMA`` to a local copy of
``hf-tiny-v2/tiny-random-MllamaForConditionalGeneration``. It is skipped otherwise.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import numpy as np
import pytest

from loka.vision import encode_png
from loka.vla.mllama_chat import openai_messages_to_mllama


def _png_url(value: int = 7, size: int = 32) -> str:
    rgb = np.full((size, size, 3), value, dtype=np.uint8)
    return "data:image/png;base64," + base64.b64encode(encode_png(rgb)).decode("ascii")


def _loka_like_messages() -> list[dict]:
    return [
        {"role": "system", "content": "You are LOKA. Reply in YAML."},
        {"role": "user", "content": "--- PERIODIC CHECK ---\nnothing new"},
        {"role": "assistant", "content": "reasoning: hold"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "--- OPERATOR DIRECTIVE ---\nwalk to the chair"},
                {"type": "image_url", "image_url": {"url": _png_url(), "detail": "high"}},
            ],
        },
    ]


def test_openai_messages_to_mllama_folds_system_and_puts_the_image_first():
    chat, images = openai_messages_to_mllama(_loka_like_messages())
    assert [m["role"] for m in chat] == ["user", "assistant", "user"]
    assert chat[0]["content"] == [
        {"type": "text", "text": "You are LOKA. Reply in YAML.\n\n--- PERIODIC CHECK ---\nnothing new"}
    ]
    assert chat[1]["content"] == [{"type": "text", "text": "reasoning: hold"}]
    assert chat[2]["content"][0] == {"type": "image"}
    assert chat[2]["content"][1]["text"].endswith("walk to the chair")
    assert len(images) == 1 and images[0].size == (32, 32)


def test_openai_messages_to_mllama_system_only_and_image_on_first_turn():
    chat, _ = openai_messages_to_mllama([{"role": "system", "content": "S"}])
    assert chat == [{"role": "user", "content": [{"type": "text", "text": "S"}]}]
    chat, images = openai_messages_to_mllama(
        [
            {"role": "system", "content": "S"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "U"},
                    {"type": "image_url", "image_url": {"url": _png_url()}},
                ],
            },
        ]
    )
    assert chat == [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "S\n\nU"}]}]
    assert len(images) == 1


def test_openai_messages_to_mllama_rejects_bad_roles_and_assistant_images():
    with pytest.raises(ValueError, match="role"):
        openai_messages_to_mllama([{"role": "tool", "content": "x"}])
    with pytest.raises(ValueError, match="user messages"):
        openai_messages_to_mllama(
            [{"role": "assistant", "content": [{"type": "image_url", "image_url": {"url": _png_url()}}]}]
        )


@pytest.mark.parametrize(
    "generated, expected",
    [
        ('\n  Hypothesis: "Walk to the chair."\n  Analysis: "clear"', 'Semantic_State:\n  Hypothesis: "Walk to the chair."\n  Analysis: "clear"'),
        ('{Hypothesis: "Walk."}', 'Semantic_State: {Hypothesis: "Walk."}'),
        ('Semantic_State:\n  Hypothesis: "again"', 'Semantic_State:\n  Hypothesis: "again"'),
    ],
)
def test_reply_is_started_in_yaml_and_joined_once(generated, expected):
    import sys
    import types
    from unittest import mock

    from loka.vla.mllama_chat import REPLY_PREFILL, MllamaChatBackend

    seen = {}

    class Inputs(dict):
        def to(self, device):
            return self

    class Processor:
        def apply_chat_template(self, chat, add_generation_prompt=True):
            return "<prompt>"

        def __call__(self, images=None, text=None, add_special_tokens=False, return_tensors="pt"):
            seen["text"] = text
            ids = mock.MagicMock()
            ids.shape = (1, 5)
            return Inputs(input_ids=ids)

        def decode(self, ids, skip_special_tokens=True):
            return generated

    backend = MllamaChatBackend.__new__(MllamaChatBackend)
    backend.processor = Processor()
    backend.model = mock.MagicMock(device="cpu")
    backend.max_new_tokens = 8
    backend._lock = mock.MagicMock()
    backend._torch = types.SimpleNamespace(inference_mode=mock.MagicMock())
    with mock.patch.dict(sys.modules, {"torch": mock.MagicMock()}):
        reply = backend.chat([{"role": "user", "content": "go"}])
        assert seen["text"] == "<prompt>" + REPLY_PREFILL
        assert reply == expected
        # The description call starts with SCENE: instead, and "" turns the prefill off.
        backend.chat([{"role": "user", "content": "go"}], prefill="SCENE:")
        assert seen["text"] == "<prompt>SCENE:"
        assert backend.chat([{"role": "user", "content": "go"}], prefill="") == generated.strip()
        assert seen["text"] == "<prompt>"


def test_parse_args_local_orchestrators():
    from loka.run_loka import parse_args

    args = parse_args(["--orchestrator-mllama", "/tmp/m", "--headless"])
    assert args.orchestrator_mllama is not None and not args.no_llm
    cpu = parse_args(["--orchestrator-mllama", "/tmp/m", "--orchestrator-device", "cpu"])
    assert cpu.orchestrator_device == "cpu"
    for bad in (
        ["--orchestrator-mllama", "/tmp/m", "--orchestrator-vila", "/tmp/v"],
        ["--orchestrator-mllama", "/tmp/m", "--no-llm"],
        ["--orchestrator-mllama", "/tmp/m", "--navila"],
        ["--orchestrator-vila", "/tmp/v", "--orchestrator-device", "cpu"],
    ):
        with pytest.raises(SystemExit):
            parse_args(bad)


def test_missing_mllama_checkpoint_points_at_the_gated_fetch(tmp_path):
    from loka.run_loka import _install_local_orchestrator, parse_args

    args = parse_args(["--orchestrator-mllama", str(tmp_path / "nope")])
    with pytest.raises(FileNotFoundError, match="Llama-3.2-Vision checkpoint not found.*license"):
        _install_local_orchestrator(args)


@pytest.mark.slow
def test_run_loka_orchestrator_mllama_end_to_end_with_a_stub_model(tmp_path, monkeypatch):
    """The whole run_loka path with the model replaced: flags, checkpoint check,
    backend install, dispatch, worker, YAML parse and Task_Targets apply."""
    import loka.vla.mllama_chat as mllama_chat
    from loka.agent import llm
    from loka.run_loka import main

    ckpt = tmp_path / "llama-3.2-11b-vision-instruct"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}", encoding="utf-8")
    (ckpt / "model-00001-of-00005.safetensors").write_bytes(b"x")
    calls = []

    class StubBackend:
        def __init__(self, model_path, device="cuda:0", max_new_tokens=1024):
            self.label = f"mllama:{Path(model_path).name}"

        def chat(self, messages):
            calls.append(messages)
            # The bold-markdown shape Llama 3.2 Vision produced on PACE.
            return (
                "**Semantic_State**\n"
                "  Hypothesis: walk to the chair\n"
                "  Analysis: The chair: straight ahead, nothing in the way.\n"
                "**Task_Targets:**\n"
                "  gait.mode: walk\n"
                "  gait.speed: 0.25\n"
                "  gait.goal_x: 3.0\n"
                "  gait.goal_y: 0.0\n"
            )

    monkeypatch.setattr(mllama_chat, "MllamaChatBackend", StubBackend)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    try:
        code = main(
            [
                "--orchestrator-mllama", str(ckpt),
                "--orchestrator-device", "cpu",
                "--headless", "-T", "8",
                "--no-props",
                "--operator", "walk to the chair",
                "--operator-at", "0.5",
                "--log-dir", str(tmp_path / "logs"),
                "--quiet-compressor",
            ]
        )
    finally:
        llm.set_chat_backend(None)
    assert code == 0
    assert calls, "the orchestrator never reached the local backend"
    assert "walk to the chair" in str(calls[0])
    # Local orchestrators default to the compact prompt.
    system = calls[0][0]["content"] if calls[0][0]["role"] == "system" else ""
    assert "MJCF kinematics" not in system and "no markdown" in system


TINY = os.environ.get("LOKA_TINY_MLLAMA", "")
#: The tiny checkpoint ships no chat template. This one has the shape of Meta's
#: Llama 3.2 Vision template, including its refusal of a system message next to
#: an image, so the test fails if the backend ever sends one.
TEST_CHAT_TEMPLATE = (
    "{%- set ns = namespace(images=false) -%}"
    "{%- for m in messages -%}{%- if m['content'] is not string -%}"
    "{%- for c in m['content'] -%}{%- if c['type'] == 'image' -%}{%- set ns.images = true -%}"
    "{%- endif -%}{%- endfor -%}{%- endif -%}{%- endfor -%}"
    "{%- if ns.images and messages[0]['role'] == 'system' -%}"
    "{{ raise_exception('Prompting with images is incompatible with system messages.') }}"
    "{%- endif -%}"
    "{{- bos_token -}}"
    "{%- for m in messages -%}"
    "{{- '<|start_header_id|>' + m['role'] + '<|end_header_id|>\n\n' -}}"
    "{%- if m['content'] is string -%}{{- m['content'] | trim -}}{%- else -%}"
    "{%- for c in m['content'] -%}{%- if c['type'] == 'image' -%}{{- '<|image|>' -}}"
    "{%- elif c['type'] == 'text' -%}{{- c['text'] | trim -}}{%- endif -%}{%- endfor -%}"
    "{%- endif -%}{{- '<|eot_id|>' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}{%- endif -%}"
)


@pytest.mark.skipif(not TINY or not Path(TINY).is_dir(), reason="set LOKA_TINY_MLLAMA to a tiny checkpoint")
def test_tiny_mllama_backend_chats_on_cpu_and_drives_the_llm_worker():
    pytest.importorskip("transformers")
    import queue

    from loka.agent import llm
    from loka.vla.mllama_chat import MllamaChatBackend

    backend = MllamaChatBackend(TINY, device="cpu", max_new_tokens=8)
    if not backend.processor.chat_template:
        backend.processor.chat_template = TEST_CHAT_TEMPLATE
    reply = backend.chat(_loka_like_messages())
    assert isinstance(reply, str)

    chat, _ = openai_messages_to_mllama(_loka_like_messages())
    prompt = backend.processor.apply_chat_template(chat, add_generation_prompt=True)
    assert prompt.count("<|image|>") == 1
    assert "walk to the chair" in prompt and "You are LOKA" in prompt
    assert "<|start_header_id|>system" not in prompt

    llm.set_chat_backend(backend)
    try:
        out: queue.Queue = queue.Queue()
        llm.stand_llm_worker(_loka_like_messages(), out)
        # A random model's text rarely parses as YAML; the worker reports None then.
        result = out.get_nowait()
        assert result is None or "raw_yaml" in result
    finally:
        llm.set_chat_backend(None)
