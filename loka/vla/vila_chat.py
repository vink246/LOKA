"""VILA1.5-8B as LOKA's orchestrator, loaded locally on a GPU.

``Efficient-Large-Model/Llama-3-VILA1.5-8B`` is the general VILA model that
NaVILA was fine-tuned from: the same SigLIP encoder, projector and Llama-3-8B,
without the navigation SFT. Running LOKA on it holds the model family fixed
against the NaVILA baseline. The backend takes the OpenAI-style messages the
LOKA runtime already builds and returns the reply text.

Weights load only from a local directory; fetch them on PACE with
``scripts/fetch_navila_checkpoint.py --repo-id Efficient-Large-Model/Llama-3-VILA1.5-8B``.
"""

from __future__ import annotations

import base64
import io
import threading
from pathlib import Path

from loka.vla.navila import load_vila_checkpoint, require_navila_checkpoint

VILA_HUB_ID = "Efficient-Large-Model/Llama-3-VILA1.5-8B"
VILA_FETCH_HINT = (
    "Download on PACE only: python scripts/fetch_navila_checkpoint.py "
    f"--repo-id {VILA_HUB_ID} --out ~/scratch/models/llama-3-vila1.5-8b"
)
LLAMA3_SYSTEM_HEADER = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
#: Prompts longer than this many tokens are past VILA1.5's training length.
PROMPT_TOKEN_WARN = 7000


def _decode_data_url(url: str):
    from PIL import Image

    if not url.startswith("data:"):
        raise ValueError("only inline data: image URLs are supported")
    payload = url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def openai_messages_to_vila(messages: list[dict]) -> tuple[str, list[tuple[str, str]], list]:
    """Split OpenAI chat messages into ``(system, turns, images)``.

    ``turns`` is ``[(role, text), ...]`` with role ``user`` or ``assistant``.
    Each inline image becomes an ``<image>`` token at the start of its message,
    in order, and the decoded PIL image is appended to ``images``.
    """
    system_parts: list[str] = []
    turns: list[tuple[str, str]] = []
    images: list = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        texts: list[str] = []
        n_images = 0
        if isinstance(content, str):
            texts.append(content)
        else:
            for part in content or ():
                kind = part.get("type")
                if kind == "text":
                    texts.append(part.get("text", ""))
                elif kind == "image_url":
                    images.append(_decode_data_url(part["image_url"]["url"]))
                    n_images += 1
        text = "\n".join(t for t in texts if t)
        if role == "system":
            system_parts.append(text)
            continue
        if role not in ("user", "assistant"):
            raise ValueError(f"unsupported message role {role!r}")
        if n_images:
            text = "<image>\n" * n_images + text
        turns.append((role, text))
    return "\n\n".join(system_parts), turns, images


class VilaChatBackend:
    """Greedy chat completion from a local VILA checkpoint. Thread-safe."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda:0",
        max_new_tokens: int = 1024,
    ) -> None:
        import torch

        self.model_path = require_navila_checkpoint(model_path, name="VILA", hint=VILA_FETCH_HINT)
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.label = f"vila:{self.model_path.name}"
        self._torch = torch
        self._lock = threading.Lock()
        self.tokenizer, self.model, self.image_processor, self.context_len = load_vila_checkpoint(
            self.model_path, device
        )

    def build_prompt(self, system: str, turns: list[tuple[str, str]]):
        """Llama-3 chat prompt and the conversation it came from."""
        from llava.conversation import conv_templates

        conv = conv_templates["llama_3"].copy()
        if system:
            conv.system = LLAMA3_SYSTEM_HEADER + system
        for role, text in turns:
            conv.append_message(conv.roles[0] if role == "user" else conv.roles[1], text)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt(), conv

    def chat(
        self, messages: list[dict], *, prefill: str = "", max_new_tokens: int | None = None
    ) -> str:
        """Greedy reply. VILA's llava generate takes no prefill, so ``prefill`` is ignored."""
        del prefill
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.conversation import SeparatorStyle
        from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token

        system, turns, images = openai_messages_to_vila(messages)
        prompt, conv = self.build_prompt(system, turns)
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.device)
        )
        n_tokens = int(input_ids.shape[-1])
        if n_tokens > PROMPT_TOKEN_WARN:
            print(
                f"[vila] prompt is {n_tokens} tokens (+{len(images)} image(s)); "
                f"past ~{PROMPT_TOKEN_WARN} VILA1.5 may degrade"
            )
        image_arg = None
        if images:
            tensor = process_images(images, self.image_processor, self.model.config)
            image_arg = [tensor.to(self.device, dtype=self._torch.float16)]
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)
        with self._lock, self._torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_arg,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=int(max_new_tokens or self.max_new_tokens),
                use_cache=True,
                stopping_criteria=[stopping],
            )
        text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return text.strip()
