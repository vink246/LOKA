"""Llama-3.2-11B-Vision-Instruct as LOKA's orchestrator, loaded locally.

Its language model is Llama-3.1-8B, the same family and size as NaVILA's
Llama-3-8B, and it has a 128K context, so LOKA's full prompt (about 22K tokens a
turn) fits unchanged. The backend takes the OpenAI-style messages the LOKA
runtime builds and returns the reply text.

Weights load only from a local directory. The repo is gated: accept the license
on Hugging Face, log in on PACE, then fetch with
``scripts/fetch_navila_checkpoint.py --repo-id meta-llama/Llama-3.2-11B-Vision-Instruct``.
Needs the ``loka-mllama`` env (``scripts/setup_mllama_env.sh``).
"""

from __future__ import annotations

import threading
from pathlib import Path

from loka.vla.navila import require_navila_checkpoint
from loka.vla.vila_chat import _decode_data_url

MLLAMA_HUB_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
MLLAMA_FETCH_HINT = (
    "Accept the license at https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct, "
    "then on PACE only: huggingface-cli login && python scripts/fetch_navila_checkpoint.py "
    f"--repo-id {MLLAMA_HUB_ID} --out ~/scratch/models/llama-3.2-11b-vision-instruct"
)
#: bf16 weights are about 21 GB; the KV cache for a 22K-token turn adds about 3.5 GB.
MLLAMA_MIN_FREE_GB = 28.0
#: Llama 3.2 Vision's context is 128K tokens; warn well before it.
PROMPT_TOKEN_WARN = 100_000
#: The reply is started for the model so it continues in LOKA's YAML layout
#: instead of opening with markdown or prose. It is prepended to the returned
#: text. Only the section name: a longer start matched the prompt's reply
#: format line for line, and the model continued it with the format's text.
REPLY_PREFILL = "Semantic_State:"


def openai_messages_to_mllama(messages: list[dict]) -> tuple[list[dict], list]:
    """Convert OpenAI chat messages to the Mllama chat-template format.

    Returns ``(chat, images)``. Each message becomes its images, as
    ``{"type": "image"}`` entries, then one text entry; the decoded PIL images
    are returned in order for the processor. Meta's template rejects a system
    message once any image is present, so the system text leads the first user
    message instead, on every turn, with or without an image.
    """
    system_texts: list[str] = []
    chat: list[dict] = []
    images: list = []
    for message in messages:
        role = message.get("role")
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"unsupported message role {role!r}")
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
        text = "\n\n".join(t for t in texts if t)
        if role == "system":
            if n_images:
                raise ValueError("Llama 3.2 Vision only takes images in user messages")
            system_texts.append(text)
            continue
        if role != "user" and n_images:
            raise ValueError("Llama 3.2 Vision only takes images in user messages")
        chat.append({"role": role, "text": text, "images": n_images})
    if system_texts:
        system = "\n\n".join(system_texts)
        first_user = next((m for m in chat if m["role"] == "user"), None)
        if first_user is None:
            chat.insert(0, {"role": "user", "text": system, "images": 0})
        else:
            first_user["text"] = system + ("\n\n" + first_user["text"] if first_user["text"] else "")
    return [
        {
            "role": m["role"],
            "content": [{"type": "image"}] * m["images"] + [{"type": "text", "text": m["text"]}],
        }
        for m in chat
    ], images


class MllamaChatBackend:
    """Greedy chat completion from a local Llama 3.2 Vision checkpoint. Thread-safe."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda:0",
        max_new_tokens: int = 1024,
    ) -> None:
        import torch
        from transformers import AutoProcessor, MllamaForConditionalGeneration

        self.model_path = require_navila_checkpoint(
            model_path, name="Llama-3.2-Vision", hint=MLLAMA_FETCH_HINT
        )
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.label = f"mllama:{self.model_path.name}"
        self.last_usage: dict = {}
        self._torch = torch
        self._lock = threading.Lock()
        # local_files_only: a missing file fails here instead of pulling from the Hub.
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map={"": device},
            local_files_only=True,
        )
        self.model.eval()

    def chat(
        self,
        messages: list[dict],
        *,
        prefill: str = REPLY_PREFILL,
        max_new_tokens: int | None = None,
    ) -> str:
        """Greedy reply. ``prefill`` starts the reply ("" for none) and is kept in the text."""
        chat, images = openai_messages_to_mllama(messages)
        prompt = self.processor.apply_chat_template(chat, add_generation_prompt=True) + prefill
        inputs = self.processor(
            images=[images] if images else None,
            text=prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.model.device)
        n_tokens = int(inputs["input_ids"].shape[-1])
        if n_tokens > PROMPT_TOKEN_WARN:
            print(f"[mllama] prompt is {n_tokens} tokens; the context is 128K")
        with self._lock, self._torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens or self.max_new_tokens),
                do_sample=False,
                # The checkpoint's generation_config samples; unset its knobs for greedy.
                temperature=None,
                top_p=None,
            )
        reply_ids = output[0, n_tokens:]
        self.last_usage = {"prompt_tokens": n_tokens, "completion_tokens": int(reply_ids.shape[-1])}
        text = self.processor.decode(reply_ids, skip_special_tokens=True).rstrip()
        if not prefill:
            return text.strip()
        # The model may restate the start it was given; do not double it.
        if text.lstrip().startswith(prefill.split(":", 1)[0]):
            return text.strip()
        return prefill + (text if text[:1].isspace() else " " + text)
