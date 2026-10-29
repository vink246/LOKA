"""Orchestrator model registry (``loka/config/orchestrator_models.yaml``).

Resolves an orchestrator model name to a provider entry. API models need no
entry: ``claude-*`` routes to Anthropic, everything else to OpenAI. Local
models are listed with their checkpoint, server and Slurm request.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "config" / "orchestrator_models.yaml"
PROVIDERS = ("openai", "anthropic", "local")
ENGINES = ("vllm", "hf")
DEFAULT_MODELS_DIR = "~/scratch/models"
#: One GPU env serves both engines (scripts/setup_vllm_env.sh). A model entry's
#: server_env, or submit_pace_ice --hf-prefix / --vllm-prefix, overrides it.
DEFAULT_SERVER_ENVS = {
    "vllm": "~/scratch/envs/loka-vllm",
    "hf": "~/scratch/envs/loka-vllm",
}

#: Hugging Face organisations of Chinese origin. Their weights, and fine-tunes
#: built on them, cannot run on PACE.
BLOCKED_ORGS = (
    "qwen",
    "deepseek-ai",
    "thudm",
    "zai-org",
    "01-ai",
    "internlm",
    "moonshotai",
    "minimaxai",
    "baichuan-inc",
    "openbmb",
    "tencent",
    "baidu",
    "xverse",
    "stepfun-ai",
)
BLOCKED_FAMILIES = (
    "qwen",
    "deepseek",
    "glm",
    "chatglm",
    "yi",
    "internlm",
    "kimi",
    "minimax",
    "baichuan",
    "minicpm",
    "hunyuan",
    "ernie",
)


def _blocked_reason(entry: dict) -> str | None:
    repo = str(entry.get("hf_repo") or "")
    org = repo.split("/", 1)[0].lower() if "/" in repo else ""
    if org in BLOCKED_ORGS:
        return f"Hugging Face org {org!r}"
    family = str(entry.get("base_family") or "").lower()
    if family in BLOCKED_FAMILIES:
        return f"base family {family!r}"
    name = str(entry.get("key") or entry.get("model") or "").lower()
    for token in BLOCKED_FAMILIES:
        if len(token) <= 2:
            continue
        if token in name:
            return f"model name contains {token!r}"
    return None


def check_origin(entry: dict) -> None:
    reason = _blocked_reason(entry)
    if reason is not None:
        raise ValueError(
            f"orchestrator model {entry.get('key') or entry.get('model')!r} is blocked "
            f"({reason}): models of Chinese origin cannot run on PACE"
        )


def load_model_registry(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path or DEFAULT_REGISTRY_PATH)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping")
    defaults = raw.get("defaults") or {}
    models = raw.get("models") or {}
    for key, entry in models.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: model {key!r} must be a mapping")
        provider = entry.get("provider")
        if provider not in PROVIDERS:
            raise ValueError(f"{path}: model {key!r} provider must be one of {PROVIDERS}")
        check_origin({**entry, "key": key})
    return {"defaults": defaults, "models": models, "path": str(path)}


def infer_provider(name: str) -> str:
    text = str(name).strip().lower()
    if text.startswith("claude"):
        return "anthropic"
    return "openai"


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_model(
    name: str,
    *,
    provider: str | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """The full entry for ``name``.

    The result always has ``key`` (registry key or ``name``), ``provider`` and
    ``model`` (the id sent to the API or the local server). Local entries also
    carry ``path``, ``engine``, ``server_env`` and ``slurm``.
    """
    if not name:
        raise ValueError("orchestrator model is empty")
    registry = load_model_registry(path)
    entry = registry["models"].get(name)
    if entry is None:
        chosen = provider or infer_provider(name)
        if chosen == "local":
            raise ValueError(
                f"local orchestrator model {name!r} is not in {registry['path']}. "
                f"Known local models: {', '.join(known_local_models(path))}"
            )
        if chosen not in PROVIDERS:
            raise ValueError(f"orchestrator provider must be one of {PROVIDERS}, got {chosen!r}")
        resolved = _merge(registry["defaults"].get(chosen) or {}, {})
        resolved.update({"key": name, "provider": chosen, "model": name})
        check_origin(resolved)
        return resolved
    if provider is not None and provider != entry["provider"]:
        raise ValueError(
            f"orchestrator model {name!r} is a {entry['provider']} model, not {provider}"
        )
    resolved = _merge(registry["defaults"].get(entry["provider"]) or {}, entry)
    resolved["key"] = name
    resolved.setdefault("model", name)
    if resolved["provider"] == "local":
        engine = resolved.get("engine", "vllm")
        if engine not in ENGINES:
            raise ValueError(f"model {name!r} engine must be one of {ENGINES}")
        resolved["engine"] = engine
        resolved.setdefault("path", f"{DEFAULT_MODELS_DIR}/{name}")
        resolved.setdefault("server_env", DEFAULT_SERVER_ENVS[engine])
        resolved.setdefault("slurm", {})
    check_origin(resolved)
    return resolved


def known_local_models(path: str | Path | None = None) -> list[str]:
    registry = load_model_registry(path)
    return sorted(k for k, v in registry["models"].items() if v.get("provider") == "local")


def expand(path: str) -> Path:
    return Path(str(path)).expanduser()
