"""Load DR-RL YAML and resolve checkpoint / XML paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DR_RL_DIR = Path(__file__).resolve().parent
LOKA_ROOT = DR_RL_DIR.parent.parent
DEFAULT_CONFIG_PATH = DR_RL_DIR / "config.yaml"
GYM_CONFIG_PATH = DR_RL_DIR / "config_gym.yaml"
DEFAULT_XML_PATH = LOKA_ROOT / "models" / "walker" / "task.xml"


def load_dr_config(path: str | Path | None = None) -> dict[str, Any]:
    env_path = os.environ.get("LOKA_DR_RL_CONFIG")
    config_path = Path(path or env_path or DEFAULT_CONFIG_PATH)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"DR-RL config at {config_path} must be a mapping")
    raw["_config_path"] = str(config_path)
    return raw


def walker_xml_path(config: dict[str, Any] | None = None) -> Path:
    cfg = config if config is not None else {}
    override = cfg.get("xml_path") or os.environ.get("LOKA_DR_RL_XML")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_XML_PATH


def checkpoint_dir(config: dict[str, Any]) -> Path:
    rel = Path(config.get("checkpoint", {}).get("dir", "results/dr_rl"))
    return rel if rel.is_absolute() else LOKA_ROOT / rel


def best_checkpoint_dir(config: dict[str, Any]) -> Path:
    rel = Path(config.get("checkpoint", {}).get("best_dir", "results/dr_rl/best"))
    return rel if rel.is_absolute() else LOKA_ROOT / rel


def checkpoint_zip_name(config: dict[str, Any]) -> str:
    return str(config.get("checkpoint", {}).get("zip_name", "ppo_walker.zip"))


def vecnormalize_name(config: dict[str, Any]) -> str:
    return str(
        config.get("checkpoint", {}).get("vecnormalize_name", "vecnormalize.pkl")
    )


def _vecnormalize_beside(policy: Path, vec_name: str) -> Path | None:
    """Find VecNormalize stats next to a policy zip (final, best, or periodic)."""
    candidates = [policy.with_name(vec_name)]
    stem = policy.stem
    if stem.endswith("_steps"):
        parts = stem.rsplit("_", 2)
        if len(parts) == 3 and parts[2] == "steps" and parts[1].isdigit():
            prefix, steps, _ = parts
            candidates.append(
                policy.with_name(f"{prefix}_vecnormalize_{steps}_steps.pkl")
            )
    for path in candidates:
        if path.is_file():
            return path
    return None


def resolve_checkpoint_paths(
    config: dict[str, Any] | None = None,
    *,
    checkpoint: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """Return (policy.zip, vecnormalize.pkl or None)."""
    env_ckpt = os.environ.get("LOKA_DR_RL_CHECKPOINT")
    ckpt = Path(checkpoint or env_ckpt) if (checkpoint or env_ckpt) else None
    cfg = config if config is not None else load_dr_config()
    zip_name = checkpoint_zip_name(cfg)
    vec_name = vecnormalize_name(cfg)

    if ckpt is not None:
        ckpt = ckpt.expanduser().resolve()
        if ckpt.is_dir():
            for name in (zip_name, "best_model.zip"):
                policy = ckpt / name
                if policy.is_file():
                    return policy, _vecnormalize_beside(policy, vec_name)
            policy = ckpt / zip_name
            return policy, _vecnormalize_beside(policy, vec_name)
        return ckpt, _vecnormalize_beside(ckpt, vec_name)

    candidates = [best_checkpoint_dir(cfg), checkpoint_dir(cfg)]
    names = (zip_name, "best_model.zip")
    for folder in candidates:
        for name in names:
            policy = folder / name
            if policy.is_file():
                return policy, _vecnormalize_beside(policy, vec_name)
    return checkpoint_dir(cfg) / zip_name, None
