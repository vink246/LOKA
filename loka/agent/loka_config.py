"""Load ``loka/config/loka.yaml``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

_PATH = Path(__file__).resolve().parents[1] / "config" / "loka.yaml"


@lru_cache(maxsize=1)
def loka_config() -> dict:
    data = yaml.safe_load(_PATH.read_text(encoding="utf-8")) or {}
    return data
