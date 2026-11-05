"""Ablation variants of LOKA, loaded from ``loka/config/loka_ablations.yaml``."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_ABLATION_PATH = Path(__file__).resolve().parents[1] / "config" / "loka_ablations.yaml"
#: Patches compose in this order, whatever order a variant lists them in.
KNOWN_ABLATIONS = ("cost_detector", "raw_telemetry")
RAW_CHANNEL_GROUPS = (
    "base",
    "command",
    "effort_dev",
    "motor_dev",
    "slip",
    "mpc_cost",
    "contact_fz",
    "prop_force",
)
#: Channels that carry the effort residual, motor deficit or slip.
RESIDUAL_CHANNEL_GROUPS = ("effort_dev", "motor_dev", "slip")
COST_CHANNEL_NAMES = ("roll", "pitch", "yaw", "x", "y", "z", "wx", "wy", "wz", "vx", "vy", "vz")


@dataclass(frozen=True)
class RawTelemetrySpec:
    window_s: float = 2.0
    budget: str | int = "match_compressor"
    budget_fraction: float = 1.0
    tokenizer: str = "o200k_base"
    max_hz: float = 50.0
    min_samples: int = 4
    precision: int = 3
    channels: tuple[str, ...] = RAW_CHANNEL_GROUPS
    drop_order: tuple[str, ...] = ("contact_fz", "prop_force", "motor_dev", "mpc_cost", "command")

    @classmethod
    def from_dict(cls, raw: dict | None) -> "RawTelemetrySpec":
        data = dict(raw or {})
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"raw_telemetry has unknown keys {sorted(unknown)}")
        channels = tuple(str(c) for c in data.get("channels", RAW_CHANNEL_GROUPS))
        drop = tuple(str(c) for c in data.get("drop_order", cls.drop_order))
        for name in (*channels, *drop):
            if name not in RAW_CHANNEL_GROUPS:
                raise ValueError(
                    f"raw_telemetry channel {name!r} is not one of {RAW_CHANNEL_GROUPS}"
                )
        if "base" not in channels:
            raise ValueError("raw_telemetry channels must include base")
        if "base" in drop:
            raise ValueError("raw_telemetry drop_order cannot drop base")
        budget = data.get("budget", "match_compressor")
        if budget != "match_compressor":
            budget = int(budget)
            if budget < 1:
                raise ValueError("raw_telemetry budget must be match_compressor or a positive int")
        spec = cls(
            window_s=float(data.get("window_s", 2.0)),
            budget=budget,
            budget_fraction=float(data.get("budget_fraction", 1.0)),
            tokenizer=str(data.get("tokenizer", "o200k_base")),
            max_hz=float(data.get("max_hz", 50.0)),
            min_samples=int(data.get("min_samples", 4)),
            precision=int(data.get("precision", 3)),
            channels=channels,
            drop_order=drop,
        )
        if spec.window_s <= 0 or spec.max_hz <= 0:
            raise ValueError("raw_telemetry window_s and max_hz must be positive")
        if not 0 < spec.budget_fraction <= 1.0:
            raise ValueError("raw_telemetry budget_fraction must be in (0, 1]")
        if spec.min_samples < 1:
            raise ValueError("raw_telemetry min_samples must be >= 1")
        if not 2 <= spec.precision <= 8:
            raise ValueError("raw_telemetry precision must be in 2..8")
        return spec


@dataclass(frozen=True)
class CostDetectorSpec:
    cost_channels: tuple[str, ...] = COST_CHANNEL_NAMES
    stand_threshold: float | None = None
    walk_threshold: float | None = None
    consecutive_steps: int = 250
    clear_below_s: float = 0.5
    hide_residual_telemetry: bool = True

    @classmethod
    def from_dict(cls, raw: dict | None) -> "CostDetectorSpec":
        data = dict(raw or {})
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"cost_detector has unknown keys {sorted(unknown)}")
        channels = data.get("cost_channels", "all")
        if channels in (None, "all"):
            channels = COST_CHANNEL_NAMES
        channels = tuple(str(c) for c in channels)
        for name in channels:
            if name not in COST_CHANNEL_NAMES:
                raise ValueError(f"cost_detector channel {name!r} is not one of {COST_CHANNEL_NAMES}")
        if not channels:
            raise ValueError("cost_detector cost_channels is empty")

        def _threshold(key: str) -> float | None:
            value = data.get(key)
            if value is None:
                return None
            value = float(value)
            if value <= 0:
                raise ValueError(f"cost_detector {key} must be positive")
            return value

        spec = cls(
            cost_channels=channels,
            stand_threshold=_threshold("stand_threshold"),
            walk_threshold=_threshold("walk_threshold"),
            consecutive_steps=int(data.get("consecutive_steps", 250)),
            clear_below_s=float(data.get("clear_below_s", 0.5)),
            hide_residual_telemetry=bool(data.get("hide_residual_telemetry", True)),
        )
        if spec.consecutive_steps < 1 or spec.clear_below_s < 0:
            raise ValueError("cost_detector consecutive_steps must be >= 1 and clear_below_s >= 0")
        return spec

    def require_thresholds(self) -> tuple[float, float]:
        if self.stand_threshold is None or self.walk_threshold is None:
            raise ValueError(
                "cost_detector stand_threshold and walk_threshold are not set. Run "
                "python scripts/calibrate_cost_detector.py --write, or set them in "
                "loka/config/loka_ablations.yaml"
            )
        return self.stand_threshold, self.walk_threshold


@dataclass(frozen=True)
class AblationSet:
    """The ablations one LOKA run applies. Empty is the full system."""

    names: tuple[str, ...] = ()
    params: dict[str, dict] = field(default_factory=dict)
    variant: str | None = None

    def __post_init__(self) -> None:
        for name in self.names:
            if name not in KNOWN_ABLATIONS:
                raise ValueError(f"Unknown ablation {name!r}. Known: {', '.join(KNOWN_ABLATIONS)}")
        ordered = tuple(name for name in KNOWN_ABLATIONS if name in self.names)
        object.__setattr__(self, "names", ordered)
        # Parse once so a bad parameter fails when the set is built.
        self.raw_telemetry
        self.cost_detector

    @property
    def raw_telemetry(self) -> RawTelemetrySpec | None:
        if "raw_telemetry" not in self.names:
            return None
        return RawTelemetrySpec.from_dict(self.params.get("raw_telemetry"))

    @property
    def cost_detector(self) -> CostDetectorSpec | None:
        if "cost_detector" not in self.names:
            return None
        return CostDetectorSpec.from_dict(self.params.get("cost_detector"))

    @property
    def hide_residual(self) -> bool:
        spec = self.cost_detector
        return bool(spec is not None and spec.hide_residual_telemetry)

    def raw_channels(self) -> tuple[str, ...]:
        """Raw channel groups the orchestrator may see."""
        spec = self.raw_telemetry
        if spec is None:
            return ()
        hidden = RESIDUAL_CHANNEL_GROUPS if self.hide_residual else ()
        return tuple(c for c in spec.channels if c not in hidden)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "apply": list(self.names),
            "params": {name: copy.deepcopy(self.params.get(name, {})) for name in self.names},
        }

    @classmethod
    def from_dict(cls, raw: dict | None) -> "AblationSet":
        data = dict(raw or {})
        return cls(
            names=tuple(str(n) for n in data.get("apply") or ()),
            params={str(k): dict(v or {}) for k, v in (data.get("params") or {}).items()},
            variant=data.get("variant"),
        )


def load_ablation_file(path: str | Path | None = None) -> dict:
    path = Path(path or DEFAULT_ABLATION_PATH)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping")
    ablations = raw.get("ablations") or {}
    variants = raw.get("variants") or {}
    if not isinstance(ablations, dict) or not isinstance(variants, dict):
        raise ValueError(f"{path}: ablations and variants must be mappings")
    for name in ablations:
        if name not in KNOWN_ABLATIONS:
            raise ValueError(f"{path}: unknown ablation {name!r}. Known: {', '.join(KNOWN_ABLATIONS)}")
    return {"ablations": ablations, "variants": variants, "path": str(path)}


def known_variants(path: str | Path | None = None) -> list[str]:
    return sorted(load_ablation_file(path)["variants"])


def resolve_variant(name: str, path: str | Path | None = None) -> dict[str, Any]:
    """The variant ``name`` with default parameters filled in.

    Returns ``{variant, apply, params, orchestrator}``; ``orchestrator`` is the
    variant's override mapping (often empty).
    """
    data = load_ablation_file(path)
    variants = data["variants"]
    if name not in variants:
        raise ValueError(
            f"Unknown ablation variant {name!r} in {data['path']}. "
            f"Known: {', '.join(sorted(variants)) or 'none'}"
        )
    entry = dict(variants[name] or {})
    unknown = set(entry) - {"apply", "params", "orchestrator"}
    if unknown:
        raise ValueError(f"variant {name!r} has unknown keys {sorted(unknown)}")
    names = [str(n) for n in entry.get("apply") or ()]
    overrides = entry.get("params") or {}
    params = {}
    for ablation in names:
        merged = copy.deepcopy(data["ablations"].get(ablation) or {})
        merged.update(copy.deepcopy(overrides.get(ablation) or {}))
        params[ablation] = merged
    ablation_set = AblationSet(names=tuple(names), params=params, variant=name)
    payload = ablation_set.to_dict()
    payload["orchestrator"] = copy.deepcopy(entry.get("orchestrator") or {})
    return payload
