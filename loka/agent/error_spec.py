"""Structured tracking-error / success criteria for LOKA."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

VALID_SIGNALS = frozenset({"qpos", "qvel", "frame"})
VALID_MODES = frozenset({
    "below_target",
    "above_target",
    "abs_above",
    "abs_deviation",
})


@dataclass(frozen=True)
class ErrorTerm:
    name: str
    signal: str  # qpos | qvel
    index: int
    mode: str
    target: float
    tolerance: float
    weight: float = 1.0
    offset: float = 0.0

    def read(self, data, frame: dict | None = None) -> float:
        if self.signal == "frame":
            if not frame:
                return float(self.offset)
            return float(self.offset + frame.get(self.name, 0.0))
        arr = data.qpos if self.signal == "qpos" else data.qvel
        return float(self.offset + arr[self.index])

    def read_frame(self, frame: dict) -> float:
        if self.signal == "frame":
            return float(self.offset + frame.get(self.name, 0.0))
        arr = frame[self.signal]
        return float(self.offset + arr[self.index])

    def excess(self, value: float) -> float:
        """Positive amount outside the deadband; 0 inside."""
        if self.mode == "below_target":
            floor = self.target - self.tolerance
            return max(0.0, floor - value)
        if self.mode == "above_target":
            ceiling = self.target + self.tolerance
            return max(0.0, value - ceiling)
        # abs_above / abs_deviation: fail when |value - target| > tolerance
        return max(0.0, abs(value - self.target) - self.tolerance)


@dataclass
class ErrorSpec:
    trigger_threshold: float = 0.25
    terms: list[ErrorTerm] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger_threshold": self.trigger_threshold,
            "terms": [asdict(term) for term in self.terms],
        }


def parse_error_tracking(raw: dict, nq: int, nv: int) -> ErrorSpec:
    """Validate LLM Error_Tracking YAML into an ErrorSpec."""
    if not isinstance(raw, dict):
        raise ValueError("Error_Tracking must be a mapping")

    threshold = float(raw.get("trigger_threshold", 0.25))
    if threshold < 0:
        raise ValueError("trigger_threshold must be >= 0")

    raw_terms = raw.get("terms")
    if not raw_terms or not isinstance(raw_terms, list):
        raise ValueError("Error_Tracking.terms must be a non-empty list")

    terms: list[ErrorTerm] = []
    for item in raw_terms:
        if not isinstance(item, dict):
            raise ValueError("Each Error_Tracking term must be a mapping")

        name = str(item.get("name", "")).strip()
        signal = str(item.get("signal", "")).strip().lower()
        mode = str(item.get("mode", "")).strip().lower()
        if not name:
            raise ValueError("Error term missing name")
        if signal not in VALID_SIGNALS:
            raise ValueError(f"Invalid signal '{signal}' for term '{name}'")
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}' for term '{name}'")

        if signal == "frame":
            index = int(item.get("index", 0))
        else:
            index = int(item["index"])
            limit = nq if signal == "qpos" else nv
            if index < 0 or index >= limit:
                raise ValueError(
                    f"Term '{name}' index {index} out of range for {signal} (0..{limit - 1})"
                )

        terms.append(
            ErrorTerm(
                name=name,
                signal=signal,
                index=index,
                mode=mode,
                target=float(item.get("target", 0.0)),
                tolerance=float(item.get("tolerance", 0.0)),
                weight=float(item.get("weight", 1.0)),
                offset=float(item.get("offset", 0.0)),
            )
        )

    return ErrorSpec(trigger_threshold=threshold, terms=terms)


def get_tracking_error(data, error_spec: ErrorSpec, frame: dict | None = None) -> float:
    total = 0.0
    for term in error_spec.terms:
        total += term.excess(term.read(data, frame)) * term.weight
    return total


def format_error_spec(error_spec: ErrorSpec) -> str:
    """Human-readable ErrorSpec for prompts and logs."""
    lines = [
        f"trigger_threshold: {error_spec.trigger_threshold}",
        "terms:",
    ]
    for term in error_spec.terms:
        lines.append(
            f"  - {term.name}: {term.signal}[{term.index}]"
            f"{f' + {term.offset}' if term.offset else ''}"
            f" mode={term.mode} target={term.target} "
            f"tol={term.tolerance} weight={term.weight}"
        )
    return "\n".join(lines)
