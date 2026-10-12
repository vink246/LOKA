"""Name registries for baselines, tasks, perturbations and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Metric:
    name: str
    scope: str
    fn: Callable[..., Any]

_BASELINES: dict[str, type] = {}
_TASKS: dict[str, type] = {}
_PERTURBATIONS: dict[str, type] = {}
_METRICS: dict[str, Any] = {}
_loaded = False


def register_baseline(name: str) -> Callable[[type], type]:
    def decorate(cls: type) -> type:
        if name in _BASELINES and _BASELINES[name] is not cls:
            raise ValueError(f"Baseline {name!r} is already registered")
        _BASELINES[name] = cls
        return cls

    return decorate


def register_task(name: str) -> Callable[[type], type]:
    def decorate(cls: type) -> type:
        if name in _TASKS and _TASKS[name] is not cls:
            raise ValueError(f"Task {name!r} is already registered")
        _TASKS[name] = cls
        return cls

    return decorate


def register_perturbation(name: str) -> Callable[[type], type]:
    def decorate(cls: type) -> type:
        if name in _PERTURBATIONS and _PERTURBATIONS[name] is not cls:
            raise ValueError(f"Perturbation {name!r} is already registered")
        _PERTURBATIONS[name] = cls
        return cls

    return decorate


def register_metric(name: str, *, scope: str) -> Callable:
    """Register a metric.

    ``scope`` is ``episode`` (one rollout) or ``trial`` (every probe of a
    search, such as the standing push).
    """
    if scope not in ("episode", "trial"):
        raise ValueError(f"metric scope must be episode or trial, got {scope!r}")

    def decorate(fn: Callable) -> Callable:
        _METRICS[name] = Metric(name=name, scope=scope, fn=fn)
        return fn

    return decorate


def get_baseline(name: str) -> type:
    _missing(name, _BASELINES, "baseline")
    return _BASELINES[name]


def get_task(name: str) -> type:
    _missing(name, _TASKS, "task")
    return _TASKS[name]


def get_perturbation(name: str) -> type:
    _missing(name, _PERTURBATIONS, "perturbation")
    return _PERTURBATIONS[name]


def get_metric(name: str):
    _missing(name, _METRICS, "metric")
    return _METRICS[name]


def known_baselines() -> list[str]:
    return sorted(_BASELINES)


def known_tasks() -> list[str]:
    return sorted(_TASKS)


def known_perturbations() -> list[str]:
    return sorted(_PERTURBATIONS)


def known_metrics() -> list[str]:
    return sorted(_METRICS)


def _missing(name: str, table: dict, kind: str) -> None:
    if name not in table:
        known = ", ".join(sorted(table)) or "none"
        raise KeyError(f"Unknown {kind} {name!r}. Known: {known}")


def ensure_registered() -> None:
    """Import the built-in registrations once."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    import loka.harness.baselines  # noqa: F401
    import loka.harness.metrics  # noqa: F401
    import loka.harness.perturbations  # noqa: F401
    import loka.harness.tasks  # noqa: F401
