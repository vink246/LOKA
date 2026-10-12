"""YAML + CLI merge for the G1 evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from loka.control.stacks import STACK_NAMES
from loka.harness.registry import (
    ensure_registered,
    get_metric,
    known_baselines,
    known_metrics,
    known_perturbations,
    known_tasks,
)

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "eval_suite.yaml"
)
_NAME = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
OFFSCREEN_MAX = (1920, 1080)


@dataclass
class ControllerSpec:
    config: str = "loka/config/g1.yaml"
    stack: str = "legacy_dcm"


@dataclass
class OrchestratorSpec:
    model: str = "gpt-5.4-mini"
    latency_mode: str = "fixed_sim"
    latency_s: float = 2.0
    request_timeout_s: float = 120.0


@dataclass
class RecordingSpec:
    enabled: bool = False
    fps: float = 30.0
    width: int = 1280
    height: int = 720
    azimuth: float = 135.0
    elevation: float = -25.0
    distance: float = 3.5
    stand_push_probes: str = "boundary"
    min_push_visible_s: float = 0.3


@dataclass
class BaselineSpec:
    name: str
    type: str
    options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any] | str:
        if not self.options and self.name == self.type:
            return self.name
        payload: dict[str, Any] = {"name": self.name, "type": self.type}
        if self.options:
            payload["options"] = dict(self.options)
        return payload


@dataclass
class PerturbationSpec:
    name: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, **self.params}


@dataclass
class TaskSpec:
    name: str
    type: str
    raw: dict[str, Any]
    perturbations: list[PerturbationSpec]
    metrics: list[str]

    def perturbation(self, name: str) -> PerturbationSpec:
        for item in self.perturbations:
            if item.name == name:
                return item
        raise KeyError(f"Task {self.name!r} has no perturbation {name!r}")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "type": self.type,
            **self.raw,
            "perturbations": [item.to_dict() for item in self.perturbations],
            "metrics": list(self.metrics),
        }
        return payload


@dataclass
class SuiteConfig:
    output_dir: Path = Path("results/g1_eval")
    seed: int = 0
    num_trials: int = 1
    init_noise: float = 0.005
    log_hz: float = 100.0
    workers: int = 1
    controller: ControllerSpec = field(default_factory=ControllerSpec)
    orchestrator: OrchestratorSpec = field(default_factory=OrchestratorSpec)
    recording: RecordingSpec = field(default_factory=RecordingSpec)
    baselines: list[BaselineSpec] = field(default_factory=list)
    tasks: list[TaskSpec] = field(default_factory=list)
    trial: int | None = None
    run_dir: Path | None = None
    unit: str | None = None

    def task(self, name: str) -> TaskSpec:
        for item in self.tasks:
            if item.name == name:
                return item
        raise KeyError(f"Unknown task {name!r}")

    def baseline(self, name: str) -> BaselineSpec:
        for item in self.baselines:
            if item.name == name:
                return item
        raise KeyError(f"Unknown baseline {name!r}")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "output_dir": str(self.output_dir),
            "seed": int(self.seed),
            "num_trials": int(self.num_trials),
            "init_noise": float(self.init_noise),
            "log_hz": float(self.log_hz),
            "workers": int(self.workers),
            "controller": {
                "config": self.controller.config,
                "stack": self.controller.stack,
            },
            "orchestrator": {
                "model": self.orchestrator.model,
                "latency": {
                    "mode": self.orchestrator.latency_mode,
                    "seconds": float(self.orchestrator.latency_s),
                },
                "request_timeout_s": float(self.orchestrator.request_timeout_s),
            },
            "recording": {
                "enabled": bool(self.recording.enabled),
                "fps": float(self.recording.fps),
                "width": int(self.recording.width),
                "height": int(self.recording.height),
                "camera": {
                    "azimuth": float(self.recording.azimuth),
                    "elevation": float(self.recording.elevation),
                    "distance": float(self.recording.distance),
                },
                "stand_push_probes": self.recording.stand_push_probes,
                "min_push_visible_s": float(self.recording.min_push_visible_s),
            },
            "baselines": [item.to_dict() for item in self.baselines],
            "tasks": [item.to_dict() for item in self.tasks],
        }
        if self.run_dir is not None:
            payload["run_dir"] = str(self.run_dir)
        if self.trial is not None:
            payload["trial"] = int(self.trial)
        if self.unit is not None:
            payload["unit"] = self.unit
        return payload


def _mapping(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a mapping")
    return raw


def _check_name(name: str, label: str) -> str:
    text = str(name).strip()
    if not text or any(ch not in _NAME for ch in text):
        raise ValueError(
            f"{label} {name!r} must be letters, digits, underscore or hyphen"
        )
    return text


def _parse_perturbation(raw: Any, label: str) -> PerturbationSpec:
    data = dict(_mapping(raw, label))
    name = _check_name(data.pop("name"), f"{label} name")
    kind = data.pop("kind", None)
    if kind not in known_perturbations():
        raise ValueError(
            f"{label} has unknown perturbation kind {kind!r}. "
            f"Known: {', '.join(known_perturbations())}"
        )
    return PerturbationSpec(name=name, kind=str(kind), params=data)


def _stand_push_perturbation(raw: dict[str, Any], task_name: str) -> PerturbationSpec:
    mass = raw.get("mass") or {}
    if not isinstance(mass, dict):
        raise ValueError(f"Task {task_name!r} mass must be a mapping")
    body = str(mass.get("body", "left_shoulder_roll_link"))
    delta = float(mass.get("delta_kg", 10.0))
    return PerturbationSpec(
        name="shoulder_mass",
        kind="mass",
        params={"body": body, "delta_kg": delta},
    )


def _parse_task(raw: Any) -> TaskSpec:
    data = dict(_mapping(raw, "task"))
    name = _check_name(data.pop("name"), "task name")
    kind = data.pop("type", None)
    if kind not in known_tasks():
        raise ValueError(
            f"Task {name!r} has unknown type {kind!r}. Known: {', '.join(known_tasks())}"
        )
    metrics = data.pop("metrics", None)
    if not isinstance(metrics, list) or not metrics:
        raise ValueError(f"Task {name!r} needs a non-empty metrics list")
    metric_names = []
    for metric in metrics:
        text = str(metric)
        if text not in known_metrics():
            raise ValueError(
                f"Task {name!r} has unknown metric {text!r}. "
                f"Known: {', '.join(known_metrics())}"
            )
        scope = get_metric(text).scope
        expect = "trial" if kind == "stand_push" else "episode"
        if scope != expect:
            raise ValueError(
                f"Task {name!r} metric {text!r} has scope {scope}, expected {expect}"
            )
        metric_names.append(text)
    pert_raw = data.pop("perturbations", None)
    if kind == "stand_push":
        if pert_raw is None:
            perturbations = [_stand_push_perturbation(data, name)]
        else:
            if not isinstance(pert_raw, list) or len(pert_raw) != 1:
                raise ValueError(
                    f"Task {name!r} stand_push takes exactly one mass perturbation"
                )
            perturbations = [_parse_perturbation(pert_raw[0], f"task {name}")]
            if perturbations[0].kind != "mass":
                raise ValueError(f"Task {name!r} stand_push perturbation must be mass")
    else:
        if not isinstance(pert_raw, list) or not pert_raw:
            raise ValueError(f"Task {name!r} needs a perturbations list")
        perturbations = [
            _parse_perturbation(item, f"task {name} perturbation") for item in pert_raw
        ]
        seen = [item.name for item in perturbations]
        if len(set(seen)) != len(seen):
            raise ValueError(f"Task {name!r} has duplicate perturbation names")
        goal = data.get("goal")
        if not isinstance(goal, dict) or "xy" not in goal:
            raise ValueError(f"Task {name!r} needs goal.xy")
        if "timeout_s" not in data:
            raise ValueError(f"Task {name!r} needs timeout_s")
    return TaskSpec(
        name=name,
        type=str(kind),
        raw=data,
        perturbations=perturbations,
        metrics=metric_names,
    )


def _parse_baseline(raw: Any) -> BaselineSpec:
    if isinstance(raw, str):
        name = _check_name(raw, "baseline")
        if name not in known_baselines():
            raise ValueError(
                f"Unknown baseline {name!r}. Known: {', '.join(known_baselines())}"
            )
        return BaselineSpec(name=name, type=name)
    data = _mapping(raw, "baseline")
    kind = str(data.get("type", data.get("name", "")))
    name = _check_name(data.get("name", kind), "baseline")
    if kind not in known_baselines():
        raise ValueError(
            f"Unknown baseline type {kind!r}. Known: {', '.join(known_baselines())}"
        )
    options = data.get("options") or {}
    if not isinstance(options, dict):
        raise ValueError(f"Baseline {name!r} options must be a mapping")
    return BaselineSpec(name=name, type=kind, options=dict(options))


def _parse_recording(raw: Any) -> RecordingSpec:
    data = dict(raw or {})
    camera = dict(data.get("camera") or {})
    spec = RecordingSpec(
        enabled=bool(data.get("enabled", False)),
        fps=float(data.get("fps", 30.0)),
        width=int(data.get("width", 1280)),
        height=int(data.get("height", 720)),
        azimuth=float(camera.get("azimuth", 135.0)),
        elevation=float(camera.get("elevation", -25.0)),
        distance=float(camera.get("distance", 3.5)),
        stand_push_probes=str(data.get("stand_push_probes", "boundary")),
        min_push_visible_s=float(data.get("min_push_visible_s", 0.3)),
    )
    if spec.fps <= 0:
        raise ValueError("recording.fps must be positive")
    if spec.width < 1 or spec.height < 1:
        raise ValueError("recording width and height must be positive")
    if spec.width > OFFSCREEN_MAX[0] or spec.height > OFFSCREEN_MAX[1]:
        raise ValueError(
            f"recording size {spec.width}x{spec.height} exceeds the scene "
            f"offscreen buffer {OFFSCREEN_MAX[0]}x{OFFSCREEN_MAX[1]}"
        )
    if spec.stand_push_probes not in ("boundary", "all"):
        raise ValueError("recording.stand_push_probes must be boundary or all")
    if spec.distance <= 0:
        raise ValueError("recording camera distance must be positive")
    return spec


def _parse_orchestrator(raw: Any) -> OrchestratorSpec:
    data = dict(raw or {})
    latency = dict(data.get("latency") or {})
    spec = OrchestratorSpec(
        model=str(data.get("model", "gpt-5.4-mini")),
        latency_mode=str(latency.get("mode", "fixed_sim")),
        latency_s=float(latency.get("seconds", 2.0)),
        request_timeout_s=float(data.get("request_timeout_s", 120.0)),
    )
    if spec.latency_mode not in ("wall", "fixed_sim", "pause"):
        raise ValueError(
            f"orchestrator latency mode must be wall, fixed_sim or pause, "
            f"got {spec.latency_mode!r}"
        )
    if spec.latency_s < 0 or spec.request_timeout_s < 0:
        raise ValueError("orchestrator latency and request_timeout_s must be >= 0")
    return spec


def parse_suite(raw: dict[str, Any]) -> SuiteConfig:
    ensure_registered()
    data = _mapping(raw, "suite")
    controller = dict(data.get("controller") or {})
    stack = str(controller.get("stack", "legacy_dcm"))
    if stack not in STACK_NAMES:
        raise ValueError(f"Unknown controller stack {stack!r}. Known: {STACK_NAMES}")
    config_path = str(controller.get("config", "loka/config/g1.yaml"))
    baselines_raw = data.get("baselines")
    if not isinstance(baselines_raw, list) or not baselines_raw:
        raise ValueError("baselines must be a non-empty list")
    baselines = [_parse_baseline(item) for item in baselines_raw]
    if len({item.name for item in baselines}) != len(baselines):
        raise ValueError("duplicate baseline names")
    tasks_raw = data.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise ValueError("tasks must be a non-empty list")
    tasks = [_parse_task(item) for item in tasks_raw]
    if len({item.name for item in tasks}) != len(tasks):
        raise ValueError("duplicate task names")
    num_trials = int(data.get("num_trials", 1))
    if num_trials < 1:
        raise ValueError("num_trials must be >= 1")
    workers = int(data.get("workers", 1))
    if workers < 1:
        raise ValueError("workers must be >= 1")
    init_noise = float(data.get("init_noise", 0.005))
    if init_noise < 0:
        raise ValueError("init_noise must be >= 0")
    log_hz = float(data.get("log_hz", 100.0))
    if log_hz <= 0:
        raise ValueError("log_hz must be positive")
    trial = data.get("trial")
    unit = data.get("unit")
    run_dir = data.get("run_dir")
    return SuiteConfig(
        output_dir=Path(data.get("output_dir", "results/g1_eval")),
        seed=int(data.get("seed", 0)),
        num_trials=num_trials,
        init_noise=init_noise,
        log_hz=log_hz,
        workers=workers,
        controller=ControllerSpec(config=config_path, stack=stack),
        orchestrator=_parse_orchestrator(data.get("orchestrator")),
        recording=_parse_recording(data.get("recording")),
        baselines=baselines,
        tasks=tasks,
        trial=None if trial is None else int(trial),
        run_dir=None if not run_dir else Path(run_dir),
        unit=None if unit is None else str(unit),
    )


def load_suite_yaml(path: str | Path) -> SuiteConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping")
    return parse_suite(raw)


def _split_csv(text: str | None) -> list[str] | None:
    if text is None:
        return None
    names = [part.strip() for part in str(text).split(",") if part.strip()]
    return names or None


def apply_cli_overrides(config: SuiteConfig, args: Any) -> SuiteConfig:
    """Return a copy of ``config`` with CLI flags applied."""
    tasks = list(config.tasks)
    task_filter = _split_csv(getattr(args, "tasks", None))
    if task_filter is not None:
        by_name = {task.name: task for task in tasks}
        missing = [name for name in task_filter if name not in by_name]
        if missing:
            raise ValueError(f"Unknown --tasks names: {missing}")
        tasks = [by_name[name] for name in task_filter]

    baselines = list(config.baselines)
    baseline_filter = _split_csv(getattr(args, "baselines", None))
    if baseline_filter is not None:
        by_name = {item.name: item for item in baselines}
        missing = [name for name in baseline_filter if name not in by_name]
        if missing:
            raise ValueError(f"Unknown --baselines names: {missing}")
        baselines = [by_name[name] for name in baseline_filter]

    num_trials = config.num_trials
    if getattr(args, "num_trials", None) is not None:
        num_trials = int(args.num_trials)
        if num_trials < 1:
            raise ValueError("--num-trials must be >= 1")
    seed = config.seed if getattr(args, "seed", None) is None else int(args.seed)
    init_noise = config.init_noise
    if getattr(args, "init_noise", None) is not None:
        init_noise = float(args.init_noise)
        if init_noise < 0:
            raise ValueError("--init-noise must be >= 0")
    workers = config.workers
    if getattr(args, "workers", None) is not None:
        workers = int(args.workers)
        if workers < 1:
            raise ValueError("--workers must be >= 1")
    output_dir = config.output_dir
    if getattr(args, "output_dir", None):
        output_dir = Path(args.output_dir)
    recording = config.recording
    if getattr(args, "no_record", False):
        recording = RecordingSpec(**{**recording.__dict__, "enabled": False})
    elif getattr(args, "record", False):
        recording = RecordingSpec(**{**recording.__dict__, "enabled": True})
    trial = config.trial
    if getattr(args, "trial", None) is not None:
        trial = int(args.trial)
        if trial < 0 or trial >= num_trials:
            raise ValueError(f"--trial must be in 0..{num_trials - 1}")
    run_dir = config.run_dir
    if getattr(args, "run_dir", None):
        run_dir = Path(args.run_dir)
    unit = config.unit
    if getattr(args, "unit", None):
        unit = str(args.unit)
    return SuiteConfig(
        output_dir=output_dir,
        seed=seed,
        num_trials=num_trials,
        init_noise=init_noise,
        log_hz=config.log_hz,
        workers=workers,
        controller=config.controller,
        orchestrator=config.orchestrator,
        recording=recording,
        baselines=baselines,
        tasks=tasks,
        trial=trial,
        run_dir=run_dir,
        unit=unit,
    )


def load_suite_config(args: Any) -> SuiteConfig:
    path = Path(getattr(args, "config", None) or DEFAULT_CONFIG_PATH)
    return apply_cli_overrides(load_suite_yaml(path), args)
