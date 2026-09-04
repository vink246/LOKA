"""YAML + CLI merge for the Walker perturbation suite."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from loka.walker_suite.faults import FORBIDDEN_RAW_KEYS, PERTURBATION_KINDS

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "walker_suite.yaml"
)
VALID_BASELINES = ("loka", "fixed_mpc")


@dataclass
class EpisodeDefaults:
    goal_distance_m: float = 10.0
    perturbation_time_s: float = 3.0
    timeout_s: float = 20.0
    speed_goal: float = 1.0


@dataclass
class TestCase:
    name: str
    perturbation: dict[str, Any]
    goal_distance_m: float | None = None
    perturbation_time_s: float | None = None
    timeout_s: float | None = None
    speed_goal: float | None = None

    def episode_params(self, defaults: EpisodeDefaults) -> EpisodeDefaults:
        return EpisodeDefaults(
            goal_distance_m=(
                defaults.goal_distance_m
                if self.goal_distance_m is None
                else float(self.goal_distance_m)
            ),
            perturbation_time_s=(
                defaults.perturbation_time_s
                if self.perturbation_time_s is None
                else float(self.perturbation_time_s)
            ),
            timeout_s=(
                defaults.timeout_s if self.timeout_s is None else float(self.timeout_s)
            ),
            speed_goal=(
                defaults.speed_goal if self.speed_goal is None else float(self.speed_goal)
            ),
        )


@dataclass
class SuiteConfig:
    output_dir: Path = Path("results/walker")
    record: bool = True
    record_camera: str = "side_follow"
    record_fps: float = 30.0
    record_width: int = 1920
    record_height: int = 1080
    log_hz: float = 100.0
    defaults: EpisodeDefaults = field(default_factory=EpisodeDefaults)
    baselines: list[str] = field(default_factory=lambda: ["loka", "fixed_mpc"])
    tests: list[TestCase] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


def _require_mapping(raw: Any, label: str) -> dict:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a mapping")
    return raw


def _validate_perturbation(raw: dict[str, Any], test_name: str) -> dict[str, Any]:
    bad = FORBIDDEN_RAW_KEYS.intersection(raw)
    if bad:
        keys = ", ".join(sorted(bad))
        raise ValueError(
            f"Test '{test_name}' uses forbidden keys {keys}; "
            "specify mass_frac / force_frac instead of raw kg or newtons."
        )
    kind = raw.get("kind")
    if kind not in PERTURBATION_KINDS:
        raise ValueError(f"Test '{test_name}' has unknown perturbation kind {kind!r}")
    if kind == "mass" and "mass_frac" not in raw:
        raise ValueError(f"Test '{test_name}' mass perturbation needs mass_frac")
    if kind == "force" and "force_frac" not in raw:
        raise ValueError(f"Test '{test_name}' force perturbation needs force_frac")
    return dict(raw)


def parse_suite_dict(raw: dict[str, Any]) -> SuiteConfig:
    raw = _require_mapping(raw, "suite config")
    defaults_raw = _require_mapping(raw.get("defaults") or {}, "defaults")
    defaults = EpisodeDefaults(
        goal_distance_m=float(defaults_raw.get("goal_distance_m", 10.0)),
        perturbation_time_s=float(defaults_raw.get("perturbation_time_s", 3.0)),
        timeout_s=float(defaults_raw.get("timeout_s", 20.0)),
        speed_goal=float(defaults_raw.get("speed_goal", 1.0)),
    )

    baselines = [str(b) for b in raw.get("baselines") or list(VALID_BASELINES)]
    unknown = [b for b in baselines if b not in VALID_BASELINES]
    if unknown:
        raise ValueError(f"Unknown baselines: {unknown}. Valid: {VALID_BASELINES}")
    if not baselines:
        raise ValueError("baselines must be non-empty")

    tests_raw = raw.get("tests")
    if not tests_raw or not isinstance(tests_raw, list):
        raise ValueError("tests must be a non-empty list")

    tests: list[TestCase] = []
    names: set[str] = set()
    for item in tests_raw:
        item = _require_mapping(item, "test")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError("Each test needs a name")
        if name in names:
            raise ValueError(f"Duplicate test name '{name}'")
        names.add(name)
        pert = _validate_perturbation(
            _require_mapping(item.get("perturbation"), f"test '{name}' perturbation"),
            name,
        )
        tests.append(
            TestCase(
                name=name,
                perturbation=pert,
                goal_distance_m=item.get("goal_distance_m"),
                perturbation_time_s=item.get("perturbation_time_s"),
                timeout_s=item.get("timeout_s"),
                speed_goal=item.get("speed_goal"),
            )
        )

    return SuiteConfig(
        output_dir=Path(raw.get("output_dir", "results/walker")),
        record=bool(raw.get("record", True)),
        record_camera=str(raw.get("record_camera", "side_follow")),
        record_fps=float(raw.get("record_fps", 30.0)),
        record_width=int(raw.get("record_width", 1920)),
        record_height=int(raw.get("record_height", 1080)),
        log_hz=float(raw.get("log_hz", 100.0)),
        defaults=defaults,
        baselines=baselines,
        tests=tests,
    )


def load_suite_yaml(path: str | Path) -> SuiteConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return parse_suite_dict(raw)


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [item.strip() for item in value.split(",") if item.strip()]
    return parts


def apply_cli_overrides(config: SuiteConfig, args: Any) -> SuiteConfig:
    """Return a copy of config with argparse overrides applied (CLI wins)."""
    tests = list(config.tests)
    baselines = list(config.baselines)
    defaults = EpisodeDefaults(
        goal_distance_m=config.defaults.goal_distance_m,
        perturbation_time_s=config.defaults.perturbation_time_s,
        timeout_s=config.defaults.timeout_s,
        speed_goal=config.defaults.speed_goal,
    )

    test_filter = _split_csv(getattr(args, "tests", None))
    if test_filter is not None:
        by_name = {test.name: test for test in tests}
        missing = [name for name in test_filter if name not in by_name]
        if missing:
            raise ValueError(f"Unknown --tests names: {missing}")
        tests = [by_name[name] for name in test_filter]

    baseline_filter = _split_csv(getattr(args, "baselines", None))
    if baseline_filter is not None:
        unknown = [b for b in baseline_filter if b not in VALID_BASELINES]
        if unknown:
            raise ValueError(f"Unknown --baselines: {unknown}")
        baselines = baseline_filter

    if getattr(args, "goal_distance", None) is not None:
        defaults.goal_distance_m = float(args.goal_distance)
    if getattr(args, "perturbation_time", None) is not None:
        defaults.perturbation_time_s = float(args.perturbation_time)
    if getattr(args, "timeout", None) is not None:
        defaults.timeout_s = float(args.timeout)
    if getattr(args, "speed_goal", None) is not None:
        defaults.speed_goal = float(args.speed_goal)

    output_dir = config.output_dir
    if getattr(args, "output_dir", None):
        output_dir = Path(args.output_dir)

    record = config.record
    if getattr(args, "no_record", False):
        record = False
    elif getattr(args, "record", None) is True:
        record = True

    return SuiteConfig(
        output_dir=output_dir,
        record=record,
        record_camera=config.record_camera,
        record_fps=config.record_fps,
        record_width=config.record_width,
        record_height=config.record_height,
        log_hz=config.log_hz,
        defaults=defaults,
        baselines=baselines,
        tests=tests,
    )


def load_suite_config(args: Any) -> SuiteConfig:
    path = Path(getattr(args, "config", None) or DEFAULT_CONFIG_PATH)
    config = load_suite_yaml(path)
    return apply_cli_overrides(config, args)
