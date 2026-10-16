"""Evaluation baselines.

``low_level`` is the MPC + whole-body controller with no orchestrator.
``loka`` is that same controller with the language orchestrator in the loop.
``unitree_rl`` is the unitree_rl_lab velocity policy on the same plant.
``navila`` is the NaVILA mid-level navigation VLA on the MPC or unitree_rl plant.

A new baseline subclasses :class:`Baseline`, registers a name, and is
selected from the suite YAML as ``{name, type, options}``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from loka.agent.faults import FaultSpec
from loka.agent.interactive import FaultVizState, commit_fault
from loka.agent.runtime import LokaConfig, LokaRuntime
from loka.control.locomotion import LocomotionConfig
from loka.harness.registry import get_baseline, register_baseline
from loka.harness.stochastic import apply_seeded_state_noise
from loka.sim import Simulation, UnitreeRLSimulation


def locomotion_config(suite) -> LocomotionConfig:
    """A fresh controller config. Do not share one across simulations."""
    config = LocomotionConfig.from_yaml(suite.controller.config)
    config.stack = suite.controller.stack
    return config


class Baseline:
    """One controllable plant. A new instance is used for every episode."""

    supports_directive = False

    def reset(self, seed: int, init_noise: float) -> None:
        raise NotImplementedError

    def step(self) -> None:
        raise NotImplementedError

    def command_goal(self, xy, speed: float, radius: float) -> None:
        raise NotImplementedError

    def submit_directive(self, text: str, xy) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not take language")

    def inject(self, fault: FaultSpec) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None

    @property
    def sim(self) -> Simulation:
        raise NotImplementedError

    @property
    def viz(self) -> FaultVizState:
        raise NotImplementedError

    @property
    def turns(self) -> list[dict]:
        return []

    @property
    def fell(self) -> bool:
        return bool(self.sim.fell)

    @property
    def goal_xy(self):
        return getattr(self, "_goal_xy", None)

    def bias_nm(self) -> float | None:
        return None

    def mutation_count(self) -> int:
        return 0


def _apply_goal(controller, xy, speed: float, radius: float) -> None:
    controller.set_task_targets(
        {
            "gait.mode": "walk",
            "gait.speed": float(speed),
            "gait.goal_x": float(xy[0]),
            "gait.goal_y": float(xy[1]),
            "gait.goal_tolerance": float(radius),
        }
    )


def _inject(sim: Simulation, viz: FaultVizState, fault: FaultSpec) -> None:
    ok, message = commit_fault(sim, viz, fault)
    print(message)
    if not ok:
        raise RuntimeError(message)


@register_baseline("low_level")
class LowLevelBaseline(Baseline):
    """MPC + WBC only. The effort observer is not in this loop, so a model
    error cannot force the gait to stand.
    """

    def __init__(self, spec, suite, log_path: Path | None = None) -> None:
        del spec, log_path
        self._sim = Simulation(locomotion_config(suite))
        self._viz = FaultVizState(external_body_id=self._sim.pelvis_id)
        self._goal_xy = None

    def reset(self, seed: int, init_noise: float) -> None:
        self._sim.reset()
        apply_seeded_state_noise(self._sim.model, self._sim.data, seed, init_noise)
        self._viz = FaultVizState(external_body_id=self._sim.pelvis_id)
        self._goal_xy = None

    def step(self) -> None:
        self._sim.step()
        self._viz.external_force = self._sim._applied_force()
        self._viz.external_body_id = self._sim.pelvis_id

    def command_goal(self, xy, speed: float, radius: float) -> None:
        self._goal_xy = np.asarray(xy, dtype=float)
        _apply_goal(self._sim.controller, xy, speed, radius)

    def inject(self, fault: FaultSpec) -> None:
        _inject(self._sim, self._viz, fault)

    @property
    def sim(self) -> Simulation:
        return self._sim

    @property
    def viz(self) -> FaultVizState:
        return self._viz


@register_baseline("unitree_rl")
class UnitreeRLBaseline(LowLevelBaseline):
    """The unitree_rl_lab G1-29dof velocity policy, sim2sim on the same scene.

    Options: ``policy_dir`` (a directory with ``deploy.yaml`` and
    ``policy.npz``; default is the vendored v0 policy) and ``min_speed``
    (slowest walking command [m/s]).
    """

    def __init__(self, spec, suite, log_path: Path | None = None) -> None:
        del log_path
        options = dict(spec.options or {})
        self._sim = UnitreeRLSimulation(
            locomotion_config(suite),
            policy_dir=options.get("policy_dir"),
            min_speed=options.get("min_speed"),
        )
        self._viz = FaultVizState(external_body_id=self._sim.pelvis_id)
        self._goal_xy = None


@register_baseline("loka")
class LokaBaseline(Baseline):
    """The full LOKA loop: the same plant, plus the orchestrator."""

    supports_directive = True

    def __init__(self, spec, suite, log_path: Path | None = None) -> None:
        del spec
        self._sim = Simulation(locomotion_config(suite))
        orch = suite.orchestrator
        self.runtime = LokaRuntime(
            self._sim,
            LokaConfig(
                enable_llm=True,
                llm_model=orch.model,
                llm_latency_mode=orch.latency_mode,
                llm_latency_s=orch.latency_s,
                llm_request_timeout_s=orch.request_timeout_s,
                print_compressor=False,
                enable_session_log=log_path is not None,
                log_path=log_path,
            ),
        )
        self.runtime.effort_observer.printout = False
        self._goal_xy = None

    def reset(self, seed: int, init_noise: float) -> None:
        # Belief, sessions and an in-flight LLM call do not have a safe reset.
        # Callers build a new baseline per episode instead of reusing one.
        self._sim.reset()
        apply_seeded_state_noise(self._sim.model, self._sim.data, seed, init_noise)
        self._goal_xy = None

    def step(self) -> None:
        self.runtime.step()

    def command_goal(self, xy, speed: float, radius: float) -> None:
        self._goal_xy = np.asarray(xy, dtype=float)
        _apply_goal(self._sim.controller, xy, speed, radius)

    def submit_directive(self, text: str, xy) -> None:
        self._goal_xy = np.asarray(xy, dtype=float)
        self.runtime.submit_operator(text)

    def inject(self, fault: FaultSpec) -> None:
        _inject(self._sim, self.runtime.viz, fault)

    def close(self) -> None:
        return None

    @property
    def sim(self) -> Simulation:
        return self._sim

    @property
    def viz(self) -> FaultVizState:
        return self.runtime.viz

    @property
    def turns(self) -> list[dict]:
        return self.runtime.turns

    def bias_nm(self) -> float | None:
        return self.runtime._bias_nm()

    def mutation_count(self) -> int:
        return len(self.runtime.loka_state.get("mutations") or [])


@register_baseline("navila")
class NavilaBaseline(Baseline):
    """NaVILA mid-level navigation VLA on the MPC or unitree_rl plant.

    Options: ``plant`` (``mpc`` | ``unitree_rl``), ``model_path`` (local
    checkpoint directory; required for real inference), ``device``,
    ``policy_dir`` / ``min_speed`` when ``plant`` is ``unitree_rl``.
    """

    supports_directive = True

    def __init__(self, spec, suite, log_path: Path | None = None) -> None:
        from loka.agent.loka_config import loka_config
        from loka.vla.navila import (
            NavilaConfig,
            NavilaPolicy,
            NavilaRuntime,
            require_navila_checkpoint,
            require_navila_gpu,
        )

        options = dict(spec.options or {})
        plant = str(options.get("plant", "mpc")).strip().lower()
        settings = (loka_config().get("navila") or {}).copy()
        settings.update({k: v for k, v in options.items() if k not in ("plant", "policy_dir", "min_speed")})
        if options.get("model_path"):
            settings["model_path"] = options["model_path"]
        cfg = NavilaConfig.from_dict(settings)
        model_path = require_navila_checkpoint(cfg.model_path)
        require_navila_gpu(cfg.device, cfg.min_free_gb, cfg.min_compute_capability)
        camera = cfg.camera.recipe()
        plant_kw = {
            "props": dict((loka_config().get("props") or {}).get("slots") or {}),
            "static_props": tuple((loka_config().get("props") or {}).get("static") or ()),
            "camera": camera,
        }
        if plant == "unitree_rl":
            self._sim = UnitreeRLSimulation(
                locomotion_config(suite),
                policy_dir=options.get("policy_dir"),
                min_speed=options.get("min_speed"),
                **plant_kw,
            )
        elif plant == "mpc":
            self._sim = Simulation(locomotion_config(suite), **plant_kw)
        else:
            raise ValueError(f"navila plant must be mpc or unitree_rl, got {plant!r}")
        policy = NavilaPolicy(
            model_path, device=cfg.device, num_frames=cfg.num_frames, max_new_tokens=cfg.max_new_tokens
        )
        if log_path is not None:
            cfg.log_dir = Path(log_path)
        self.runtime = NavilaRuntime(
            self._sim, cfg, plant=plant, policy=policy, enable_camera=True
        )
        self._goal_xy = None
        self._plant = plant

    def reset(self, seed: int, init_noise: float) -> None:
        self._sim.reset()
        apply_seeded_state_noise(self._sim.model, self._sim.data, seed, init_noise)
        self._goal_xy = None

    def step(self) -> None:
        self.runtime.step()

    def command_goal(self, xy, speed: float, radius: float) -> None:
        self._goal_xy = np.asarray(xy, dtype=float)
        _apply_goal(self._sim.controller, xy, speed, radius)

    def submit_directive(self, text: str, xy) -> None:
        self._goal_xy = np.asarray(xy, dtype=float)
        self.runtime.submit_instruction(text)

    def inject(self, fault: FaultSpec) -> None:
        _inject(self._sim, self.runtime.viz, fault)

    def close(self) -> None:
        self.runtime.close()

    @property
    def sim(self) -> Simulation:
        return self._sim

    @property
    def viz(self) -> FaultVizState:
        return self.runtime.viz


def build_baseline(spec, suite, log_path: Path | None = None) -> Baseline:
    return get_baseline(spec.type)(spec, suite, log_path=log_path)
