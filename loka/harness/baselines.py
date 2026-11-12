"""Evaluation baselines.

``low_level`` is the MPC + whole-body controller with no orchestrator.
``loka`` is that same controller with the language orchestrator in the loop.
``loka_ablated`` is LOKA with components switched off (``loka_ablations.yaml``).
``unitree_rl`` is the unitree_rl_lab velocity policy on the same plant.
``navila`` is the NaVILA mid-level navigation VLA on the MPC or unitree_rl plant.

A new baseline subclasses :class:`Baseline`, registers a name, and is
selected from the suite YAML as ``{name, type, options}``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from loka.agent.ablations import AblationSet
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

    def walk_speed(self, speed: float) -> float:
        """Cruise speed of a goal commanded at ``speed``, after any plant floor [m/s]."""
        floor = getattr(self.sim.controller, "walk_speed", None)
        return float(speed) if floor is None else float(floor(speed))

    def holding_stop(self) -> bool:
        """True while the agent stands by its own choice: its gait is not walking."""
        gait = getattr(self.sim.controller, "gait", None)
        return gait is not None and not gait.wants_walk()

    def declared_done(self) -> bool:
        """True once the agent has ended its own run and will not move again."""
        return False

    def model_invocations(self) -> int:
        """Calls sent to the agent's model, answered or not. 0 without one."""
        return 0

    def camera_frames(self) -> int:
        """Answered model calls that carried a rendered camera frame."""
        return sum(
            1 for turn in self.turns if turn.get("frame_path") or turn.get("camera_frame")
        )

    #: Props placed after every reset, ``[(kind, x, y, yaw), ...]``.
    _spawns: tuple = ()

    def _reset_plant(self, seed: int, init_noise: float) -> None:
        """Keyframe reset, robot-only seeded noise, then the scene's props."""
        sim = self.sim
        sim.reset()
        apply_seeded_state_noise(
            sim.model, sim.data, seed, init_noise, nq=sim.robot_nq, nv=sim.robot_nv
        )
        for kind, x, y, yaw in self._spawns:
            print(sim.props.spawn(kind, x, y, yaw))


def _prop_plant(props) -> dict:
    """Plant keyword arguments for a scene's props: the ``loka.yaml`` slot pool."""
    if not props:
        return {}
    from loka.agent.loka_config import loka_config

    pool = loka_config().get("props") or {}
    return {
        "props": dict(pool.get("slots") or {}),
        "static_props": tuple(pool.get("static") or ()),
    }


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


def _terrain_belief_option(spec) -> str:
    options = (getattr(spec, "options", None) or {}) if spec is not None else {}
    mode = str(options.get("terrain_belief", "empty")).strip().lower()
    if mode not in ("empty", "oracle"):
        raise ValueError(
            f"baseline {getattr(spec, 'name', '?')!r}: terrain_belief must be empty or oracle, got {mode!r}"
        )
    return mode


@register_baseline("low_level")
class LowLevelBaseline(Baseline):
    """MPC + WBC only. The effort observer is not in this loop, so a model
    error cannot force the gait to stand.

    Options: ``terrain_belief`` (``empty``, the flat-floor controller, or
    ``oracle``, the plant's own stairs, for the controller-only bound).
    """

    def __init__(self, spec, suite, log_path: Path | None = None, terrain=None, props=None) -> None:
        del log_path
        self._spawns = tuple(props or ())
        self._sim = Simulation(locomotion_config(suite), terrain=terrain, **_prop_plant(props))
        self._terrain_belief = _terrain_belief_option(spec)
        self._viz = FaultVizState(external_body_id=self._sim.pelvis_id)
        self._goal_xy = None
        self._seed_terrain()

    def _seed_terrain(self) -> None:
        if self._terrain_belief == "oracle":
            self._sim.controller.terrain.set_volumes(self._sim.plant_terrain.volumes)

    def reset(self, seed: int, init_noise: float) -> None:
        self._reset_plant(seed, init_noise)
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

    def __init__(self, spec, suite, log_path: Path | None = None, terrain=None, props=None) -> None:
        del log_path
        options = dict(spec.options or {})
        self._spawns = tuple(props or ())
        self._sim = UnitreeRLSimulation(
            locomotion_config(suite),
            policy_dir=options.get("policy_dir"),
            min_speed=options.get("min_speed"),
            terrain=terrain,
            **_prop_plant(props),
        )
        self._viz = FaultVizState(external_body_id=self._sim.pelvis_id)
        self._goal_xy = None


def loka_runtime_config(spec, suite, log_path: Path | None = None) -> LokaConfig:
    """``LokaConfig`` for a ``loka`` or ``loka_ablated`` baseline spec.

    Options: ``orchestrator`` overrides the suite orchestrator (model,
    provider, latency, prompt_style, reply_check, max_tokens); ``ablations``
    (``loka_ablated`` only) is the resolved variant.
    """
    from loka.agent.model_registry import resolve_model

    options = dict(spec.options or {})
    orch = suite.orchestrator_for(spec)
    entry = dict(options.get("model_entry") or resolve_model(orch.model, provider=orch.provider))
    if orch.max_tokens is not None:
        entry["max_tokens"] = int(orch.max_tokens)
    if orch.temperature is not None:
        entry["temperature"] = float(orch.temperature)
    ablations = AblationSet.from_dict(options.get("ablations")) if spec.type == "loka_ablated" else AblationSet()
    extra = {}
    if "terrain_belief" in options:
        extra["terrain_belief"] = _terrain_belief_option(spec)
    if "terrain_refine" in options:
        extra["terrain_refine"] = bool(options["terrain_refine"])
    if "vision" in options:
        extra["enable_vision"] = bool(options["vision"])
    return LokaConfig(
        **extra,
        enable_llm=True,
        llm_model=str(entry["model"]),
        llm_provider=str(entry["provider"]),
        llm_options=entry,
        llm_latency_mode=orch.latency_mode,
        llm_latency_s=orch.latency_s,
        llm_request_timeout_s=orch.request_timeout_s,
        prompt_style=orch.prompt_style,
        reply_check=orch.reply_check,
        ablations=ablations,
        print_compressor=False,
        enable_session_log=log_path is not None,
        log_path=log_path,
    )


@register_baseline("loka")
class LokaBaseline(Baseline):
    """The full LOKA loop: the same plant, plus the orchestrator.

    ``options.orchestrator`` picks the model per baseline, e.g.
    ``{model: claude-sonnet-4-5}`` or a local key from
    ``loka/config/orchestrator_models.yaml``. Terrain options:
    ``terrain_belief`` (``empty`` or ``oracle``), ``terrain_refine`` (false
    is the no-refine ablation) and ``vision`` (the head camera on every turn).
    """

    supports_directive = True

    def __init__(self, spec, suite, log_path: Path | None = None, terrain=None, props=None) -> None:
        config = loka_runtime_config(spec, suite, log_path)
        camera = config.vision.recipe() if config.enable_vision else None
        self._spawns = tuple(props or ())
        self._sim = Simulation(
            locomotion_config(suite), terrain=terrain, camera=camera, **_prop_plant(props)
        )
        self.runtime = LokaRuntime(self._sim, config)
        self.runtime.effort_observer.printout = False
        self._goal_xy = None

    def reset(self, seed: int, init_noise: float) -> None:
        # Belief, sessions and an in-flight LLM call do not have a safe reset.
        # Callers build a new baseline per episode instead of reusing one.
        self._reset_plant(seed, init_noise)
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

    def model_invocations(self) -> int:
        return int(self.runtime.llm_invocations)


@register_baseline("loka_ablated")
class LokaAblatedBaseline(LokaBaseline):
    """LOKA with components switched off by a variant of ``loka_ablations.yaml``.

    Options: ``variant`` (a name under ``variants``), resolved at config parse
    time into ``ablations``; ``orchestrator`` as for ``loka``.
    """


@register_baseline("navila")
class NavilaBaseline(Baseline):
    """NaVILA mid-level navigation VLA on the MPC or unitree_rl plant.

    Options: ``plant`` (``mpc`` | ``unitree_rl``), ``model_path`` (local
    checkpoint directory; required for real inference), ``device``,
    ``forward_speed`` and the other ``navila:`` keys of ``loka.yaml``,
    ``policy_dir`` / ``min_speed`` when ``plant`` is ``unitree_rl``.
    ``latency`` (``{mode, seconds, request_timeout_s}``) defaults to the suite
    orchestrator's, so NaVILA and LOKA pay the same simulated latency per call.
    ``script`` (``"move forward 100 cm; stop"``) replaces the model with fixed
    actions and needs no checkpoint or GPU (harness smoke tests only).
    """

    supports_directive = True

    def __init__(self, spec, suite, log_path: Path | None = None, terrain=None, props=None) -> None:
        from loka.agent.loka_config import loka_config
        from loka.vla.navila import (
            NavilaConfig,
            NavilaPolicy,
            NavilaRuntime,
            apply_scratch_cache_defaults,
            parse_script,
            prefer_egl_rendering,
            require_navila_checkpoint,
            require_navila_gpu,
        )

        options = dict(spec.options or {})
        plant = str(options.get("plant", "mpc")).strip().lower()
        settings = (loka_config().get("navila") or {}).copy()
        reserved = ("plant", "policy_dir", "min_speed", "latency", "script")
        settings.update({k: v for k, v in options.items() if k not in reserved})
        if options.get("model_path"):
            settings["model_path"] = options["model_path"]
        latency = dict(options.get("latency") or {})
        settings["latency_mode"] = str(latency.get("mode", suite.orchestrator.latency_mode))
        settings["latency_s"] = float(latency.get("seconds", suite.orchestrator.latency_s))
        settings["request_timeout_s"] = float(
            latency.get("request_timeout_s", suite.orchestrator.request_timeout_s)
        )
        cfg = NavilaConfig.from_dict(settings)
        script = parse_script(str(options["script"])) if options.get("script") else None
        if script is None:
            model_path = require_navila_checkpoint(cfg.model_path)
            apply_scratch_cache_defaults()
            require_navila_gpu(cfg.device, cfg.min_free_gb, cfg.min_compute_capability)
            prefer_egl_rendering()
        self._spawns = tuple(props or ())
        camera = cfg.camera.recipe()
        plant_kw = {
            "props": dict((loka_config().get("props") or {}).get("slots") or {}),
            "static_props": tuple((loka_config().get("props") or {}).get("static") or ()),
            "camera": camera,
            "terrain": terrain,
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
        policy = None
        if script is None:
            policy = NavilaPolicy(
                model_path, device=cfg.device, num_frames=cfg.num_frames, max_new_tokens=cfg.max_new_tokens
            )
        if log_path is not None:
            cfg.log_dir = Path(log_path).parent / "navila"
        self.runtime = NavilaRuntime(
            self._sim, cfg, plant=plant, policy=policy, script=script, enable_camera=True
        )
        self._goal_xy = None
        self._plant = plant

    def reset(self, seed: int, init_noise: float) -> None:
        self._reset_plant(seed, init_noise)
        self._goal_xy = None

    def holding_stop(self) -> bool:
        """NaVILA stands between actions while it waits for a reply; only ``stop`` counts."""
        return bool(self.runtime.stopped) and not self.runtime.executor.busy

    def declared_done(self) -> bool:
        """``stop``, ``max_actions`` or a failed query: NaVILA takes no further action."""
        return bool(self.runtime._finished) and not self.runtime.executor.busy

    def model_invocations(self) -> int:
        return int(self.runtime.invocations)

    @property
    def turns(self) -> list[dict]:
        return self.runtime.turns

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


def build_baseline(spec, suite, log_path: Path | None = None, terrain=None, props=None) -> Baseline:
    """One baseline on a plant with ``terrain`` (stairs specs) built in, if any,
    and ``props`` (``[(kind, x, y, yaw), ...]``) placed after every reset."""
    cls = get_baseline(spec.type)
    kwargs = {}
    if terrain:
        kwargs["terrain"] = terrain
    if props:
        kwargs["props"] = props
    return cls(spec, suite, log_path=log_path, **kwargs)
