"""NaVILA navigation VLA baseline for the G1.

Mid-level actions (``move forward d cm``, ``turn left|right theta degrees``,
``stop``) drive the shared ``gait.*`` Task_Targets surface on either the MPC
plant or the unitree_rl policy. Real weights load only from a local directory
(run ``scripts/fetch_navila_checkpoint.py`` on PACE). Script mode needs no
GPU and never imports torch.
"""

from __future__ import annotations

import json
import math
import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from loka.agent.faults import FaultSpec
from loka.agent.interactive import FaultVizState, commit_fault, restore_faults
from loka.control.gait import wrap_angle
from loka.control.unitree_rl import MIN_WALK_SPEED, UnitreeRLController, yaw_of
from loka.vision import EgoCameraSpec, EgoRenderer, encode_png

#: Hugging Face repo id. Used only by ``scripts/fetch_navila_checkpoint.py``.
NAVILA_HUB_ID = "a8cheng/navila-llama3-8b-8f"
FETCH_HINT = (
    "Download on PACE only: "
    "python scripts/fetch_navila_checkpoint.py --out ~/scratch/models/navila-llama3-8b-8f"
)
#: PACE scratch layout used when HF_HOME / XDG_CACHE_HOME are not exported.
SCRATCH_CACHE_DEFAULTS = {
    "HF_HOME": "~/scratch/huggingface",
    "XDG_CACHE_HOME": "~/scratch/.cache",
}

_FORWARD_RE = re.compile(
    r"(?:move\s+)?forward\s+(\d+(?:\.\d+)?)\s*cm",
    re.IGNORECASE,
)
_TURN_RE = re.compile(
    r"turn\s+(left|right)\s+(\d+(?:\.\d+)?)\s*degrees?",
    re.IGNORECASE,
)
_STOP_RE = re.compile(r"\bstop\b", re.IGNORECASE)

DEFAULT_LOG_DIR = Path("logs") / "navila"


@dataclass(frozen=True)
class NavAction:
    """One mid-level navigation command from NaVILA (or a script)."""

    kind: str  # forward | turn_left | turn_right | stop
    value: float = 0.0  # cm for forward, degrees for turns

    def describe(self) -> str:
        if self.kind == "forward":
            return f"move forward {self.value:g} cm"
        if self.kind == "turn_left":
            return f"turn left {self.value:g} degrees"
        if self.kind == "turn_right":
            return f"turn right {self.value:g} degrees"
        return "stop"


@dataclass
class NavilaConfig:
    """Runtime knobs. Defaults match ``navila:`` in ``loka.yaml``."""

    model_path: str = ""
    device: str = "cuda:0"
    num_frames: int = 8
    max_new_tokens: int = 32
    forward_speed: float = 0.25
    turn_tolerance_deg: float = 5.0
    goal_tolerance: float = 0.12
    timeout_factor: float = 3.0
    max_actions: int = 40
    unparseable_fallback: str = "stop"
    min_free_gb: float = 20.0
    min_compute_capability: tuple[int, int] = (8, 0)
    log_dir: Path | None = None
    save_frames: bool = True
    camera: EgoCameraSpec = field(default_factory=lambda: EgoCameraSpec(
        width=512,
        height=512,
        fovy_deg=90.0,
        pitch_down_deg=10.0,
        detail="high",
        save_frames=True,
    ))

    @classmethod
    def from_dict(cls, raw: dict | None) -> "NavilaConfig":
        raw = dict(raw or {})
        cam = EgoCameraSpec.from_config(raw.pop("camera", None) or {})
        fields = {k: raw[k] for k in cls.__dataclass_fields__ if k in raw and k != "camera"}
        if "log_dir" in fields and fields["log_dir"] is not None:
            fields["log_dir"] = Path(fields["log_dir"])
        if "min_compute_capability" in fields:
            cc = fields["min_compute_capability"]
            fields["min_compute_capability"] = (int(cc[0]), int(cc[1]))
        cfg = cls(camera=cam, **fields)
        return cfg


def parse_action(text: str, *, fallback: str = "stop") -> NavAction:
    """Extract a mid-level action from NaVILA (or script) text."""
    cleaned = (text or "").strip()
    # Prefer the clause after the canonical cue when present.
    marker = re.search(r"the\s+next\s+action\s+is\s*:?\s*", cleaned, re.IGNORECASE)
    if marker is not None:
        cleaned = cleaned[marker.end() :]
    # Take the first line that looks like an action.
    for line in cleaned.splitlines():
        line = line.strip().strip("\"'.").strip()
        if not line:
            continue
        m = _FORWARD_RE.search(line)
        if m:
            return NavAction("forward", float(m.group(1)))
        m = _TURN_RE.search(line)
        if m:
            side = m.group(1).lower()
            return NavAction(f"turn_{side}", float(m.group(2)))
        if _STOP_RE.search(line):
            return NavAction("stop")
    # Whole-string fallback search.
    m = _FORWARD_RE.search(text or "")
    if m:
        return NavAction("forward", float(m.group(1)))
    m = _TURN_RE.search(text or "")
    if m:
        return NavAction(f"turn_{m.group(1).lower()}", float(m.group(2)))
    if _STOP_RE.search(text or ""):
        return NavAction("stop")
    fb = (fallback or "stop").strip().lower()
    if fb == "stop":
        return NavAction("stop")
    return parse_action(fb, fallback="stop")


def parse_script(spec: str) -> list[NavAction]:
    """Parse ``move forward 100 cm; turn left 30 degrees; stop``."""
    actions: list[NavAction] = []
    for part in (spec or "").split(";"):
        part = part.strip()
        if not part:
            continue
        actions.append(parse_action(part, fallback="stop"))
    if not actions:
        raise ValueError("navila script is empty")
    return actions


def sample_history(
    frames: list[np.ndarray],
    num_frames: int = 8,
) -> list[np.ndarray]:
    """Build an 8-frame NaVILA window: first + uniform history + current.

    ``frames`` is chronological (oldest first). The last entry is the current
    observation. Earlier frames are the memory bank; the first is always kept,
    and the rest are uniformly sampled so the total is ``num_frames``.
    """
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    if not frames:
        raise ValueError("need at least one frame")
    current = frames[-1]
    if num_frames == 1:
        return [current]
    history = frames[:-1]
    n_hist = num_frames - 1
    if not history:
        return [current] * n_hist + [current]
    if len(history) >= n_hist:
        # Uniform indices over history, forcing the first slot to index 0.
        idxs = [int(round(x)) for x in np.linspace(0, len(history) - 1, n_hist)]
        idxs[0] = 0
        picked = [history[i] for i in idxs]
    else:
        # Pad on the left by repeating the first frame.
        picked = [history[0]] * (n_hist - len(history)) + list(history)
    return list(picked) + [current]


def build_prompt(instruction: str, num_frames: int = 8) -> str:
    """Official NaVILA navigation prompt with ``num_frames`` ``<image>`` tokens."""
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    hist = "<image>\n" * max(0, num_frames - 1)
    return (
        f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
        f'of historical observations {hist}, and current observation <image>\n. Your assigned task is: "{instruction}" '
        f"Analyze this series of images to decide your next action, which could be turning left or right by a specific "
        f"degree, moving forward a certain distance, or stop if the task is completed."
    )


def require_navila_checkpoint(path: str | Path | None) -> Path:
    """Refuse Hub ids and missing dirs. Never triggers a download."""
    if path is None or str(path).strip() == "":
        raise FileNotFoundError(
            f"navila.model_path / --navila-model is empty. {FETCH_HINT}"
        )
    text = str(path).strip()
    if "/" in text and not Path(text).expanduser().exists() and not text.startswith(
        (".", "/", "~")
    ):
        # Looks like an org/name Hub id rather than a filesystem path.
        if re.fullmatch(r"[\w.-]+/[\w.-]+", text):
            raise FileNotFoundError(
                f"Refusing Hugging Face id {text!r}; pass a local directory. {FETCH_HINT}"
            )
    resolved = Path(text).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"NaVILA checkpoint not found at {resolved}. {FETCH_HINT}"
        )
    # Lightweight completeness check: config + at least one weight shard.
    config = resolved / "config.json"
    if not config.is_file():
        raise FileNotFoundError(
            f"{resolved} has no config.json (incomplete download?). {FETCH_HINT}"
        )
    weights = list(resolved.glob("*.safetensors")) + list(resolved.glob("*.bin"))
    if not weights:
        # Some layouts nest weights; accept a non-empty tree with index files.
        if not any(resolved.rglob("*.safetensors")) and not any(resolved.rglob("*.bin")):
            raise FileNotFoundError(
                f"{resolved} has no weight files. {FETCH_HINT}"
            )
    return resolved


def apply_scratch_cache_defaults() -> dict[str, str]:
    """Fill unset HF_HOME / XDG_CACHE_HOME from ``SCRATCH_CACHE_DEFAULTS``.

    Hugging Face reads these when it is first imported, so call this before
    importing torch, transformers, huggingface_hub or llava.
    """
    resolved = {}
    for key, value in SCRATCH_CACHE_DEFAULTS.items():
        if not os.environ.get(key):
            os.environ[key] = str(Path(value).expanduser())
        resolved[key] = os.environ[key]
    return resolved


def prefer_egl_rendering() -> str:
    """Default ``MUJOCO_GL`` to ``egl`` for the render subprocesses.

    Call after :func:`require_navila_gpu` passes. A headless node otherwise
    falls back to OSMesa, which needs ``libOSMesa`` in the env; EGL renders on
    the GPU NaVILA already has. An exported ``MUJOCO_GL`` wins.
    """
    return os.environ.setdefault("MUJOCO_GL", "egl")


def require_navila_gpu(
    device: str = "cuda:0",
    min_free_gb: float = 20.0,
    min_compute_capability: tuple[int, int] = (8, 0),
) -> str:
    """Fail fast when the NaVILA env or GPU is unsuitable."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is not installed. Activate the loka-navila env on a PACE GPU node "
            "(see scripts/setup_navila_env.sh)."
        ) from exc
    try:
        import llava  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "NaVILA's llava package is not installed. Run scripts/setup_navila_env.sh on PACE."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. NaVILA needs an NVIDIA GPU (Ampere or newer). "
            "Use --navila-script on this machine, or run --navila on PACE."
        )
    if not device.startswith("cuda"):
        raise RuntimeError(f"NaVILA device must be a cuda device, got {device!r}")
    index = int(device.split(":")[1]) if ":" in device else 0
    if index < 0 or index >= torch.cuda.device_count():
        raise RuntimeError(
            f"device {device!r} is out of range ({torch.cuda.device_count()} GPU(s))"
        )
    props = torch.cuda.get_device_properties(index)
    major, minor = int(props.major), int(props.minor)
    need_major, need_minor = min_compute_capability
    if (major, minor) < (need_major, need_minor):
        raise RuntimeError(
            f"GPU {props.name} is compute {major}.{minor}; NaVILA needs "
            f">= {need_major}.{need_minor} (Ampere+) for flash-attn 2 / bf16."
        )
    free_b, total_b = torch.cuda.mem_get_info(index)
    free_gb = free_b / (1024**3)
    total_gb = total_b / (1024**3)
    if free_gb < float(min_free_gb):
        raise RuntimeError(
            f"GPU {props.name} has {free_gb:.1f} GB free / {total_gb:.1f} GB total; "
            f"need >= {min_free_gb:g} GB free for NaVILA fp16."
        )
    print(
        f"[navila] GPU {props.name} compute {major}.{minor} "
        f"{free_gb:.1f}/{total_gb:.1f} GB free on {device}"
    )
    return props.name


class NavilaPolicy:
    """Local-checkpoint NaVILA wrapper. Imports torch/llava only on construction."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda:0",
        num_frames: int = 8,
        max_new_tokens: int = 32,
    ) -> None:
        path = require_navila_checkpoint(model_path)
        self.model_path = path
        self.device = device
        self.num_frames = int(num_frames)
        self.max_new_tokens = int(max_new_tokens)
        self._load()

    def _load(self) -> None:
        import torch
        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model

        # The builder branches on the name (lora / mpt / mistral / gemma), so it
        # must be the checkpoint directory name, as in NaVILA-Bench vlm_server.py.
        model_name = get_model_name_from_path(str(self.model_path))
        # A device other than "cuda" maps every module onto that one GPU.
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            str(self.model_path), model_name, None, device=self.device
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self._torch = torch

    def generate(self, frames: list[np.ndarray], instruction: str) -> str:
        """Greedy decode one mid-level action string from an 8-frame window."""
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token
        from PIL import Image

        window = sample_history(frames, self.num_frames)
        images = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in window]
        image_tensor = process_images(images, self.image_processor, self.model.config)
        image_tensor = image_tensor.to(self.device, dtype=self._torch.float16)

        prompt = build_prompt(instruction, self.num_frames)
        conv = conv_templates["llama_3"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        full = conv.get_prompt()
        input_ids = (
            tokenizer_image_token(full, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.device)
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)
        with self._torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=[image_tensor],
                do_sample=False,
                temperature=0.0,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping],
            )
        text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        # Strip the echoed prompt when the decoder returns the full conversation.
        if prompt in text:
            text = text.split(prompt, 1)[-1]
        return text.strip()


@dataclass
class _ActiveAction:
    action: NavAction
    start_xy: np.ndarray
    start_yaw: float
    target_xy: np.ndarray | None
    target_yaw: float | None
    deadline: float
    start_sim: float
    kind: str


class NavActionExecutor:
    """Closed-loop mid-level actions on any controller with ``set_task_targets``."""

    def __init__(
        self,
        controller,
        *,
        plant: str = "mpc",
        forward_speed: float = 0.25,
        turn_tolerance_deg: float = 5.0,
        goal_tolerance: float = 0.12,
        timeout_factor: float = 3.0,
    ) -> None:
        self.controller = controller
        self.plant = plant
        self.forward_speed = float(forward_speed)
        self.turn_tol = math.radians(float(turn_tolerance_deg))
        self.goal_tol = float(goal_tolerance)
        self.timeout_factor = float(timeout_factor)
        self.active: _ActiveAction | None = None
        self.last_outcome: dict | None = None

    def _speed_cap(self) -> float:
        if isinstance(self.controller, UnitreeRLController):
            return float(self.controller.policy.cmd_high[0])
        return 0.30

    def _cruise(self) -> float:
        return float(np.clip(self.forward_speed, 0.05, self._speed_cap()))

    def clear(self) -> None:
        self.active = None
        self.controller.set_task_targets({"gait.mode": "stand", "gait.goal_active": 0.0})

    def start(self, action: NavAction, qpos: np.ndarray, sim_time: float) -> dict:
        """Arm ``action`` from the current pelvis pose."""
        self.last_outcome = None
        xy = np.asarray(qpos[:2], dtype=float)
        yaw = yaw_of(qpos[3:7])
        if action.kind == "stop":
            self.controller.set_task_targets({"gait.mode": "stand", "gait.goal_active": 0.0})
            self.active = None
            outcome = {
                "action": action.describe(),
                "kind": "stop",
                "status": "done",
                "sim_time": float(sim_time),
            }
            self.last_outcome = outcome
            return outcome

        if action.kind == "forward":
            dist = float(action.value) / 100.0
            target = xy + dist * np.array([math.cos(yaw), math.sin(yaw)])
            cruise = self._cruise()
            nominal = max(dist / max(cruise, 1e-3), 1.0)
            self.controller.set_task_targets(
                {
                    "gait.mode": "walk",
                    "gait.speed": cruise,
                    "gait.goal_x": float(target[0]),
                    "gait.goal_y": float(target[1]),
                    "gait.goal_tolerance": self.goal_tol,
                    "gait.goal_active": 1.0,
                }
            )
            self.active = _ActiveAction(
                action=action,
                start_xy=xy.copy(),
                start_yaw=yaw,
                target_xy=target,
                target_yaw=None,
                deadline=float(sim_time) + self.timeout_factor * nominal,
                start_sim=float(sim_time),
                kind="forward",
            )
            return {"action": action.describe(), "kind": "forward", "status": "started"}

        # Turns.
        sign = 1.0 if action.kind == "turn_left" else -1.0
        delta = sign * math.radians(float(action.value))
        target_yaw = wrap_angle(yaw + delta)
        abs_delta = abs(delta)
        if self.plant == "unitree_rl":
            # Arc: walk at min speed while yaw tracks the target heading.
            cruise = max(MIN_WALK_SPEED, self._cruise() * 0.5)
            yaw_rate = 0.2  # unitree_rl v0 ang_vel_z cap
            nominal = max(abs_delta / yaw_rate, 1.0)
            self.controller.set_task_targets(
                {
                    "gait.mode": "walk",
                    "gait.speed": cruise,
                    "gait.heading": float(target_yaw),
                    "gait.goal_active": 0.0,
                    "gait.goal_heading": float(target_yaw),
                }
            )
            kind = "turn_arc"
        else:
            # In-place crawl toward the new heading.
            turn_rate = 0.4
            nominal = max(abs_delta / turn_rate, 1.0)
            self.controller.set_task_targets(
                {
                    "gait.mode": "walk",
                    "gait.speed": 0.0,
                    "gait.heading": float(target_yaw),
                    "gait.goal_active": 0.0,
                }
            )
            kind = "turn_inplace"
        self.active = _ActiveAction(
            action=action,
            start_xy=xy.copy(),
            start_yaw=yaw,
            target_xy=None,
            target_yaw=target_yaw,
            deadline=float(sim_time) + self.timeout_factor * nominal,
            start_sim=float(sim_time),
            kind=kind,
        )
        return {"action": action.describe(), "kind": kind, "status": "started"}

    def poll(self, qpos: np.ndarray, sim_time: float) -> dict | None:
        """Return an outcome dict once when the active action finishes; else None."""
        if self.active is None:
            return None
        act = self.active
        xy = np.asarray(qpos[:2], dtype=float)
        yaw = yaw_of(qpos[3:7])
        done = False
        status = "done"
        if act.kind == "forward" and act.target_xy is not None:
            gait = getattr(self.controller, "gait", None)
            arrived = bool(getattr(gait, "goal_arrived", False))
            rem = float(np.linalg.norm(act.target_xy - xy))
            if arrived or rem <= self.goal_tol:
                done = True
        elif act.target_yaw is not None:
            if abs(wrap_angle(act.target_yaw - yaw)) <= self.turn_tol:
                done = True
        if not done and float(sim_time) >= act.deadline:
            done = True
            status = "timeout"
        if not done:
            return None
        displacement = float(np.linalg.norm(xy - act.start_xy))
        yaw_change = wrap_angle(yaw - act.start_yaw)
        outcome = {
            "action": act.action.describe(),
            "kind": act.kind,
            "status": status,
            "sim_time": float(sim_time),
            "duration_s": float(sim_time) - act.start_sim,
            "displacement_m": displacement,
            "yaw_change_rad": float(yaw_change),
            "yaw_change_deg": float(math.degrees(yaw_change)),
        }
        self.active = None
        self.controller.set_task_targets({"gait.mode": "stand", "gait.goal_active": 0.0})
        self.last_outcome = outcome
        return outcome

    @property
    def busy(self) -> bool:
        return self.active is not None


class NavilaRuntime:
    """Plant + async NaVILA (or script) mid-level action loop."""

    belief_search = None

    def __init__(
        self,
        sim,
        config: NavilaConfig | None = None,
        *,
        plant: str = "mpc",
        policy: Any | None = None,
        script: list[NavAction] | None = None,
        faults: list[FaultSpec] | None = None,
        enable_camera: bool = True,
    ) -> None:
        self.sim = sim
        self.controller = sim.controller
        self.config = config or NavilaConfig()
        self.plant = plant
        self.policy = policy
        self.script = list(script or [])
        self.script_mode = bool(script is not None and policy is None)
        self.faults = list(faults or [])
        self._fired: set[int] = set()
        self.viz = FaultVizState(external_body_id=sim.pelvis_id)
        self.executor = NavActionExecutor(
            self.controller,
            plant=plant,
            forward_speed=self.config.forward_speed,
            turn_tolerance_deg=self.config.turn_tolerance_deg,
            goal_tolerance=self.config.goal_tolerance,
            timeout_factor=self.config.timeout_factor,
        )
        self.instruction: str | None = None
        self._actions_this_instruction = 0
        self._frame_bank: list[np.ndarray] = []
        self._pending: queue.Queue = queue.Queue()
        self._infer_generation = 0
        self._inferring = False
        self._finished = False
        self._awaiting_query = False
        self.log_path = Path(self.config.log_dir or DEFAULT_LOG_DIR)
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.session_path = self.log_path / "session.jsonl"
        self.frames_dir = self.log_path / "frames"
        if self.config.save_frames:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.renderer: EgoRenderer | None = None
        if enable_camera and getattr(sim, "has_camera", False) and not self.script_mode:
            self.renderer = EgoRenderer(sim.recipe)
        # Script mode: queue the scripted actions as if the VLA emitted them.
        self._script_index = 0
        self.history: list[dict] = []

    # -- operator / faults -------------------------------------------------

    def submit_operator(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        if text.lower().startswith("act "):
            action = parse_action(text[4:], fallback=self.config.unparseable_fallback)
            self._cancel_inference()
            self.executor.clear()
            self._finished = False
            self._awaiting_query = False
            self._start_action(action)
            print(f"[navila] act: {action.describe()}")
            return
        self._set_instruction(text)
        print(f"[navila] instruction: {text}")

    def submit_instruction(self, text: str) -> None:
        self.submit_operator(text)

    def _set_instruction(self, text: str) -> None:
        self._cancel_inference()
        self.executor.clear()
        self.instruction = text.strip()
        self._actions_this_instruction = 0
        self._frame_bank.clear()
        self._finished = False
        self._awaiting_query = True
        self._script_index = 0
        if self.script_mode and self.script:
            # Script consumes the list once per instruction reset.
            pass

    def inject_fault_now(self, fault: FaultSpec) -> bool:
        ok, message = commit_fault(self.sim, self.viz, fault)
        print(message)
        return ok

    def clear_interactive_faults(self) -> None:
        print(restore_faults(self.sim, self.viz) or "[FAULT] nothing to clear")

    def close(self) -> None:
        self._cancel_inference()
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    # -- stepping ----------------------------------------------------------

    def step(self) -> dict:
        now = float(self.sim.data.time)
        for index, fault in enumerate(self.faults):
            if index not in self._fired and now >= fault.time:
                self._fired.add(index)
                self.inject_fault_now(fault)

        self._drain_replies()
        outcome = self.executor.poll(self.sim.robot_qpos, now)
        if outcome is not None:
            self._log({"event": "action_done", **outcome})
            if outcome.get("kind") == "stop":
                self._finished = True
                self._awaiting_query = False
            elif self.script_mode and self._script_index >= len(self.script):
                self._finished = True
                self._awaiting_query = False
            elif not self._finished:
                # Ask for the next mid-level action once this one is done.
                self._awaiting_query = True

        if (
            self._awaiting_query
            and not self.executor.busy
            and not self._inferring
            and not self._finished
            and self.instruction is not None
        ):
            self._awaiting_query = False
            self._dispatch_query()

        self.sim.step()
        self.viz.external_force = self.sim._applied_force()
        self.viz.external_body_id = self.sim.pelvis_id
        sample = {"t": self.sim.data.time}
        self.history.append(sample)
        return sample

    def run(self, duration: float, stop_on_fall: bool = True) -> list[dict]:
        while self.sim.data.time < duration:
            self.step()
            if stop_on_fall and self.sim.fell:
                break
        return self.history

    # -- inference ---------------------------------------------------------

    def _cancel_inference(self) -> None:
        self._infer_generation += 1
        self._inferring = False
        # Drain stale replies.
        while True:
            try:
                self._pending.get_nowait()
            except queue.Empty:
                break

    def _dispatch_query(self) -> None:
        if self._actions_this_instruction >= int(self.config.max_actions):
            print(
                f"[navila] max_actions={self.config.max_actions} reached; standing"
            )
            self.executor.clear()
            self._finished = True
            self._log(
                {
                    "event": "max_actions",
                    "sim_time": float(self.sim.data.time),
                    "instruction": self.instruction,
                }
            )
            return

        if self.script_mode:
            if self._script_index >= len(self.script):
                self._finished = True
                return
            action = self.script[self._script_index]
            self._script_index += 1
            self._log(
                {
                    "event": "script_action",
                    "sim_time": float(self.sim.data.time),
                    "instruction": self.instruction,
                    "raw": action.describe(),
                    "parsed": action.describe(),
                }
            )
            self._start_action(action)
            return

        if self.policy is None:
            raise RuntimeError("NaVILA policy is not loaded; pass --navila-model or --navila-script")

        frame = self._capture_frame()
        self._frame_bank.append(frame)
        frames = list(self._frame_bank)
        instruction = self.instruction or ""
        generation = self._infer_generation
        dispatch_sim = float(self.sim.data.time)
        dispatch_wall = time.perf_counter()
        self._inferring = True
        self.executor.clear()  # hold stand while waiting

        def worker() -> None:
            try:
                t0 = time.perf_counter()
                raw = self.policy.generate(frames, instruction)
                wall_ms = (time.perf_counter() - t0) * 1e3
                self._pending.put(
                    {
                        "generation": generation,
                        "raw": raw,
                        "wall_ms": wall_ms,
                        "dispatch_sim": dispatch_sim,
                        "dispatch_wall": dispatch_wall,
                        "frame": frame,
                        "ok": True,
                    }
                )
            except Exception as exc:  # noqa: BLE001 — surface to the control thread
                self._pending.put(
                    {
                        "generation": generation,
                        "ok": False,
                        "error": str(exc),
                        "dispatch_sim": dispatch_sim,
                        "dispatch_wall": dispatch_wall,
                    }
                )

        threading.Thread(target=worker, daemon=True).start()
        print(f"[navila] querying VLA at t={dispatch_sim:.2f}s …")

    def _drain_replies(self) -> None:
        try:
            reply = self._pending.get_nowait()
        except queue.Empty:
            return
        if int(reply.get("generation", -1)) != self._infer_generation:
            return
        self._inferring = False
        if not reply.get("ok", False):
            print(f"[navila] inference failed: {reply.get('error')}")
            self._log(
                {
                    "event": "inference_error",
                    "sim_time": float(self.sim.data.time),
                    "error": reply.get("error"),
                }
            )
            self._finished = True
            return
        raw = str(reply["raw"])
        action = parse_action(raw, fallback=self.config.unparseable_fallback)
        apply_sim = float(self.sim.data.time)
        wall_latency = time.perf_counter() - float(reply["dispatch_wall"])
        sim_latency = apply_sim - float(reply["dispatch_sim"])
        frame_path = None
        if self.config.save_frames and reply.get("frame") is not None:
            frame_path = self._save_frame(reply["frame"], apply_sim)
        self._log(
            {
                "event": "inference",
                "sim_time": apply_sim,
                "instruction": self.instruction,
                "raw": raw,
                "parsed": action.describe(),
                "kind": action.kind,
                "value": action.value,
                "inference_ms": float(reply["wall_ms"]),
                "wall_latency_s": wall_latency,
                "sim_latency_s": sim_latency,
                "frame": str(frame_path) if frame_path else None,
            }
        )
        print(f"[navila] t={apply_sim:.2f}s -> {action.describe()!r} ({raw!r})")
        self._start_action(action)

    def _start_action(self, action: NavAction) -> None:
        self._actions_this_instruction += 1
        info = self.executor.start(action, self.sim.robot_qpos, float(self.sim.data.time))
        self._log({"event": "action_start", "sim_time": float(self.sim.data.time), **info})
        if action.kind == "stop":
            self._finished = True

    def _capture_frame(self) -> np.ndarray:
        if self.renderer is None:
            # Placeholder so script/stub tests can run without a camera.
            cam = self.config.camera
            return np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
        return self.renderer.render_rgb(self.sim.snapshot())

    def _save_frame(self, rgb: np.ndarray, sim_time: float) -> Path:
        path = self.frames_dir / f"t{sim_time:07.2f}_{datetime.now():%H%M%S_%f}.png"
        path.write_bytes(encode_png(rgb))
        return path

    def _log(self, row: dict) -> None:
        row = dict(row)
        row.setdefault("wall_time", datetime.now().isoformat(timespec="seconds"))
        with self.session_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
