#!/usr/bin/env python3
"""The G1 locomotion plant with the LOKA orchestrator attached.

    python -m loka.run_loka
    python -m loka.run_loka --mode walk --speed 0.3
    python -m loka.run_loka --operator "lean left and crouch"
    python -m loka.run_loka --no-llm --goal 3,3
    python -m loka.run_loka --fault mass --headless -T 12
    python -m loka.run_loka --unitree-rl --goal 3,3 --goal-heading 1.57
    python -m loka.run_loka --vision --props "chair:1.5,0.2,3.14;box:1,-0.4"
    python -m loka.run_loka --no-llm --ego-view --props "table:2,0"
    python -m loka.run_loka --record --goal 3,1        # recordings/run_loka_<stamp>.mp4
    python -m loka.run_loka --scene obstacle_course --vision --ego-view --record
    python -m loka.run_loka --navila --navila-script "move forward 100 cm; stop" --scene obstacle_course
    python -m loka.run_loka --navila --navila-model ~/scratch/models/navila-llama3-8b-8f \\
        --headless -T 60 --scene obstacle_course --record --operator "walk to the chair"

``--operator`` is free-text language and requires the LLM (``OPENAI_API_KEY``),
except with ``--navila`` where it is the navigation instruction.
``--orchestrator-mllama PATH`` runs the orchestrator on a local
Llama-3.2-11B-Vision-Instruct checkpoint instead of OpenAI. Its language model is
Llama-3.1-8B, the same family and size as NaVILA's, and its 128K context holds
LOKA's full prompt. Needs the ``loka-mllama`` env (``scripts/setup_mllama_env.sh``)
on a GPU node; the repo is gated, see ``scripts/fetch_navila_checkpoint.py``.
Both default to ``--prompt-style compact``: the same rules and reply schema in
about 2-3K tokens a turn instead of ~20K. ``--prompt-style full`` sends the
prompt LOKA is evaluated with. They also default to ``--describe-first`` (with
``--vision``: the model first describes the frame, each object with its
direction and a rough distance, and that description is added to the turn as
written), ``--reply-check`` (a reply that copies the format's placeholders,
echoes the prompt, is empty, or sends a walk goal at the robot's own position
is asked again once) and ``--max-memory 0`` (no earlier turns in view). All
four flags also work with OpenAI, e.g. ``--prompt-style compact
--describe-first --reply-check --max-memory 0`` to test GPT under the
local-model setup.
``--orchestrator-vila PATH`` does the same with Llama-3-VILA1.5-8B (NaVILA's
base model, ``loka-navila`` env), but VILA reads only 4,096 tokens, so it sees
the start of the system prompt and nothing after it.
``--vision`` attaches a 512x512 head-camera frame to every orchestrator turn
(settings in ``loka/config/loka.yaml`` under ``vision``). The scene carries a
pool of HumanoidBench props (chair, table, box, cube, shelf) parked off-scene;
``--props`` places some at startup and the ``spawn`` / ``move`` / ``remove``
commands place them while running. ``--scene NAME`` places a saved layout
from ``loka/config/scenes/`` (``scene save NAME`` on stdin writes one).
``--no-props`` builds the plain scene.
``--ego-view`` opens a second window streaming that head camera live (rate
from ``vision.stream_hz``); it needs no LLM.
``--record [PATH]`` saves the whole session as an mp4 from a fixed overview
camera (``recording`` in ``loka.yaml``), headless or with the viewer.
``--unitree-rl`` swaps the MPC plant for the unitree_rl_lab velocity policy on
the same scene. It has no orchestrator, so it implies ``--no-llm``; goals,
``--mode walk``, faults and the fault keys work as they do for the plant.
``--navila`` runs the NaVILA navigation VLA (mid-level actions) on the MPC
plant by default, or on unitree_rl with ``--unitree-rl``. Weights load only
from a local directory (``--navila-model`` / ``navila.model_path``); download
them on PACE with ``scripts/fetch_navila_checkpoint.py``. ``--navila-script``
replays mid-level actions with no GPU and no checkpoint. Real ``--navila``
needs the ``loka-navila`` env (``scripts/setup_navila_env.sh``) on a GPU node.
``--no-llm`` accepts a world goal (``--goal 3,3`` or ``goal 3 3`` on
stdin) and structured ``--task`` setpoints. The control loop never blocks on
the model.

To drive the same plant by hand, without any language in the loop, use
``python -m loka.dashboard`` instead.

Walk geometry (the footstep plan, DCM reference and swing arcs) is classical;
the LLM only sets high-level ``gait.*`` Task_Targets (mode, speed, heading,
cadence, …).

The effort observer prints joint torque bias and per-component state cost
on one updating line. The orchestrator runs unless you pass ``--no-llm``.

Session logs rewrite ``logs/loka/session.log`` (+ ``.jsonl``) each run.

Interactive commands (stdin, with the viewer or ``--headless`` when stdin is a
terminal) and viewer keys:
  mass / friction|ice / push / dead / scale|weak / clear / help
  spawn <kind> x y [yaw] | move <prop> x y [yaw] | remove <prop> | props | clear props
  search mass [body] | search com <x|y|z> [body] | search friction | search gear <actuator> | search stop
  Keys: 2 right knee at 75% torque, 3 push, 4 ice,
        5 left-shoulder +5 kg, 6 torso +5 kg,
        7 right-shoulder +5 kg, 8 clear, H help,
        M torso-mass search, X/Y/Z torso COM search
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import mujoco

from loka.control.locomotion import DEFAULT_MODEL, LocomotionConfig, TASK_PARAMETER_NAMES
from loka.control.stacks import add_stack_argument, apply_stack_arg
from loka.sim import Simulation, UnitreeRLSimulation
from loka.agent.faults import FaultSpec
from loka.agent.belief_search import parse_search_command, search_for_keycode
from loka.agent.interactive import (
    FAULT_HELP,
    FaultVizState,
    KeyDebounce,
    commit_fault,
    fault_for_keycode,
    parse_fault_command,
    parse_goal_command,
    parse_prop_command,
    restore_faults,
)
from loka.agent.loka_config import loka_config
from loka.agent.runtime import LokaConfig, LokaRuntime
from loka.agent.session_log import default_log_path
from loka.recording import OverviewCamera, SessionRecorder
from loka.scene_props import load_scene, parse_props_spec
from loka.vision import EGO_CAMERA, EgoCameraSpec, EgoViewStream
from loka.viz import begin_frame, render_fault_overlays, render_gait_overlays


def parse_task_spec(spec: str) -> dict[str, float]:
    """Parse ``height=0.60,lean_x=0.03`` into Task_Targets keys."""

    def _is_float_token(token: str) -> bool:
        try:
            float(token)
            return True
        except ValueError:
            return False

    updates: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(
                f"Bad --task fragment {part!r}; expected name=value"
            )
        name, raw = part.split("=", 1)
        name = name.strip()
        if name not in TASK_PARAMETER_NAMES and name not in (
            "com_offset_x",
            "com_offset_y",
        ):
            raise argparse.ArgumentTypeError(
                f"Unknown task setpoint {name!r}; "
                f"allowed: {sorted(TASK_PARAMETER_NAMES)}"
            )
        if name == "gait.mode" and not _is_float_token(raw):
            updates[name] = raw.strip()
            continue
        try:
            updates[name] = float(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Non-numeric value for {name}: {raw!r}"
            ) from exc
    if not updates:
        raise argparse.ArgumentTypeError("--task was empty")
    return updates


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("-T", "--duration", type=float, default=None)
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Ablation: plant only (no language; use --task for structured setpoints)",
    )
    parser.add_argument(
        "--operator",
        type=str,
        default=None,
        help="Free-text operator request (requires LLM; queued at --operator-at)",
    )
    parser.add_argument(
        "--task",
        type=parse_task_spec,
        default=None,
        help="Structured Task_Targets for ablations, e.g. height=0.60,lean_x=0.03",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="World goal x,y the plant walks to. Works with --no-llm. Example: 3,3",
    )
    parser.add_argument("--operator-at", type=float, default=3.0)
    parser.add_argument(
        "--fault",
        choices=("none", "mass", "friction", "actuator_dead", "push"),
        default="none",
    )
    parser.add_argument("--fault-at", type=float, default=4.0)
    parser.add_argument("--fault-mu", type=float, default=0.25)
    parser.add_argument("--fault-mass", type=float, default=5.0)
    parser.add_argument("--fault-actuator", type=str, default="right_knee")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Session log directory (default: logs/loka; rewritten each run)",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable writing compressor/LLM session logs",
    )
    parser.add_argument(
        "--quiet-compressor",
        action="store_true",
        help="Do not print the full compressor user-turn to stdout (still logged)",
    )
    parser.add_argument(
        "--mode",
        choices=("stand", "walk"),
        default="stand",
        help="Initial locomotion mode (walk engages gait clock + CP footsteps)",
    )
    parser.add_argument("--speed", type=float, default=0.25, help="Initial gait.speed if --mode walk")
    parser.add_argument(
        "--heading",
        type=float,
        default=0.0,
        help="Initial gait.heading [rad] world-frame travel direction",
    )
    parser.add_argument(
        "--unitree-rl",
        action="store_true",
        help="Run the unitree_rl_lab velocity policy instead of the MPC plant (implies --no-llm)",
    )
    parser.add_argument(
        "--policy-dir",
        type=Path,
        default=None,
        help="unitree_rl policy directory with deploy.yaml and policy.npz "
        "(default: models/unitree_rl/g1_29dof_velocity_v0)",
    )
    parser.add_argument(
        "--goal-heading",
        type=float,
        default=None,
        help="World yaw [rad] to face while walking to --goal and at it. --unitree-rl only",
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Attach a head-camera image to every orchestrator turn (needs the LLM). "
        "Also on when vision.enabled is true in loka.yaml",
    )
    parser.add_argument(
        "--ego-view",
        action="store_true",
        help="Open a live window streaming the head camera (ego_cam) next to the "
        "viewer. Works with --no-llm and --unitree-rl; not with --headless",
    )
    parser.add_argument(
        "--record",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="Record the session to an mp4 from a fixed overview camera (+x/-y, "
        "looking along +y). Default path recordings/run_loka_<timestamp>.mp4; "
        "camera in loka.yaml under recording. Encoded when the run ends",
    )
    parser.add_argument(
        "--props",
        type=str,
        default=None,
        help='Props placed at startup, "kind:x,y[,yaw];..." e.g. "chair:2,0.5,1.57;box:1.5,-0.3"',
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        metavar="NAME",
        help="Place a saved prop layout at startup: a name in loka/config/scenes/ "
        "(e.g. obstacle_course) or a path to a scene .yaml. Combines with --props",
    )
    parser.add_argument(
        "--no-props",
        action="store_true",
        help="Build the plain scene without the prop pool",
    )
    parser.add_argument(
        "--navila",
        action="store_true",
        help="Run the NaVILA navigation VLA (implies --no-llm). Needs --navila-model "
        "or --navila-script. Plant is MPC unless --unitree-rl is also set",
    )
    parser.add_argument(
        "--navila-model",
        type=Path,
        default=None,
        help="Local NaVILA checkpoint directory (never a Hub id). "
        "Download on PACE: python scripts/fetch_navila_checkpoint.py --out …",
    )
    parser.add_argument(
        "--navila-device",
        type=str,
        default=None,
        help="CUDA device for NaVILA (default: navila.device in loka.yaml, else cuda:0)",
    )
    parser.add_argument(
        "--navila-script",
        type=str,
        default=None,
        metavar="ACTIONS",
        help='Replay mid-level actions with no GPU, e.g. "move forward 100 cm; turn left 30 degrees; stop"',
    )
    parser.add_argument(
        "--orchestrator-mllama",
        type=Path,
        default=None,
        metavar="PATH",
        help="Run LOKA's orchestrator on a local Llama-3.2-11B-Vision-Instruct checkpoint "
        "instead of OpenAI. Needs the loka-mllama env on a GPU node",
    )
    parser.add_argument(
        "--orchestrator-vila",
        type=Path,
        default=None,
        metavar="PATH",
        help="Run LOKA's orchestrator on a local Llama-3-VILA1.5-8B checkpoint (NaVILA's "
        "base model) instead of OpenAI. Needs the loka-navila env on a GPU node",
    )
    parser.add_argument(
        "--describe-first",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Before each turn with a camera frame, have the model describe the frame "
        "(objects with direction and rough distance) and add that description to the "
        "turn as written. Needs --vision. Default: on with --orchestrator-mllama / "
        "--orchestrator-vila, off otherwise (works with OpenAI too)",
    )
    parser.add_argument(
        "--reply-check",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Ask the model again once when its reply copies the format's <...> placeholders, "
        "echoes the prompt, is empty, or sends a walk goal where the robot already stands. "
        "Default: on with --orchestrator-mllama / --orchestrator-vila, off otherwise",
    )
    parser.add_argument(
        "--max-memory",
        type=int,
        default=None,
        metavar="N",
        help="Completed orchestrator turns kept in view (session.max_memory). Default: 0 "
        "with --orchestrator-mllama / --orchestrator-vila (small models copy their own "
        "earlier replies), loka.yaml otherwise",
    )
    parser.add_argument(
        "--prompt-style",
        choices=("full", "compact"),
        default=None,
        help="Orchestrator prompt. full is the evaluated LOKA prompt (~20K tokens a turn); "
        "compact keeps the rules and reply schema in ~2-3K tokens. Default: compact with "
        "--orchestrator-mllama / --orchestrator-vila, full otherwise",
    )
    parser.add_argument(
        "--orchestrator-device",
        type=str,
        default="cuda:0",
        help="Device for --orchestrator-mllama / --orchestrator-vila (default: cuda:0). "
        "cpu skips the GPU check; --orchestrator-mllama only, for tiny test checkpoints",
    )
    add_stack_argument(parser)
    args = parser.parse_args(argv)
    local = [flag for flag, value in (
        ("--orchestrator-mllama", args.orchestrator_mllama),
        ("--orchestrator-vila", args.orchestrator_vila),
    ) if value is not None]
    if len(local) > 1:
        parser.error("pass only one of --orchestrator-mllama / --orchestrator-vila")
    if local and (args.no_llm or args.navila or args.unitree_rl):
        parser.error(f"{local[0]} replaces the LLM; drop --no-llm / --navila / --unitree-rl")
    if args.orchestrator_vila is not None and not args.orchestrator_device.startswith("cuda"):
        parser.error("--orchestrator-vila needs a cuda --orchestrator-device")
    if args.prompt_style is None:
        args.prompt_style = "compact" if local else "full"
    if args.describe_first is None:
        args.describe_first = bool(local)
    if args.reply_check is None:
        args.reply_check = bool(local)
    if args.max_memory is None:
        args.max_memory = 0 if local else None
    elif args.max_memory < 0:
        parser.error("--max-memory must be >= 0")
    if args.vision and (args.no_llm or args.unitree_rl) and not args.navila:
        parser.error("--vision feeds the orchestrator; drop --no-llm / --unitree-rl")
    if args.ego_view and args.headless:
        parser.error("--ego-view opens a window; drop --headless")
    args.prop_spawns = []
    if args.no_props and (args.props is not None or args.scene is not None):
        parser.error("--no-props removes the prop pool; drop --props / --scene")
    if args.scene is not None:
        try:
            args.prop_spawns += load_scene(args.scene)
        except ValueError as exc:
            parser.error(f"--scene: {exc}")
    if args.props is not None:
        try:
            args.prop_spawns += parse_props_spec(args.props)
        except ValueError as exc:
            parser.error(f"--props: {exc}")
    if args.navila_script is not None or args.navila_model is not None or args.navila_device is not None:
        if not args.navila:
            parser.error("--navila-model / --navila-script / --navila-device need --navila")
    if args.navila:
        args.no_llm = True
        if args.task is not None:
            parser.error("--navila does not take --task; use --operator or --navila-script")
        if args.navila_script is None and args.navila_model is None:
            # model_path may still come from loka.yaml; checked in main.
            pass
        if args.vision:
            parser.error("--navila has its own camera; drop --vision")
    if args.unitree_rl:
        if (args.operator or args.task) and not args.navila:
            parser.error("--unitree-rl has no orchestrator; drop --operator / --task")
        args.no_llm = True
    elif args.policy_dir is not None or args.goal_heading is not None:
        parser.error("--policy-dir and --goal-heading need --unitree-rl")
    if args.operator and args.no_llm and not args.navila:
        parser.error(
            "--operator is language and requires the LLM; "
            "drop --no-llm, use --navila, or use --goal 3,3 / --task for the plant"
        )
    if args.operator and args.goal:
        parser.error("pass only one of --operator or --goal")
    if args.goal is not None:
        parsed_goal = parse_goal_command(f"goal {args.goal}")
        if parsed_goal is None:
            parser.error(f"--goal must be x,y, got {args.goal!r}")
        args.goal_xy = (parsed_goal[0], parsed_goal[1])
        args.goal_speed = parsed_goal[2]
    else:
        args.goal_xy = None
        args.goal_speed = None
    if args.operator and args.task:
        parser.error("pass only one of --operator or --task")
    return args


def build_faults(args: argparse.Namespace) -> list[FaultSpec]:
    if args.fault == "none":
        return []
    if args.fault == "mass":
        return [
            FaultSpec(
                "mass",
                args.fault_at,
                {"delta_kg": args.fault_mass, "body": "torso_link"},
            )
        ]
    if args.fault == "friction":
        return [FaultSpec("friction", args.fault_at, {"mu": args.fault_mu})]
    if args.fault == "actuator_dead":
        return [
            FaultSpec(
                "actuator_dead",
                args.fault_at,
                {"actuator": args.fault_actuator},
            )
        ]
    if args.fault == "push":
        return [
            FaultSpec(
                "push",
                args.fault_at,
                {"impulse": 6.0, "direction": (1.0, 0.0, 0.0)},
            )
        ]
    return []


class _PlantRuntime:
    """Faults and stepping for a plant with no orchestrator (``--unitree-rl``)."""

    belief_search = None

    def __init__(self, sim: Simulation, faults: list[FaultSpec]) -> None:
        self.sim = sim
        self.controller = sim.controller
        self.viz = FaultVizState(external_body_id=sim.pelvis_id)
        self.faults = list(faults)
        self._fired: set[int] = set()

    def inject_fault_now(self, fault: FaultSpec) -> bool:
        ok, message = commit_fault(self.sim, self.viz, fault)
        print(message)
        return ok

    def clear_interactive_faults(self) -> None:
        print(restore_faults(self.sim, self.viz) or "[FAULT] nothing to clear")

    def submit_operator(self, text: str) -> None:
        print(f"[!] plant has no language path; ignored: {text!r}")

    def step(self) -> None:
        now = float(self.sim.data.time)
        for index, fault in enumerate(self.faults):
            if index not in self._fired and now >= fault.time:
                self._fired.add(index)
                self.inject_fault_now(fault)
        self.sim.step()
        self.viz.external_force = self.sim._applied_force()

    def run(self, duration: float, stop_on_fall: bool = True) -> list[dict]:
        while self.sim.data.time < duration:
            self.step()
            if stop_on_fall and self.sim.fell:
                break
        return self.sim.history


def _handle_interactive_line(
    runtime: LokaRuntime, line: str, *, llm_enabled: bool, navila: bool = False
) -> None:
    prop_cmd = parse_prop_command(line)
    if prop_cmd is not None:
        props = runtime.sim.props
        if props is None:
            print("[!] no prop pool in this scene (started with --no-props)")
        else:
            # Applied by the simulation thread before its next control step.
            props.submit(*prop_cmd)
        return
    if line.strip().lower().startswith("search"):
        if runtime.belief_search is None:
            print("[!] belief search needs the MPC plant")
            return
        spec = parse_search_command(line)
        if spec is not None:
            runtime.belief_search.request(spec)
        return
    parsed = parse_fault_command(line)
    if parsed == "help":
        print(FAULT_HELP)
        return
    if parsed == "clear":
        runtime.clear_interactive_faults()
        return
    if isinstance(parsed, FaultSpec):
        runtime.inject_fault_now(parsed)
        return
    text = line.strip()
    if not text:
        return
    goal = parse_goal_command(text)
    if goal is not None and not llm_enabled and not navila:
        x, y, speed = goal
        _apply_goal(runtime, x, y, speed)
        return
    if llm_enabled or navila:
        runtime.submit_operator(text)
        if not navila:
            print(f"[operator] queued: {text}")
    else:
        print(
            "[!] Not a fault or goal command, and LLM is off. "
            "Type 'goal x y' or 'help'."
        )


def _apply_goal(
    runtime: LokaRuntime,
    x: float,
    y: float,
    speed: float | None,
    heading: float | None = None,
) -> None:
    """Arm a world goal on the plant. The gait stands once it is inside tolerance."""
    cruise = 0.15 if speed is None else float(speed)
    targets = {
        "gait.mode": "walk",
        "gait.speed": cruise,
        "gait.goal_x": float(x),
        "gait.goal_y": float(y),
    }
    if heading is not None:
        targets["gait.goal_heading"] = float(heading)
    applied = runtime.controller.set_task_targets(targets)
    print(f"[goal] walk to ({x:g}, {y:g}) at {cruise:g} m/s: {applied}")


def _stdin_command_thread(
    runtime: LokaRuntime,
    stop: threading.Event,
    *,
    llm_enabled: bool,
    navila: bool = False,
) -> None:
    print(FAULT_HELP)
    if navila:
        print(
            "[interactive] type a navigation instruction, 'act move forward 50 cm', "
            "a fault command, or 'help', then Enter"
        )
    else:
        print("[interactive] type a fault command, 'goal x y', or an operator request, then Enter")
    while not stop.is_set():
        try:
            line = sys.stdin.readline()
        except Exception:
            break
        if not line:
            break
        _handle_interactive_line(runtime, line, llm_enabled=llm_enabled, navila=navila)


def _make_key_callback(runtime: LokaRuntime):
    keys = KeyDebounce()

    def key_callback(keycode: int) -> None:
        # GLFW keycodes: digits and letters are their Unicode code points.
        action = fault_for_keycode(keycode)
        if action is None:
            if runtime.belief_search is None:
                return
            spec = search_for_keycode(keycode)
            if spec is not None and keys.allow(keycode):
                runtime.belief_search.request(spec)
            return
        if not keys.allow(keycode):
            return
        if action == "help":
            print(FAULT_HELP)
        elif action == "clear":
            runtime.clear_interactive_faults()
        else:
            runtime.inject_fault_now(action)

    return key_callback


def _install_local_orchestrator(args: argparse.Namespace) -> str | None:
    """Route orchestrator turns to a local VLM. Returns its label, or None for OpenAI.

    Runs before the plant is built, so a bad checkpoint, env or GPU fails before
    the render processes start.
    """
    if args.orchestrator_mllama is None and args.orchestrator_vila is None:
        return None
    from loka.agent.llm import set_chat_backend
    from loka.vla.navila import (
        apply_scratch_cache_defaults,
        prefer_egl_rendering,
        require_cuda_gpu,
        require_navila_checkpoint,
    )

    device = args.orchestrator_device
    if args.orchestrator_mllama is not None:
        from loka.vla.mllama_chat import MLLAMA_FETCH_HINT, MLLAMA_MIN_FREE_GB

        tag, name, hint, path = "mllama", "Llama-3.2-Vision", MLLAMA_FETCH_HINT, args.orchestrator_mllama
        gpu = {
            "min_free_gb": MLLAMA_MIN_FREE_GB,
            "model": "Llama 3.2 Vision",
            "setup_script": "scripts/setup_mllama_env.sh",
            "packages": ("transformers", "accelerate"),
        }
    else:
        from loka.vla.vila_chat import VILA_FETCH_HINT

        tag, name, hint, path = "vila", "VILA", VILA_FETCH_HINT, args.orchestrator_vila
        gpu = {
            "min_free_gb": 20.0,
            "model": "VILA",
            "setup_script": "scripts/setup_navila_env.sh",
            "packages": ("llava",),
        }
    checkpoint = require_navila_checkpoint(path, name=name, hint=hint)
    caches = apply_scratch_cache_defaults()
    print(f"[{tag}] HF_HOME={caches['HF_HOME']} XDG_CACHE_HOME={caches['XDG_CACHE_HOME']}")
    if device.startswith("cuda"):
        require_cuda_gpu(device, **gpu)
        print(f"[{tag}] MUJOCO_GL={prefer_egl_rendering()} for the ego camera and recording")
    else:
        print(f"[{tag}] device={device}: GPU check skipped; only tiny test checkpoints are practical")
    if tag == "mllama":
        from loka.vla.mllama_chat import MllamaChatBackend

        backend = MllamaChatBackend(checkpoint, device=device)
    else:
        from loka.vla.vila_chat import VilaChatBackend

        backend = VilaChatBackend(checkpoint, device=device)
    set_chat_backend(backend)
    print(f"[{tag}] orchestrator: {backend.label} on {device}")
    return backend.label


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    llm_enabled = not args.no_llm
    config = LocomotionConfig.from_yaml(args.config) if args.config else LocomotionConfig()
    apply_stack_arg(config, args)
    settings = loka_config()
    vision_cfg = settings.get("vision") or {}
    navila_cfg_raw = dict(settings.get("navila") or {})
    vision = llm_enabled and (args.vision or bool(vision_cfg.get("enabled", False)))
    if args.navila:
        camera = EgoCameraSpec.from_config(navila_cfg_raw.get("camera") or {})
    else:
        camera = EgoCameraSpec.from_config(vision_cfg)
    args.ego_hz = float(vision_cfg.get("stream_hz", 15.0))
    prop_cfg = settings.get("props") or {}
    need_camera = vision or args.ego_view or args.navila
    plant = {
        "props": None if args.no_props else dict(prop_cfg.get("slots") or {}),
        "static_props": tuple(prop_cfg.get("static") or ()),
        "camera": camera.recipe() if need_camera else None,
    }
    if args.navila:
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

        if args.navila_device is not None:
            navila_cfg_raw["device"] = args.navila_device
        if args.navila_model is not None:
            navila_cfg_raw["model_path"] = str(args.navila_model)
        nav_cfg = NavilaConfig.from_dict(navila_cfg_raw)
        script_actions = None
        policy = None
        if args.navila_script is not None:
            script_actions = parse_script(args.navila_script)
        else:
            path = require_navila_checkpoint(nav_cfg.model_path)
            caches = apply_scratch_cache_defaults()
            print(f"[navila] HF_HOME={caches['HF_HOME']} XDG_CACHE_HOME={caches['XDG_CACHE_HOME']}")
            require_navila_gpu(
                nav_cfg.device, nav_cfg.min_free_gb, nav_cfg.min_compute_capability
            )
            print(f"[navila] MUJOCO_GL={prefer_egl_rendering()} for the ego camera and recording")
            policy = NavilaPolicy(
                path,
                device=nav_cfg.device,
                num_frames=nav_cfg.num_frames,
                max_new_tokens=nav_cfg.max_new_tokens,
            )
        plant_name = "unitree_rl" if args.unitree_rl else "mpc"
        if args.unitree_rl:
            sim = UnitreeRLSimulation(
                config, model_path=args.model, policy_dir=args.policy_dir, **plant
            )
        else:
            sim = Simulation(config, model_path=args.model, **plant)
        runtime = NavilaRuntime(
            sim,
            nav_cfg,
            plant=plant_name,
            policy=policy,
            script=script_actions,
            faults=build_faults(args),
            enable_camera=policy is not None,
        )
    elif args.unitree_rl:
        sim = UnitreeRLSimulation(
            config, model_path=args.model, policy_dir=args.policy_dir, **plant
        )
        runtime = _PlantRuntime(sim, build_faults(args))
    else:
        llm_model = _install_local_orchestrator(args)
        sim = Simulation(config, model_path=args.model, **plant)
        log_path = None
        if not args.no_log:
            log_path = default_log_path(args.log_dir)
        describe_first = bool(args.describe_first) and vision
        if args.describe_first and not vision and "--describe-first" in (argv or sys.argv[1:]):
            print("[LOKA] --describe-first needs the camera; add --vision. Describing is off.")
        memory = {} if args.max_memory is None else {"max_memory": int(args.max_memory)}
        print(
            f"[LOKA] prompt={args.prompt_style} describe_first={describe_first} "
            f"reply_check={args.reply_check} max_memory={memory.get('max_memory', 'loka.yaml')}"
        )
        runtime = LokaRuntime(
            sim,
            LokaConfig(
                enable_llm=llm_enabled,
                llm_model=llm_model,
                prompt_style=args.prompt_style,
                describe_first=describe_first,
                reply_check=bool(args.reply_check),
                **memory,
                enable_vision=vision,
                vision=camera,
                enable_session_log=not args.no_log,
                log_path=log_path,
                print_compressor=not args.quiet_compressor,
            ),
            faults=build_faults(args),
        )
        runtime.effort_observer.printout = True
    if args.prop_spawns and sim.props is None:
        print("[PROPS] --props / --scene ignored: props.slots in loka.yaml is empty")
        args.prop_spawns = []
    for kind, x, y, yaw in args.prop_spawns:
        try:
            print(sim.props.spawn(kind, x, y, yaw))
        except ValueError as exc:
            print(f"[PROPS] {exc}")
    log = getattr(runtime, "log", None)
    if log is not None:
        props = ";".join(f"{k}:{x:g},{y:g},{yaw:g}" for k, x, y, yaw in args.prop_spawns)
        log.note(
            f"argv: {' '.join(sys.argv[1:]) or '(none)'}\nstartup props: {props or '(none)'}",
            sim_time=0.0,
        )
    recorder = None
    if args.record is not None:
        record_cfg = settings.get("recording") or {}
        path = Path(args.record) if args.record else (
            Path(record_cfg.get("directory", "recordings"))
            / f"run_loka_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        )
        recorder = SessionRecorder(sim, path, OverviewCamera.from_config(record_cfg))
        print(f"[record] capturing at {recorder.camera.fps:g} fps -> {path} (encoded at exit)")
    try:
        return _run(args, sim, runtime, llm_enabled, recorder)
    finally:
        if hasattr(runtime, "close"):
            runtime.close()
        if recorder is not None:
            recorder.close()


def _run(args, sim, runtime, llm_enabled: bool, recorder=None) -> int:
    if args.mode == "walk":
        applied = runtime.controller.set_task_targets(
            {
                "gait.mode": "walk",
                "gait.speed": float(args.speed),
                "gait.heading": float(args.heading),
            }
        )
        print(f"[gait] initial walk Task_Targets: {applied}")

    duration = args.duration or (20.0 if args.headless else None)
    stop = threading.Event()

    if args.operator or args.task or args.goal_xy or (args.navila and args.navila_script):
        def _queue_later():
            while not stop.is_set() and sim.data.time < args.operator_at:
                time.sleep(0.05)
            if stop.is_set():
                return
            if args.task is not None:
                applied = runtime.controller.set_task_targets(args.task)
                print(f"[ablation] Task_Targets applied: {applied}")
            if args.goal_xy is not None:
                _apply_goal(
                    runtime,
                    args.goal_xy[0],
                    args.goal_xy[1],
                    args.goal_speed,
                    args.goal_heading,
                )
            elif args.navila and args.navila_script and not args.operator:
                runtime.submit_instruction(args.navila_script)
            elif args.operator:
                runtime.submit_operator(args.operator)

        threading.Thread(target=_queue_later, daemon=True).start()

    if args.navila:
        plant = "navila+" + ("unitree_rl" if args.unitree_rl else "mpc")
    elif args.unitree_rl:
        plant = "unitree_rl policy"
    else:
        plant = "MPC plant"
    print(
        f"G1 {plant}: orchestrator {'on' if llm_enabled else 'off'}  fault={args.fault}"
    )

    stdin_stop = threading.Event()
    if sys.stdin.isatty():
        threading.Thread(
            target=_stdin_command_thread,
            args=(runtime, stdin_stop),
            kwargs={"llm_enabled": llm_enabled, "navila": bool(args.navila)},
            daemon=True,
        ).start()

    if args.headless:
        try:
            if recorder is None:
                history = runtime.run(duration or 15.0)
            else:
                end = duration or 15.0
                recorder.maybe_capture()
                while sim.data.time < end and not sim.fell:
                    runtime.step()
                    recorder.maybe_capture()
                history = getattr(runtime, "history", sim.history)
        finally:
            stop.set()
            stdin_stop.set()
        fell = sim.fell
        print(sim.stats().report())
        print(f"tracking samples: {len(history)}  fell={fell}")
        x, y = (float(v) for v in sim.data.qpos[:2])
        print(f"final pelvis xy : ({x:.3f}, {y:.3f})")
        return 1 if fell else 0

    import mujoco.viewer
    from loka.main import RENDER_HZ

    steps_per_frame = max(1, round(1.0 / (RENDER_HZ * sim.config.control_dt)))
    frame_dt = steps_per_frame * sim.config.control_dt

    ego_view = None
    if args.ego_view:
        ego_view = EgoViewStream(sim.recipe, hz=args.ego_hz)
        print(f"[ego view] streaming {EGO_CAMERA} at up to {args.ego_hz:g} Hz ({ego_view.backend})")

    try:
        with mujoco.viewer.launch_passive(
            sim.model,
            sim.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=_make_key_callback(runtime),
        ) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            next_frame = time.perf_counter()
            while viewer.is_running():
                if duration is not None and sim.data.time >= duration:
                    break
                for _ in range(steps_per_frame):
                    runtime.step()
                    if recorder is not None:
                        recorder.maybe_capture()
                    if sim.fell:
                        break
                begin_frame(viewer)
                render_fault_overlays(viewer, sim.model, sim.data, runtime.viz)
                if not args.unitree_rl:
                    render_gait_overlays(viewer, runtime.controller)
                viewer.sync()
                if ego_view is not None:
                    ego_view.offer(sim)
                next_frame += frame_dt
                sleep = next_frame - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_frame = time.perf_counter()
                if sim.fell:
                    print(f"FELL at t={sim.data.time:.2f}s", file=sys.stderr)
                    break
    finally:
        stop.set()
        stdin_stop.set()
        if ego_view is not None:
            ego_view.close()

    print(sim.stats().report())
    return 1 if sim.fell else 0


if __name__ == "__main__":
    raise SystemExit(main())
