#!/usr/bin/env python3
"""Baby-step G1 torso tracking: position MPC → actuator PD.

No CPG. Arms locked by default; waist/hips unlocked. MPC tracks a fixed torso
setpoint (XY anchored on Start, Z from the height slider, upright) and outputs
joint positions ``q_ref``. Actuator PD produces torques.

Planners (UI dropdown): Convex QP | Predictive Sampling | iLQG

Usage::

    python sim/run_g1_sim.py          # terminal 1
    python -m loka.main               # terminal 2
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loka.nodes.actuator_manager import ActuatorConfig, ActuatorManager, SoftStarter, quat_to_rpy
from loka.nodes.position_mpc import PositionMPCWeights, TorsoPositionMPC
from loka.nodes.state_estimator import GroundTruthStateEstimator
from loka.ui.dashboard import DashboardUI, SharedControlState
from loka.utils.dds_interface import DDSInterface

logger = logging.getLogger("loka.main")

CONFIG_DIR = Path(__file__).resolve().parent / "config"
CONTROL_DT = 0.002


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def shared_from_configs(g1: Mapping, ctrl: Mapping) -> SharedControlState:
    torso = ctrl.get("torso", {})
    mpc = ctrl.get("mpc", {})
    w = mpc.get("weights", {})
    act = ctrl.get("actuator", {})
    planner_key = str(mpc.get("planner", "convex_qp"))
    planner_label = {
        "convex_qp": "Convex QP",
        "predictive_sampling": "Predictive Sampling",
        "ilqg": "iLQG",
    }.get(planner_key, "Convex QP")
    return SharedControlState(
        freeze_arms=bool(act.get("freeze_arms", True)),
        # Hips / waist unlocked by default so MPC can use them for torso attitude.
        freeze_waist=bool(act.get("freeze_waist", False)),
        enable_leg_torque=False,
        planner=planner_label,
        stride_period=0.8,
        clearance=0.0,
        sweep_amplitude=0.0,
        duty_factor=0.6,
        v_ref=0.0,
        y_offset=0.0,
        z_ref=float(torso.get("z_target", g1.get("standing_height", 0.793))),
        theta_ref=0.0,
        w_z=float(w.get("w_z", 50.0)),
        w_p=float(w.get("w_p", 20.0)),
        w_u=float(w.get("w_u", 1.0)),
        w_theta=float(w.get("w_theta", 40.0)),
        w_q=float(w.get("w_q", 5.0)),
        w_tau=float(w.get("w_tau", 0.01)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("interface", nargs="?", default=None)
    parser.add_argument("--domain", type=int, default=None)
    parser.add_argument("--no-ui", action="store_true")
    parser.add_argument("--auto-walk", action="store_true", help="Engage stand immediately")
    parser.add_argument("--g1-config", type=Path, default=CONFIG_DIR / "g1_params.yaml")
    parser.add_argument("--ctrl-config", type=Path, default=CONFIG_DIR / "default_cpg_mpc.yaml")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def resolve_dds(args: argparse.Namespace, g1: Mapping) -> Tuple[int, str]:
    dds_cfg = g1.get("dds", {})
    if args.interface:
        return (args.domain if args.domain is not None else 0), args.interface
    domain = args.domain if args.domain is not None else int(dds_cfg.get("domain_id", 1))
    return domain, str(dds_cfg.get("interface", "lo"))


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    try:
        g1_cfg = load_yaml(args.g1_config)
        ctrl_cfg = load_yaml(args.ctrl_config)
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        return 1

    domain, interface = resolve_dds(args, g1_cfg)
    dds_topics = g1_cfg.get("dds", {})
    try:
        dds = DDSInterface(
            domain_id=domain,
            interface=interface,
            topic_lowcmd=str(dds_topics.get("topic_lowcmd", "rt/lowcmd")),
            topic_lowstate=str(dds_topics.get("topic_lowstate", "rt/lowstate")),
            topic_sportmodestate=str(
                dds_topics.get("topic_sportmodestate", "rt/sportmodestate")
            ),
        )
    except Exception as exc:
        logger.error("DDS setup failed: %s", exc)
        return 1

    default_q = np.array(g1_cfg.get("default_joint_pos", [0.0] * 29), dtype=np.float64)
    estimator = GroundTruthStateEstimator(dds)
    estimator.start()

    act_cfg = ActuatorConfig.from_dicts(g1_cfg, ctrl_cfg, default_q=default_q)
    act_cfg.enable_leg_torque = False
    act_cfg.freeze_arms = bool(ctrl_cfg.get("actuator", {}).get("freeze_arms", True))
    act_cfg.freeze_waist = bool(ctrl_cfg.get("actuator", {}).get("freeze_waist", False))
    actuators = ActuatorManager(act_cfg, dds=dds)
    soft_start = SoftStarter(duration=float(g1_cfg.get("soft_start_duration", 2.0)))

    mpc_cfg = ctrl_cfg.get("mpc", {})
    mpc_w = PositionMPCWeights.from_dict(mpc_cfg.get("weights", {}))
    ps = mpc_cfg.get("predictive_sampling", {})
    ilqg = mpc_cfg.get("ilqg", {})
    planner_key = str(mpc_cfg.get("planner", "convex_qp"))
    planner_label = {
        "convex_qp": "Convex QP",
        "predictive_sampling": "Predictive Sampling",
        "ilqg": "iLQG",
    }.get(planner_key, "Convex QP")
    mpc = TorsoPositionMPC(
        q_nominal=default_q,
        weights=mpc_w,
        planner=planner_label,
        horizon_steps=int(mpc_cfg.get("horizon", 50)),
        model_dt=float(mpc_cfg.get("dt", 0.02)),
        dyn_stiffness=float(mpc_cfg.get("dyn_stiffness", 16.0)),
        dyn_damping=float(mpc_cfg.get("dyn_damping", 8.0)),
        num_samples=int(ps.get("num_samples", 64)),
        sample_std=float(ps.get("noise_std", 0.08)),
        ilqg_iters=int(ilqg.get("max_iters", 5)),
    )

    shared = shared_from_configs(g1_cfg, ctrl_cfg)
    if args.auto_walk:
        shared.walking = True

    ui_cfg = ctrl_cfg.get("ui", {})
    dashboard: Optional[DashboardUI] = None
    if not args.no_ui:
        dashboard = DashboardUI(
            shared,
            title=str(ui_cfg.get("window_title", "LOKA G1 Torso MPC")),
            width=int(ui_cfg.get("width", 1040)),
            height=int(ui_cfg.get("height", 1560)),
        )
        dashboard.start()

    logger.info(
        "Torso position-MPC @ %.0f Hz | domain=%s iface=%s | planner=%s | horizon=%.2fs | soft-start=%.1fs",
        1.0 / CONTROL_DT,
        domain,
        interface,
        mpc.planner,
        mpc.horizon_steps * mpc.model_dt,
        soft_start.duration,
    )
    logger.info(
        "No CPG. Arms locked by default; waist/hips unlocked. "
        "MPC outputs q_ref only → actuator PD. Pick planner in the UI."
    )

    was_engaged = False
    t_wall0 = time.perf_counter()
    last_diag_print = 0.0
    DIAG_INTERVAL = 0.5  # print MPC diagnostics every 0.5 s
    time.sleep(0.1)

    try:
        while True:
            step_start = time.perf_counter()
            t = step_start - t_wall0
            snap = shared.snapshot()

            actuators.set_enable_leg_torque(False)
            actuators.set_freeze_arms(snap.freeze_arms)
            actuators.set_freeze_waist(snap.freeze_waist)
            mpc.set_planner(snap.planner)
            mpc.update_weights(
                w_z=snap.w_z,
                w_xy=snap.w_p,
                w_theta=snap.w_theta,
                w_q=snap.w_q,
            )

            if snap.estop or snap.reset_requested:
                actuators.emergency_stop()
                soft_start.cancel()
                was_engaged = False
                shared.clear_reset()
                shared.set_status("ESTOP")
                if snap.estop:
                    time.sleep(CONTROL_DT)
                    continue

            state = estimator.get_state()

            if snap.walking and not was_engaged:
                mpc.reset_anchor(state)
                soft_start.begin(t, state.joint_pos, default_q)
                was_engaged = True
                logger.info(
                    "Engaged: soft-start then torso MPC (%s, z_ref=%.3f)",
                    snap.planner,
                    snap.z_ref,
                )
            elif not snap.walking and was_engaged:
                soft_start.cancel()
                was_engaged = False
                logger.info("Disengaged")

            if not snap.walking:
                actuators.send(tau_ff=np.zeros(29), q_ref=state.joint_pos)
                shared.set_status("IDLE")
                time.sleep(max(0.0, CONTROL_DT - (time.perf_counter() - step_start)))
                continue

            if soft_start.active:
                q_ref = soft_start.evaluate(t)
                cost = 0.0
                phase = "SOFT"
                err = np.zeros(5)
                planner_tag = snap.planner
            else:
                result = mpc.solve(
                    state,
                    z_ref=snap.z_ref,
                    use_waist=not snap.freeze_waist,
                )
                q_ref = result.q_ref
                cost = result.cost
                err = result.torso_error
                phase = "MPC"
                planner_tag = result.planner

            actuators.send(tau_ff=np.zeros(29), q_ref=q_ref)

            rpy = quat_to_rpy(state.base_quat)
            shared.set_status(
                f"{phase}/{planner_tag} | z={state.base_pos[2]:.3f}/{snap.z_ref:.2f} "
                f"pitch={rpy[1]:+.2f} roll={rpy[0]:+.2f} "
                f"e_z={err[2]:+.3f}",
                cost=cost,
            )

            # --- Diagnostic print every DIAG_INTERVAL seconds ---
            if t - last_diag_print >= DIAG_INTERVAL and phase != "IDLE":
                last_diag_print = t
                delta = q_ref - default_q
                leg_delta = delta[:12]
                leg_qref = q_ref[:12]
                leg_meas = state.joint_pos[:12]
                leg_names = ["Lhp", "Lhr", "Lhy", "Lkn", "Lap", "Lar",
                             "Rhp", "Rhr", "Rhy", "Rkn", "Rap", "Rar"]
                sys.stdout.write(
                    f"\n[DIAG t={t:.1f}s {phase}/{planner_tag}] "
                    f"err=[ex={err[0]:+.3f} ey={err[1]:+.3f} ez={err[2]:+.3f} "
                    f"eroll={err[3]:+.3f} epitch={err[4]:+.3f}]  cost={cost:.4f}\n"
                    f"  q_ref  legs: {np.array2string(leg_qref, precision=3, suppress_small=True)}\n"
                    f"  q_meas legs: {np.array2string(leg_meas, precision=3, suppress_small=True)}\n"
                    f"  Δ(MPC-nom):  {np.array2string(leg_delta, precision=3, suppress_small=True)}\n"
                    f"  max|Δ|={np.max(np.abs(leg_delta)):.4f}  "
                    f"track_err={np.linalg.norm(leg_qref - leg_meas):.4f}\n"
                )
                sys.stdout.flush()

            time.sleep(max(0.0, CONTROL_DT - (time.perf_counter() - step_start)))
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        try:
            actuators.emergency_stop()
        except Exception:
            pass
        if dashboard is not None:
            dashboard.stop()
        estimator.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
