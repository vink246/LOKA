"""High-rate walk traces, aimed at the remaining balance hypotheses.

    python -m loka.diagnose_walk                  # run, save, print a report
    python -m loka.diagnose_walk --speed 0.20
    python -m loka.diagnose_walk --report logs/walk/last.npz

``verify_gait`` already tells us *that* the robot falls sideways, and that the
planner is innocent in isolation. This module records the quantities that
distinguish *why* the closed loop diverges -- CoP against the sole edge, CoM
error saturation per axis, capture-point correction against its clamp, swing
landing error -- so the next change is aimed at a measured mechanism rather
than a guessed one.

The G1 sole is 60 mm wide (contact sites at y = ±25 mm heel, ±30 mm toe). A
CoP sitting at ±30 mm of the stance-foot centre is on the edge; anything
beyond that is the QP asking for a ZMP the foot cannot produce.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from loka.control.gait import GRAVITY, MAX_COM_REF_ERROR, MAX_DCM_CORRECTION, heading_frame
from loka.sim import Simulation

#: Lateral half-width of a G1 sole, from the contact sites in ``g1_29dof.xml``.
SOLE_HALF_WIDTH = 0.030
#: Sagittal half-length about the foot-centre site.
SOLE_HALF_LENGTH = 0.085

COLUMNS = (
    "t",
    "step",
    "tau",
    "phase",
    "swing",
    "swing_leg",
    "stance_leg",
    # Heading-frame positions / velocities.
    "com_f",
    "com_l",
    "com_vf",
    "com_vl",
    "com_ref_f",  # saturated, handed to the QP
    "com_ref_l",
    "com_plan_f",  # unsaturated plan integrator
    "com_plan_l",
    "com_vref_f",
    "com_vref_l",
    "dcm_f",
    "dcm_l",
    "dcm_ref_f",
    "dcm_ref_l",
    "zmp_f",
    "zmp_l",
    "cop_des_f",
    "cop_des_l",
    "cop_got_f",
    "cop_got_l",
    "foot_stance_f",
    "foot_stance_l",
    "foot_swing_f",
    "foot_swing_l",
    "foothold_f",
    "foothold_l",
    "foothold_nom_f",
    "foothold_nom_l",
    "corr_f",
    "corr_l",
    "corr_norm",
    "swing_err",
    "swing_clear",
    "roll_deg",
    "pitch_deg",
    "margin_l",  # support_margin +y (left) [m]
    "margin_r",
    "n_contact",
    "n_planned",
    "sat_com",  # 1 if ||plan - measured|| exceeded MAX_COM_REF_ERROR
    "sat_lat_share",  # fraction of saturated error that is lateral
    "cop_edge",  # |cop_des - stance_foot|_lateral / SOLE_HALF_WIDTH; nan in DS
    "swing_s",  # Bézier parameter in [0, 1]
    "swing_z",  # swing-foot height above ground [m]
    "swing_des_z",  # commanded swing-foot height [m]
    "swing_tilt_deg",  # sole tilt vs world +z [deg]; 0 = flat
    "yaw",
    "yaw_ref",
    "heading_err",
    "plan_yaw",
)


@dataclass
class WalkTrace:
    """One recorded walk: a column-major table plus the policy that produced it."""

    columns: dict[str, np.ndarray]
    speed: float
    duration: float
    fell: bool
    elapsed: float
    policy: dict[str, float]

    def __getitem__(self, name: str) -> np.ndarray:
        return self.columns[name]

    @property
    def n(self) -> int:
        return int(self.columns["t"].size)


def _project(xy: np.ndarray, forward: np.ndarray, left: np.ndarray) -> tuple[float, float]:
    xy = np.asarray(xy, dtype=float).reshape(2)
    return float(xy @ forward), float(xy @ left)


def _cop(forces: np.ndarray, positions: np.ndarray) -> np.ndarray | None:
    """Force-weighted CoP in world xy, or None if nothing is pushing down."""
    fz = np.asarray(forces, dtype=float).reshape(-1, 3)[:, 2]
    total = float(fz.sum())
    if total < 1.0:
        return None
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    return (fz[:, None] * pos[:, :2]).sum(axis=0) / total


def record_walk(
    *,
    speed: float = 0.25,
    duration: float = 10.0,
    decimate: int = 4,
    script: list | None = None,
    pushes: list | None = None,
    config=None,
    **gait,
) -> WalkTrace:
    """Run a walk and sample the controller every ``decimate`` ticks.

    ``script`` is a ``walk_bench`` event list ``(time, task updates)``. When it
    is omitted the walk starts immediately at ``speed``.
    """
    sim = Simulation(config, pushes=list(pushes or []))
    controller = sim.controller
    if script is None:
        pending = [(0.0, {"gait.mode": "walk", "gait.speed": speed, **gait})]
    else:
        pending = list(script)
    applied: dict = {}
    omega = float(np.sqrt(GRAVITY / max(0.30, controller.nominal_height)))

    rows: list[list[float]] = []
    tick = 0
    while sim.data.time < duration:
        now = float(sim.data.time)
        while pending and pending[0][0] <= now:
            _, updates = pending.pop(0)
            applied.update(controller.set_task_targets(updates))
        sim.step()
        tick += 1
        if tick % decimate != 0 and not sim.fell:
            continue

        gait_sched = controller.gait
        tel = controller.telemetry
        forward, left = heading_frame(float(tel.yaw_ref))
        robot = controller.robot
        gait_out = getattr(controller, "last_gait", None)
        if gait_out is None:
            gait_out = controller._last_gait
        dyn = robot.dynamics(sim.data.qvel)

        com = np.asarray(tel.com[:2], dtype=float)
        vel = np.asarray(tel.com_velocity[:2], dtype=float)
        dcm = com + vel / omega
        com_ref = np.asarray(tel.com_reference[:2], dtype=float)
        com_plan = np.asarray(gait_sched._com_ref, dtype=float)
        plan_err = com_plan - com
        plan_err_norm = float(np.linalg.norm(plan_err))
        sat = 1.0 if plan_err_norm > MAX_COM_REF_ERROR else 0.0
        if plan_err_norm > 1e-9:
            _, err_l = _project(plan_err, forward, left)
            sat_lat_share = abs(err_l) / plan_err_norm
        else:
            sat_lat_share = 0.0

        step = int(gait_sched._step)
        support = gait_sched._steps.get(step)
        landing = gait_sched._steps.get(step + 1)
        zmp = support.pos if support is not None else com
        feet = robot.foot_center_positions()
        ground_z = float(controller._ground_height)

        swing = gait_out.swing if gait_out is not None else None
        in_swing = bool(swing is not None and swing.active)
        swing_leg = int(swing.leg) if in_swing else -1
        if support is not None and support.leg >= 0:
            stance_leg = int(support.leg)
        elif in_swing:
            stance_leg = 1 - swing_leg
        else:
            stance_leg = -1

        if stance_leg >= 0:
            stance_foot = feet[stance_leg, :2]
        else:
            stance_foot = feet.mean(axis=0)[:2]
        if in_swing:
            swing_foot = feet[swing_leg, :2]
        else:
            swing_foot = np.full(2, np.nan)

        foothold = landing.pos if landing is not None else np.full(2, np.nan)
        foothold_nom = (
            landing.nominal
            if landing is not None and landing.nominal is not None
            else np.full(2, np.nan)
        )
        corr = landing.correction if landing is not None else np.zeros(2)

        cop_des = _cop(tel.desired_forces, dyn.contact_pos)
        cop_got = _cop(tel.contact_forces, dyn.contact_pos)
        if cop_des is None:
            cop_des = np.full(2, np.nan)
        if cop_got is None:
            cop_got = np.full(2, np.nan)

        cop_des_f, cop_des_l = _project(cop_des, forward, left)
        _, stance_l = _project(stance_foot, forward, left)
        # Double-support CoP sits between the feet (~4× a sole half-width) and
        # is not a saturation. Only single-support samples can read as "on the
        # edge of the stance sole".
        cop_edge = (
            abs(cop_des_l - stance_l) / SOLE_HALF_WIDTH
            if in_swing and np.isfinite(cop_des_l)
            else float("nan")
        )
        swing_s = float(swing.s) if in_swing else float("nan")
        swing_z = float(feet[swing_leg, 2] - ground_z) if in_swing else float("nan")
        swing_des_z = float(swing.des_pos[2] - ground_z) if in_swing else float("nan")

        margin = robot.support_margin()
        n_planned = int(np.asarray(tel.contact_mask).sum())
        # Measured contacts: sole points near the ground, independent of the plan.
        measured = controller._contact_mask()

        def pack(xy):
            if xy is None or not np.all(np.isfinite(xy)):
                return (float("nan"), float("nan"))
            return _project(xy, forward, left)

        row = [
            float(sim.data.time),
            float(step),
            float(gait_sched._tau),
            float(tel.gait_phase),
            1.0 if in_swing else 0.0,
            float(swing_leg),
            float(stance_leg),
            *pack(com),
            *pack(vel),
            *pack(com_ref),
            *pack(com_plan),
            *pack(tel.com_velocity_reference[:2]),
            *pack(dcm),
            *pack(gait_out.dcm_ref if gait_out is not None else np.full(2, np.nan)),
            *pack(zmp),
            *pack(cop_des),
            *pack(cop_got),
            *pack(stance_foot),
            *pack(swing_foot),
            *pack(foothold),
            *pack(foothold_nom),
            *pack(corr),
            float(np.linalg.norm(corr)),
            float(tel.swing_error),
            float(tel.swing_clearance),
            float(np.degrees(tel.rpy[0])),
            float(np.degrees(tel.rpy[1])),
            float(margin[3]),
            float(margin[2]),
            float(int(measured.sum())),
            float(n_planned),
            sat,
            sat_lat_share,
            cop_edge,
            swing_s,
            swing_z,
            swing_des_z,
            float(np.degrees(tel.swing_tilt)) if in_swing else float("nan"),
            float(tel.rpy[2]),
            float(tel.yaw_ref),
            float(tel.heading_error),
            float(gait_out.stance_yaw) if gait_out is not None else 0.0,
        ]
        rows.append(row)
        if sim.fell:
            break

    table = np.asarray(rows, dtype=float)
    columns = {name: table[:, i] for i, name in enumerate(COLUMNS)}
    return WalkTrace(
        columns=columns,
        speed=speed,
        duration=duration,
        fell=bool(sim.fell),
        elapsed=float(sim.data.time),
        policy=dict(applied),
    )


def save_trace(trace: WalkTrace, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        speed=trace.speed,
        duration=trace.duration,
        fell=int(trace.fell),
        elapsed=trace.elapsed,
        **{f"p_{k}": v for k, v in trace.policy.items()},
        **trace.columns,
    )


def load_trace(path: Path) -> WalkTrace:
    data = np.load(path, allow_pickle=False)
    columns = {name: np.asarray(data[name]) for name in COLUMNS}
    policy = {
        key[2:]: float(data[key])
        for key in data.files
        if key.startswith("p_")
    }
    return WalkTrace(
        columns=columns,
        speed=float(data["speed"]),
        duration=float(data["duration"]),
        fell=bool(data["fell"]),
        elapsed=float(data["elapsed"]),
        policy=policy,
    )


def _step_slices(trace: WalkTrace) -> list[tuple[int, slice]]:
    """``(step_index, slice)`` covering each support step that has samples."""
    steps = trace["step"].astype(int)
    if steps.size == 0:
        return []
    bounds = np.flatnonzero(np.diff(steps) != 0) + 1
    starts = np.r_[0, bounds]
    ends = np.r_[bounds, steps.size]
    return [(int(steps[s]), slice(s, e)) for s, e in zip(starts, ends)]


def per_step(trace: WalkTrace) -> list[dict[str, float]]:
    """One row per support step: the numbers that grow or saturate."""
    rows = []
    for step, sl in _step_slices(trace):
        if step <= 0:
            continue  # opening double-support transfer
        c = {k: v[sl] for k, v in trace.columns.items()}
        swing = c["swing"] > 0.5
        n = max(int(swing.sum()), 1)
        lift = int(np.argmax(swing)) if swing.any() else 0
        land = int(len(swing) - 1 - np.argmax(swing[::-1])) if swing.any() else -1

        def at(name, i):
            arr = c[name]
            if arr.size == 0:
                return float("nan")
            return float(arr[i])

        rows.append(
            dict(
                step=float(step),
                stance=at("stance_leg", 0),
                com_l_peak=float(np.nanmax(np.abs(c["com_l"]))),
                com_l_mean=float(np.nanmean(c["com_l"])),
                com_l_lift=at("com_l", lift),
                com_vl_lift=at("com_vl", lift),
                com_vl_land=at("com_vl", land),
                com_plan_l_peak=float(np.nanmax(np.abs(c["com_plan_l"]))),
                dcm_err_l=float(np.nanmean(c["dcm_l"] - c["dcm_ref_l"])),
                dcm_err_l_eos=at("dcm_l", -1) - at("dcm_ref_l", -1),
                corr_l=float(np.nanmean(c["corr_l"][swing])) if swing.any() else 0.0,
                corr_peak=float(np.nanmax(c["corr_norm"])) if n else 0.0,
                sat_frac=float(np.mean(c["sat_com"])),
                sat_lat_share=float(np.mean(c["sat_lat_share"])),
                cop_edge_p95=float(np.nanpercentile(c["cop_edge"][np.isfinite(c["cop_edge"])], 95))
                if np.isfinite(c["cop_edge"]).any()
                else float("nan"),
                cop_edge_max=float(np.nanmax(c["cop_edge"]))
                if np.isfinite(c["cop_edge"]).any()
                else float("nan"),
                swing_err_max=float(np.nanmax(c["swing_err"])) if swing.any() else 0.0,
                swing_z_land=at("swing_z", land),
                swing_des_z_land=at("swing_des_z", land),
                swing_s_land=at("swing_s", land),
                land_err_f=at("foot_swing_f", land) - at("foothold_f", land),
                land_err_l=at("foot_swing_l", land) - at("foothold_l", land),
                swing_tilt_land=at("swing_tilt_deg", land),
                roll_peak=float(np.nanmax(np.abs(c["roll_deg"]))),
                margin_min=float(np.nanmin(np.minimum(c["margin_l"], c["margin_r"]))),
            )
        )
    return rows


def summarize(trace: WalkTrace) -> dict:
    """Compact verdicts against each remaining hypothesis."""
    c = trace.columns
    n = trace.n
    if n == 0:
        return {"fell": trace.fell, "elapsed": trace.elapsed, "samples": 0}

    finite_cop = np.isfinite(c["cop_edge"])
    sat = c["sat_com"] > 0.5
    steps = per_step(trace)
    peaks = [s["com_l_peak"] for s in steps]
    growth = []
    for a, b in zip(peaks, peaks[1:]):
        if a > 1e-4:
            growth.append(b / a)

    corr_peak = float(np.nanmax(c["corr_norm"])) if n else 0.0
    # Touchdown height on the first four steps, before the fall dominates.
    early = [s for s in steps if s["step"] <= 4]
    right_land = [s["swing_z_land"] for s in early if int(s["stance"]) == 0]
    left_land = [s["swing_z_land"] for s in early if int(s["stance"]) == 1]
    right_tilt = [s["swing_tilt_land"] for s in early if int(s["stance"]) == 0]
    left_tilt = [s["swing_tilt_land"] for s in early if int(s["stance"]) == 1]
    return dict(
        fell=trace.fell,
        elapsed=trace.elapsed,
        samples=n,
        speed=trace.speed,
        n_steps=int(c["step"].max()) if n else 0,
        com_l_max=float(np.nanmax(np.abs(c["com_l"]))),
        com_f_err_rms=float(np.sqrt(np.nanmean((c["com_f"] - c["com_plan_f"]) ** 2))),
        com_l_err_rms=float(np.sqrt(np.nanmean((c["com_l"] - c["com_plan_l"]) ** 2))),
        com_plan_l_max=float(np.nanmax(np.abs(c["com_plan_l"]))),
        sat_frac=float(np.mean(sat)),
        sat_lat_share=float(np.mean(c["sat_lat_share"][sat])) if sat.any() else 0.0,
        cop_edge_p95=float(np.nanpercentile(c["cop_edge"][finite_cop], 95))
        if finite_cop.any()
        else float("nan"),
        cop_edge_frac_gt1=float(np.mean(c["cop_edge"][finite_cop] > 1.0))
        if finite_cop.any()
        else float("nan"),
        corr_peak=corr_peak,
        corr_sat_frac=float(np.mean(c["corr_norm"] > 0.95 * MAX_DCM_CORRECTION)),
        swing_err_p95=float(np.nanpercentile(c["swing_err"][c["swing"] > 0.5], 95))
        if (c["swing"] > 0.5).any()
        else 0.0,
        roll_peak=float(np.nanmax(np.abs(c["roll_deg"]))),
        peak_growth_median=float(np.median(growth)) if growth else float("nan"),
        left_land_z=float(np.nanmean(left_land)) if left_land else float("nan"),
        right_land_z=float(np.nanmean(right_land)) if right_land else float("nan"),
        left_land_tilt=float(np.nanmean(left_tilt)) if left_tilt else float("nan"),
        right_land_tilt=float(np.nanmean(right_tilt)) if right_tilt else float("nan"),
        steps=steps,
    )


def format_report(trace: WalkTrace) -> str:
    s = summarize(trace)
    lines = [
        f"walk  speed={s['speed']:.2f} m/s   t={s['elapsed']:.2f}s   "
        f"{'FELL' if s['fell'] else 'up'}   steps={s['n_steps']}   "
        f"samples={s['samples']}",
        "",
        "Tracking (heading frame):",
        f"  sagittal CoM error rms     {s['com_f_err_rms']*1e3:7.1f} mm",
        f"  lateral  CoM error rms     {s['com_l_err_rms']*1e3:7.1f} mm",
        f"  lateral  CoM |peak|        {s['com_l_max']*1e3:7.1f} mm",
        f"  lateral  CoM *plan* |peak| {s['com_plan_l_max']*1e3:7.1f} mm"
        f"   (limit-cycle amplitude; isolated planner is ±23 mm)",
        f"  roll peak                  {s['roll_peak']:7.1f} deg",
        f"  swing tracking p95         {s['swing_err_p95']*1e3:7.1f} mm",
        "",
        "Hypothesis checks:",
        f"  H-sat  CoM-ref 2D clamp active          {100*s['sat_frac']:5.1f}% of samples"
        f"   (lateral share of the clamped error: {100*s['sat_lat_share']:4.0f}%)",
        f"  H-cop  |CoP-foot|_lat / half-width p95  {s['cop_edge_p95']:5.2f}"
        f"   (1.0 = sole edge; {100*s['cop_edge_frac_gt1']:4.1f}% beyond)",
        f"  H-cap  capture correction peak          {s['corr_peak']*1e3:6.1f} mm"
        f"   / clamp {MAX_DCM_CORRECTION*1e3:.0f} mm"
        f"   ({100*s['corr_sat_frac']:4.1f}% saturated)",
        f"  H-swg  swing-foot height at touchdown   "
        f"L {s['left_land_z']*1e3:.1f} mm   R {s['right_land_z']*1e3:.1f} mm"
        f"   (first 4 steps; 0 = planted)",
        f"  H-plt  sole tilt at touchdown            "
        f"L {s['left_land_tilt']:.1f}°   R {s['right_land_tilt']:.1f}°"
        f"   (0 = world-flat)",
        f"  H-amp  |com_l| peak growth / step       {s['peak_growth_median']:5.2f}x"
        f"   (1.0 = bounded cycle; >1 compounds)",
        "",
        f"{'step':>4} {'st':>3} {'|c|_mm':>7} {'vl_lift':>8} {'vl_land':>8} "
        f"{'dcm_e':>7} {'corr':>6} {'sat%':>5} {'cop95':>6} {'z_mm':>6} "
        f"{'tilt':>5} {'land_f':>7}",
    ]
    for row in s["steps"]:
        lines.append(
            f"{int(row['step']):4d} {int(row['stance']):3d} "
            f"{row['com_l_peak']*1e3:7.1f} {row['com_vl_lift']:+8.3f} "
            f"{row['com_vl_land']:+8.3f} {row['dcm_err_l_eos']*1e3:+7.1f} "
            f"{row['corr_l']*1e3:+6.1f} {100*row['sat_frac']:5.0f} "
            f"{row['cop_edge_p95']:6.2f} {row['swing_z_land']*1e3:6.1f} "
            f"{row['swing_tilt_land']:5.1f} {row['land_err_f']*1e3:+7.1f}"
        )
    lines.append("")
    lines.append(
        "Columns: st=stance leg (0=L 1=R); |c| = peak |com_l|; "
        "vl_* = lateral CoM velocity at lift-off / touchdown; "
        "dcm_e = DCM_l − DCM_ref_l at end of step; corr = mean in-swing "
        "lateral foothold correction; sat% = CoM-ref clamp; cop95 = CoP "
        "edge ratio in single support (1=sole edge); z_mm = swing-foot "
        "height at last swing sample (near 0 = landed); tilt = sole tilt "
        "vs world +z at that sample (0 = flat); land_f = sagittal "
        "landing error vs the foothold."
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--speed", type=float, default=0.25)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--decimate", type=int, default=4,
                        help="Keep every Nth 500 Hz tick (default 4 → 125 Hz)")
    parser.add_argument("--out", type=Path, default=Path("logs/walk/last.npz"))
    parser.add_argument("--report", type=Path, default=None,
                        help="Print a report from an existing npz instead of running")
    parser.add_argument("--heading", type=float, default=0.0)
    parser.add_argument("--capture-gain", type=float, default=None)
    parser.add_argument("--step-period", type=float, default=None)
    parser.add_argument("--duty-factor", type=float, default=None)
    parser.add_argument("--stance-width", type=float, default=None)
    parser.add_argument("--scenario", default=None,
                        help="Replay a loka.walk_bench scenario by name")
    parser.add_argument("--stack", default=None)
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.report is not None:
        trace = load_trace(args.report)
        print(format_report(trace))
        return 0

    script = None
    pushes = None
    if args.scenario:
        from loka.walk_bench import scenarios

        match = [row for row in scenarios() if row.name == args.scenario]
        if not match:
            raise SystemExit(f"unknown scenario {args.scenario!r}")
        chosen = match[0]
        script = list(chosen.script)
        pushes = list(chosen.pushes)
        args.duration = chosen.duration
        print(f"recording scenario {chosen.name} for up to {args.duration:.1f}s …")

    gait = {"gait.heading": args.heading}
    if args.capture_gain is not None:
        gait["gait.capture_gain"] = args.capture_gain
    if args.step_period is not None:
        gait["gait.step_period"] = args.step_period
    if args.duty_factor is not None:
        gait["gait.duty_factor"] = args.duty_factor
    if args.stance_width is not None:
        gait["gait.stance_width"] = args.stance_width

    if script is None:
        print(f"recording walk at {args.speed:.2f} m/s for up to {args.duration:.1f}s …")
    from loka.control.locomotion import LocomotionConfig
    from loka.control.stacks import apply_stack_arg

    config = LocomotionConfig.from_yaml(args.config) if args.config else LocomotionConfig()
    apply_stack_arg(config, args)
    trace = record_walk(
        speed=args.speed,
        duration=args.duration,
        decimate=args.decimate,
        script=script,
        pushes=pushes,
        config=config,
        **gait,
    )
    save_trace(trace, args.out)
    print(f"wrote {args.out}")
    print(format_report(trace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
