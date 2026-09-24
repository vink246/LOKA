"""Telemetry compression and tracking-error evaluation for LOKA."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from loka.error_spec import ErrorSpec, ErrorTerm, get_tracking_error  # noqa: F401

TELEMETRY_WINDOW_S = 2.0
ANOMALY_COLLECTION_S = 1.0

@dataclass(frozen=True)
class TelemetrySections:
    """Toggle each telemetry block included in the LLM prompt."""

    section_0_state: bool = True
    section_1_cost: bool = True
    section_2_motor_load: bool = True
    section_3_directive: bool = True


DEFAULT_TELEMETRY_SECTIONS = TelemetrySections()


def _mean_term_value(frames, term: ErrorTerm) -> float | None:
    if not frames:
        return None
    return float(np.mean([term.read_frame(frame) for frame in frames]))


def _term_status(term: ErrorTerm, value: float) -> str:
    excess = term.excess(value)
    if excess <= 0.0:
        return "[NOMINAL]"
    if excess * term.weight > 0.5:
        return f"[ERR: {term.name.upper()} CRITICAL]"
    return f"[ERR: {term.name.upper()} OUT OF BAND]"


def format_tracked_state_section(anom_slice, nominal_buffer, has_baseline, error_spec: ErrorSpec):
    if not anom_slice or not error_spec.terms:
        return ""

    lines = [
        "0. TRACKED STATE (Error_Tracking terms)",
        "Values use each term's formula: offset + signal[index].",
        f"Window = mean over recent {TELEMETRY_WINDOW_S:.1f}s anomaly frames; "
        "Snapshot = value at trigger instant.\n",
    ]

    snapshot_frame = anom_slice[-1]
    for term in error_spec.terms:
        current = _mean_term_value(anom_slice, term)
        snap = term.read_frame(snapshot_frame)
        if current is None:
            continue

        status = _term_status(term, current)
        band = (
            f"mode={term.mode} target={term.target:.3g} "
            f"tol={term.tolerance:.3g} weight={term.weight:.3g}"
        )
        if has_baseline:
            nominal = _mean_term_value(nominal_buffer, term)
            if nominal is not None:
                lines.append(
                    f"- {term.name}: Nominal {nominal:.3g} | "
                    f"Current {current:.3g} | Snapshot {snap:.3g}  {status}"
                )
                lines.append(f"  ({band})")
                continue

        lines.append(
            f"- {term.name}: Current {current:.3g} | Snapshot {snap:.3g}  {status}"
        )
        lines.append(f"  ({band})")

    return "\n".join(lines) + "\n\n"


def extract_cost_proxies(buffer, error_spec: ErrorSpec | None = None):
    """Generic cost-ish proxies: Control effort + ErrorSpec term deviations from target."""
    if not buffer:
        return {"Control": 0.0}

    ctrl_arr = np.array([f["ctrl"] for f in buffer])
    proxies = {"Control": float(np.mean(np.abs(ctrl_arr)))}

    if error_spec is not None:
        for term in error_spec.terms:
            values = np.array([term.read_frame(frame) for frame in buffer])
            proxies[term.name] = float(np.mean(np.abs(values - term.target)))

    return proxies


def _actuator_capacity(model, actuator_id: int) -> float:
    gear = abs(float(model.actuator_gear[actuator_id, 0]))
    ctrl_limit = max(abs(float(x)) for x in model.actuator_ctrlrange[actuator_id])
    capacity = gear * ctrl_limit
    return capacity if capacity > 1e-9 else 1.0


def _mean_motor_loads(frames, model):
    """Per-actuator commanded |ctrl| vs realized |force|, capacity from *belief* gears."""
    if not frames:
        return {}

    force_stacks = []
    cmd_stacks = []
    for frame in frames:
        force = frame.get("actuator_torque")
        if force is None:
            force = frame.get("actuator_force")
        if force is None:
            continue
        force_stacks.append(np.abs(np.asarray(force, dtype=float)))
        cmd = frame.get("planner_cmd", frame.get("ctrl"))
        if cmd is None:
            cmd = np.zeros_like(force_stacks[-1])
        cmd_stacks.append(np.abs(np.asarray(cmd, dtype=float)))

    if not force_stacks:
        return {}

    stacked = np.stack(force_stacks, axis=0)
    cmd_stacked = np.stack(cmd_stacks, axis=0)
    mean_abs = np.mean(stacked, axis=0)
    peak_abs = np.max(stacked, axis=0)
    cmd_mean = np.mean(cmd_stacked, axis=0)
    cmd_peak = np.max(cmd_stacked, axis=0)

    loads = {}
    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        capacity = _actuator_capacity(model, actuator_id)
        mean_load = float(mean_abs[actuator_id])
        peak_load = float(peak_abs[actuator_id])
        loads[name] = {
            "mean": mean_load,
            "peak": peak_load,
            "cmd_mean": float(cmd_mean[actuator_id]),
            "cmd_peak": float(cmd_peak[actuator_id]),
            "util_mean": mean_load / capacity,
            "util_peak": peak_load / capacity,
        }
    return loads


def _format_motor_load_section(anom_slice, nominal_baseline, model, has_baseline):
    lines = [
        "2. MOTOR LOAD (Per-Actuator Force)",
        "Reports commanded |ctrl| vs realized |actuator_force| for each MJCF motor.",
        "Capacity / utilization use the planner's believed |gear| * max|ctrlrange|.",
        "Commanded effort with near-zero force means the plant is not delivering torque.",
        "",
    ]

    anom_loads = _mean_motor_loads(anom_slice, model)
    if not anom_loads:
        lines.append("- No actuator_force samples available in this window.\n")
        return "\n".join(lines) + "\n"

    nom_loads = _mean_motor_loads(nominal_baseline, model) if has_baseline else {}

    for name, stats in anom_loads.items():
        if has_baseline and name in nom_loads:
            nom = nom_loads[name]
            lines.append(
                f"- {name}: Nominal cmd {nom['cmd_mean']:.2f} force {nom['mean']:.2f} "
                f"(util {nom['util_mean']:.0%}) | "
                f"Current cmd {stats['cmd_mean']:.2f} force {stats['mean']:.2f} "
                f"(util {stats['util_mean']:.0%}) | "
                f"Peak force {stats['peak']:.2f} (util {stats['util_peak']:.0%})"
            )
        else:
            lines.append(
                f"- {name}: Current cmd {stats['cmd_mean']:.2f} force {stats['mean']:.2f} "
                f"(util {stats['util_mean']:.0%}) | "
                f"Peak force {stats['peak']:.2f} (util {stats['util_peak']:.0%})"
            )

    lines.append("")
    return "\n".join(lines) + "\n"


def _format_cost_section(anom_costs, nom_costs, has_baseline):
    lines = ["1. COST / DEVIATION LANDSCAPE"]
    for cost_name, a_val in anom_costs.items():
        if has_baseline and cost_name in nom_costs:
            n_val = nom_costs[cost_name]
            if a_val > (n_val * 2.0 + 0.05):
                status = "[ERR: CRITICAL DEVIATION]"
            elif a_val > (n_val * 1.3 + 0.02):
                status = "[ERR: MODERATE DEVIATION]"
            else:
                status = "[NOMINAL]"
            lines.append(
                f"- {cost_name}: Nominal {n_val:.2f} | Current {a_val:.2f}  {status}"
            )
        else:
            lines.append(
                f"- {cost_name}: Nominal [Unknown] | Current {a_val:.2f}  [NO BASELINE]"
            )
    return "\n".join(lines) + "\n\n"


def _format_directive_section(sections: TelemetrySections):
    sources = []
    if sections.section_0_state:
        sources.append("tracked state")
    if sections.section_1_cost:
        sources.append("cost / deviation landscape")
    if sections.section_2_motor_load:
        sources.append("motor load")

    if not sources:
        source_text = "available telemetry"
    elif len(sources) == 1:
        source_text = sources[0]
    else:
        source_text = ", ".join(sources[:-1]) + f", and {sources[-1]}"

    lines = [
        "3. SYSTEM ORCHESTRATOR DIRECTIVE",
        f"Diagnose using the {source_text}. "
        "Update the YAML scratchpad to rewrite MPC parameters and, if the plant "
        "or environment no longer matches the nominal model, Model_Mutations "
        "(MPC belief: gear / friction / mass / com) and Error_Tracking.\n",
    ]
    return "\n".join(lines) + "\n"


def synthesize_generalized_telemetry(
    nominal_baseline,
    anomaly_buffer,
    model,
    sections: TelemetrySections = DEFAULT_TELEMETRY_SECTIONS,
    error_spec: ErrorSpec | None = None,
):
    if not anomaly_buffer:
        return "Error: No anomaly buffer available."

    print("\n" + "=" * 55)
    print("[SYSTEM] EXECUTING TELEMETRY COMPRESSION")
    print("=" * 55)

    recent_frames = int(TELEMETRY_WINDOW_S / model.opt.timestep)
    anom_slice = (
        list(anomaly_buffer)[-recent_frames:]
        if len(anomaly_buffer) > recent_frames
        else list(anomaly_buffer)
    )

    nominal_baseline = list(nominal_baseline or [])
    has_baseline = len(nominal_baseline) > 0

    prompt = "--- TELEMETRY ANOMALY REPORT ---\n"
    if has_baseline:
        prompt += (
            "The MPC is not behaving nominally. Compare the Nominal Baseline to the Current Anomaly.\n\n"
        )
        nom_costs = extract_cost_proxies(nominal_baseline, error_spec)
    else:
        prompt += (
            "CRITICAL: Cold Start Failure. No nominal baseline exists. "
            "Evaluating against absolute Error_Tracking limits.\n\n"
        )
        nom_costs = {}

    if sections.section_0_state and error_spec is not None:
        prompt += format_tracked_state_section(
            anom_slice, nominal_baseline, has_baseline, error_spec
        )

    if sections.section_1_cost:
        anom_costs = extract_cost_proxies(anom_slice, error_spec)
        prompt += _format_cost_section(anom_costs, nom_costs, has_baseline)

    if sections.section_2_motor_load:
        prompt += _format_motor_load_section(
            anom_slice, nominal_baseline, model, has_baseline
        )

    if sections.section_3_directive:
        prompt += _format_directive_section(sections)

    return prompt
