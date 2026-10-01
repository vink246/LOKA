"""Early plant-health anomaly clues for standing LOKA.

``Error_Tracking`` remains the LLM-editable *mission* criteria.
This module scores *balance distress* so LOKA can intervene before a fall.

Important: intentional Task_Targets changes (crouch / lean) leave CoM tracking
residuals while the stance stays healthy. Those must **not** open a failure
episode. CoM error alone is therefore only a trigger when paired with distress
(tilt, margin, slip, force mismatch, strain, QP fallback) or a clear
**lateral** bias (asymmetric mass).

While ``walking`` is true, planned swing produces contact motion, force
mismatch, and brief single support — those are **not** faults.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Quiet stand: CoM error ~0.2 mm. Mild shoulder mass / push: tens of mm + tilt.
COM_ERR_WARN_M = 0.008
COM_ERR_STRONG_M = 0.025
TILT_WARN_RAD = np.deg2rad(0.6)
TILT_STRONG_RAD = np.deg2rad(2.0)
MARGIN_WARN_M = 0.035
FORCE_MISMATCH_WARN_N = 40.0
CONTACT_SPEED_WARN = 0.08
UTIL_WARN = 0.85
MIN_CONTACTS_HEALTHY = 6

# Walk: looser plant-health gates (swing + SS are normal).
WALK_COM_LAT_STRONG_M = 0.08
WALK_FORCE_MISMATCH_WARN_N = 220.0
WALK_CONTACT_SPEED_WARN = 0.45
WALK_MIN_CONTACTS_HEALTHY = 3
WALK_TILT_WARN_RAD = np.deg2rad(1.5)
WALK_TILT_STRONG_RAD = np.deg2rad(4.0)

# Dispatch when score exceeds this after the collection window.
EARLY_ANOMALY_TRIGGER = 0.12


@dataclass(frozen=True)
class AnomalyAssessment:
    score: float
    clues: tuple[str, ...]
    triggered: bool
    balance_healthy: bool = True

    def tag_line(self) -> str:
        if not self.clues:
            return ""
        return "  early clues: " + " ".join(self.clues)


def _balance_healthy(
    *,
    tilt: float,
    margin_min: float,
    contact_speed: float,
    force_err: float,
    peak_util: float,
    n_contacts: int,
    wbc_fail: int,
    mpc_fail: int,
    walking: bool,
) -> bool:
    tilt_warn = WALK_TILT_WARN_RAD if walking else TILT_WARN_RAD
    force_warn = WALK_FORCE_MISMATCH_WARN_N if walking else FORCE_MISMATCH_WARN_N
    speed_warn = WALK_CONTACT_SPEED_WARN if walking else CONTACT_SPEED_WARN
    min_c = WALK_MIN_CONTACTS_HEALTHY if walking else MIN_CONTACTS_HEALTHY
    return (
        tilt < tilt_warn
        and margin_min >= MARGIN_WARN_M
        and contact_speed < speed_warn
        and force_err < force_warn
        and peak_util < UTIL_WARN
        and n_contacts >= min_c
        and wbc_fail == 0
        and mpc_fail == 0
    )


def assess_stand_anomaly(frame: dict) -> AnomalyAssessment:
    """Score one telemetry frame for early non-nominal plant behavior."""
    clues: list[str] = []
    distress_score = 0.0
    tracking_score = 0.0
    walking = bool(frame.get("walking", False))

    com_error = np.asarray(frame.get("com_error", np.zeros(3)), dtype=float)
    # Vertical CoM residual is expected during crouch / height tracking — ignore for trigger.
    com_xy = float(np.linalg.norm(com_error[:2]))
    com_lat = float(abs(com_error[1]))
    com_fore = float(abs(com_error[0]))
    rpy = np.asarray(frame.get("rpy", np.zeros(3)), dtype=float)
    tilt = float(np.linalg.norm(rpy[:2]))
    margin = np.asarray(frame.get("support_margin", (0.1, 0.1, 0.1, 0.1)), dtype=float)
    margin_min = float(np.min(margin[:4])) if margin.size >= 4 else 0.1
    des = np.asarray(frame.get("desired_forces", np.zeros((1, 3))), dtype=float)
    got = np.asarray(frame.get("contact_forces", np.zeros((1, 3))), dtype=float)
    force_err = (
        float(np.linalg.norm(des - got)) if des.shape == got.shape else 0.0
    )
    contact_speed = float(frame.get("contact_speed_max", 0.0))
    util = frame.get("joint_util")
    peak_util = float(np.max(util)) if util is not None and len(util) else 0.0
    mask = np.asarray(frame.get("contact_mask", ()), dtype=bool)
    n_contacts = int(mask.sum()) if mask.size else 8
    # Prefer per-tick deltas; cumulative counters permanently arm QP_FALLBACK.
    wbc_fail = int(frame.get("wbc_failures", 0) or 0)
    mpc_fail = int(frame.get("mpc_failures", 0) or 0)

    tilt_warn = WALK_TILT_WARN_RAD if walking else TILT_WARN_RAD
    tilt_strong = WALK_TILT_STRONG_RAD if walking else TILT_STRONG_RAD
    force_warn = WALK_FORCE_MISMATCH_WARN_N if walking else FORCE_MISMATCH_WARN_N
    speed_warn = WALK_CONTACT_SPEED_WARN if walking else CONTACT_SPEED_WARN
    lat_strong = WALK_COM_LAT_STRONG_M if walking else COM_ERR_STRONG_M
    lat_warn = 0.04 if walking else COM_ERR_WARN_M

    healthy = _balance_healthy(
        tilt=tilt,
        margin_min=margin_min,
        contact_speed=contact_speed,
        force_err=force_err,
        peak_util=peak_util,
        n_contacts=n_contacts,
        wbc_fail=wbc_fail,
        mpc_fail=mpc_fail,
        walking=walking,
    )

    # -- Distress (always eligible to trigger) ------------------------------
    if tilt >= tilt_strong:
        clues.append("[TILT_HIGH]")
        distress_score += 0.30
    elif tilt >= tilt_warn:
        clues.append("[TILT_ELEVATED]")
        distress_score += 0.12 if not walking else 0.06

    if margin_min < MARGIN_WARN_M:
        clues.append("[SUPPORT_MARGIN_LOW]")
        distress_score += 0.20

    if force_err >= force_warn:
        clues.append("[FORCE_MISMATCH]")
        distress_score += 0.15 if not walking else 0.08

    # Planned swing moves sole sites; ignore mild contact speed while walking.
    if contact_speed >= speed_warn:
        clues.append("[CONTACT_SLIP]")
        distress_score += 0.18 if not walking else 0.05

    if peak_util >= UTIL_WARN:
        clues.append("[ACTUATOR_STRAIN]")
        distress_score += 0.12

    if wbc_fail > 0 or mpc_fail > 0:
        clues.append("[QP_FALLBACK]")
        # Only a burst of failures this tick counts; walking often has 1 soft fail.
        if walking:
            distress_score += 0.08 if (wbc_fail + mpc_fail) >= 2 else 0.0
        else:
            distress_score += 0.25

    # -- Tracking residuals (gated) -----------------------------------------
    if com_lat >= lat_strong:
        side = "LEFT" if com_error[1] > 0 else "RIGHT"
        clues.append(f"[COM_BIAS_{side}]")
        tracking_score += 0.22 if not walking else 0.12
    elif com_lat >= lat_warn:
        side = "LEFT" if com_error[1] > 0 else "RIGHT"
        clues.append(f"[COM_BIAS_{side}]")
        tracking_score += 0.10 if not walking else 0.04

    if not healthy:
        if com_xy >= COM_ERR_STRONG_M:
            clues.append("[COM_ERROR_HIGH]")
            tracking_score += 0.30 if not walking else 0.10
        elif com_xy >= COM_ERR_WARN_M:
            clues.append("[COM_ERROR_ELEVATED]")
            tracking_score += 0.12 if not walking else 0.04
        if com_fore >= COM_ERR_WARN_M and not walking:
            fore = "FORWARD" if com_error[0] > 0 else "BACK"
            clues.append(f"[COM_BIAS_{fore}]")
            tracking_score += 0.08
    elif com_xy >= COM_ERR_STRONG_M:
        clues.append("[COM_TRACKING_RESIDUAL]")

    score = distress_score + tracking_score
    trigger_bar = 0.22 if walking else EARLY_ANOMALY_TRIGGER

    triggered = (
        distress_score >= trigger_bar
        or (com_lat >= lat_strong and tracking_score >= trigger_bar)
        or (not healthy and score >= trigger_bar)
    )

    seen: set[str] = set()
    uniq = []
    for c in clues:
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    return AnomalyAssessment(
        score=float(score),
        clues=tuple(uniq),
        triggered=bool(triggered),
        balance_healthy=healthy,
    )


def max_anomaly_over_frames(frames: Sequence[dict]) -> AnomalyAssessment:
    best = AnomalyAssessment(0.0, (), False, True)
    for frame in frames:
        assessment = assess_stand_anomaly(frame)
        if assessment.score > best.score or (
            assessment.triggered and not best.triggered
        ):
            best = assessment
    return best
