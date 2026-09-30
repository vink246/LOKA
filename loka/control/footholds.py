"""Footstep policies the gait clock can drive.

``DcmFootholdPolicy`` is the historical chain: it calls the scheduler methods
that used to run inline, so ``legacy_dcm`` stays on the same code path.
``AlipStepFootholdPolicy`` keeps the chain and adds the ALIP stepping correction.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from loka.control.alip import plane_from_world, predicted_preimpact
from loka.control import gait as gait_mod
from loka.control.gait import (
    MODE_TREAD,
    MIN_FOOT_SEPARATION,
    SWING_CATCHUP_SPEED,
    heading_frame,
)


class FootholdPolicy(Protocol):
    def refresh_plan(self, sched, *, t_step: float, speed: float) -> None: ...

    def correct_target(self, sched, **kwargs) -> None: ...

    def com_reference(self, sched, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def on_touchdown(self, sched) -> None: ...


class DcmFootholdPolicy:
    """Englsberger DCM chain. The algorithm still lives on ``GaitScheduler``."""

    def refresh_plan(self, sched, *, t_step: float, speed: float) -> None:
        sched._refresh_plan(t_step=t_step, speed=speed)

    def correct_target(
        self, sched, *, com, com_vel, s, t_step, t_swing, t_ds, in_swing, height, **_ignored,
    ) -> None:
        omega = sched._omega(height)
        dcm = np.asarray(com, dtype=float) + np.asarray(com_vel, dtype=float) / omega
        if gait_mod.STEP_TIMING:
            sched._step_adjust(
                dcm=dcm, s=s, omega=omega, t_step=t_step,
                t_swing=t_swing, t_ds=t_ds, in_swing=in_swing,
            )
        elif in_swing:
            sched._apply_dcm_correction(dcm=dcm, s=s, omega=omega, t_step=t_step)

    def com_reference(self, sched, *, com, dt, height, t_step, t_ds, omega, **_ignored):
        dcm_ref = sched._dcm_at(omega=omega, t_step=t_step, t_ds=t_ds)
        sched._dcm_ref = dcm_ref
        sched._integrate_com_ref(dt=dt, omega=omega, dcm_ref=dcm_ref)
        vel = sched._com_vel_ref + sched._in_place_return(com)
        return sched._com_ref_output(com), vel, dcm_ref.copy()

    def on_touchdown(self, sched) -> None:
        return None


def _project_step(sched, pos: np.ndarray, s: float) -> np.ndarray:
    """Stride box, lateral gap, and the late-swing catch-up budget."""
    target = sched._steps.get(sched._step + 1)
    support = sched._steps.get(sched._step)
    if target is None or support is None or support.initial:
        return pos
    if s > 0.45:
        _, t_swing, _ = sched._durations()
        budget = SWING_CATCHUP_SPEED * max((1.0 - s) * t_swing, 1e-3)
        delta = pos - np.asarray(target.pos, dtype=float)
        norm = float(np.linalg.norm(delta))
        if norm > budget:
            pos = np.asarray(target.pos, dtype=float) + delta * (budget / norm)
    forward, left = heading_frame(float(support.yaw))
    side = 1.0 if target.leg == 0 else -1.0
    rel = pos - np.asarray(support.pos, dtype=float)
    stride = float(np.dot(rel, forward))
    limit = float(sched.config.step_length_max)
    excess = stride - float(np.clip(stride, -limit, limit))
    pos = pos - excess * forward
    rel = pos - np.asarray(support.pos, dtype=float)
    gap = side * float(np.dot(rel, left))
    if gap < MIN_FOOT_SEPARATION:
        pos = pos + side * (MIN_FOOT_SEPARATION - gap) * left
    return pos


#: Momentum used for the reference error:
#: "alip" is L about the stance contact, "com" is m H times CoM velocity.
ALIP_ERROR_MOMENTUM = "com"

#: Optional cap on the ALIP correction off the nominal step [m]. None leaves
#: only the kinematic step box.
ALIP_MAX_CORRECTION: float | None = None

#: Correction size [m] above which the gait treats capture as hot.
ALIP_HOT_CORRECTION = 0.10

#: Correction size [m] above which the step goes to the DCM capture nudge.
#: None disables it; 0.1 and 0.2 did not change the 10 N·s push results.
ALIP_HANDOFF_CORRECTION: float | None = None

class AlipStepFootholdPolicy:
    """Nominal chain step plus ALIP feedback on the predicted pre-impact error.

    The foot goes to the chain's nominal step plus
    ``capture_gain * (dx + dv / omega)`` of the error from the gait's CoM
    reference, carried to impact by the ALIP model. The correction is
    re-solved every tick of swing and bounded only by the step box, so a
    push or a lagging CoM moves the foot. Turning and treading in place use
    the DCM nudge. The convex MPC stays on the DCM CoM reference.
    """

    def __init__(self) -> None:
        #: Per-step yaw cap [rad]. None keeps the chain's own (legacy) cap;
        #: the DCM CoM reference was tuned against it.
        self.turn_cap = None

    @staticmethod
    def _gain(sched) -> float:
        override = getattr(sched, "alip_capture_gain", None)
        return float(sched.config.capture_gain if override is None else override)

    def _nominal_feedback_steps(
        self, sched, support, target, e_sag, e_lat, forward, left, mass, height
    ):
        """Nominal chain step plus the ALIP law on the predicted pre-impact error.

        ``e`` is measured minus the gait's CoM reference, carried to impact.
        Undisturbed, ``e`` is small and the foot goes where the chain put it,
        so normal walking is the legacy gait. A push or a lagging CoM grows
        ``e`` and the law moves the foot, bounded by the step box rather than
        a fixed correction clamp. ``L`` (not CoM velocity) carries the swing
        leg's and arms' momentum into the prediction.
        """
        mh = max(mass, 1e-3) * max(float(height), 0.05)
        # Capture-point shape on the predicted pre-impact error: the step moves
        # by capture_gain x (dx + dv / omega). The model's deadbeat/LQR gain
        # is ~2x hotter and, on a crawl gait that is two-thirds double
        # support, walked the robot backward into a fall.
        omega = float(np.sqrt(9.81 / max(float(height), 0.05)))
        k = self._gain(sched) * np.array([1.0, 1.0 / omega])
        nominal = np.asarray(target.nominal, dtype=float) - np.asarray(support.pos, dtype=float)
        es = np.array([float(e_sag[0]), float(e_sag[1]) / mh])
        el = np.array([float(e_lat[0]), float(e_lat[1]) / mh])
        corr = np.array([float(k @ es), float(k @ el)])
        norm = float(np.linalg.norm(corr))
        hot = ALIP_MAX_CORRECTION is not None and norm > ALIP_MAX_CORRECTION
        if hot:
            corr *= ALIP_MAX_CORRECTION / norm
        # Same flag the DCM nudge raises: the speed schedule holds while the
        # feet are busy catching the CoM.
        sched._capture_hot = bool(hot or norm > ALIP_HOT_CORRECTION)
        return (
            float(nominal @ forward) + corr[0],
            float(nominal @ left) + corr[1],
            ALIP_HANDOFF_CORRECTION is not None and norm > ALIP_HANDOFF_CORRECTION,
        )

    def refresh_plan(self, sched, *, t_step: float, speed: float) -> None:
        sched._refresh_plan(t_step=t_step, speed=speed)

    def correct_target(
        self, sched, *, com, com_vel, L_meas, s, t_step, t_ds, in_swing, height, **_ignored,
    ) -> None:
        com_vel = np.asarray(com_vel, dtype=float).reshape(-1)[:2]
        target = sched._steps.get(sched._step + 1)
        support = sched._steps.get(sched._step)
        if support is None:
            return
        mass = float(getattr(sched, "mass", 35.0))
        forward, left = heading_frame(float(support.yaw))
        L = np.zeros(3) if L_meas is None else np.asarray(L_meas, dtype=float)
        sag, lat = plane_from_world(com, support.pos, L, forward, left)
        if target is None or target.nominal is None or support.initial:
            return
        if target.frozen:
            return
        if s > float(np.clip(sched.config.foothold_retarget_s, 0.05, 0.95)):
            target.frozen = True
            return
        if in_swing and (
            getattr(sched, "_turning", False) or float(sched.config.mode) == MODE_TREAD
        ):
            # Turning and treading in place: the DCM nudge's error is measured
            # against the planned end-of-step DCM, which follows the chain.
            omega = sched._omega(height)
            dcm = np.asarray(com, dtype=float).reshape(2) + com_vel / omega
            sched._apply_dcm_correction(dcm=dcm, s=s, omega=omega, t_step=t_step)
            return
        if not in_swing:
            # Double support: hold the last foothold, as the DCM nudge does.
            return
        # Error from the gait's own CoM reference, in (x, L/(mH)), carried
        # to pre-impact by the ALIP model. The reference and measurement
        # share the double-support transfer, so it cancels.
        tau = float(sched._tau)
        t_ss = max(t_step - t_ds, 1e-3)
        mh = max(mass, 1e-3) * max(float(height), 0.05)
        ref_rel = np.asarray(sched._com_ref, dtype=float).reshape(2) - np.asarray(
            support.pos, dtype=float
        )
        v_ref = np.asarray(sched._com_vel_ref, dtype=float).reshape(2)
        errs = []
        for meas, axis in ((sag, forward), (lat, left)):
            if ALIP_ERROR_MOMENTUM == "com":
                # Both sides are CoM quantities, so the swing leg's and
                # arms' centroidal momentum is not read as a speed error.
                moment = mh * float(com_vel @ axis)
            else:
                moment = float(meas[1])
            e_now = np.array([
                float(meas[0]) - float(ref_rel @ axis),
                moment - mh * float(v_ref @ axis),
            ])
            errs.append(predicted_preimpact(
                e_now, t_in_step=tau, t_ss=t_ss, t_ds=t_ds, mass=mass, height=height,
            ))
        u_sag, u_lat, big = self._nominal_feedback_steps(
            sched, support, target, errs[0], errs[1], forward, left, mass, height
        )
        if big:
            # A large error is a capture, not a tracking error: the DCM
            # nudge re-anchors on the corrected foothold, while this law
            # keeps pulling toward the pre-push reference speed.
            omega = sched._omega(height)
            dcm = np.asarray(com, dtype=float).reshape(2) + com_vel / omega
            sched._apply_dcm_correction(dcm=dcm, s=s, omega=omega, t_step=t_step)
            return
        pos = np.asarray(support.pos, dtype=float) + u_sag * forward + u_lat * left
        pos = _project_step(sched, pos, s)
        target.correction = pos - np.asarray(target.nominal, dtype=float)
        target.pos = pos

    def com_reference(self, sched, *, com, dt, height, t_step, t_ds, omega=None, **_ignored):
        # The convex MPC was tuned against the DCM CoM reference. ALIP only
        # replaces where the feet go.
        om = sched._omega(height) if omega is None else omega
        dcm_ref = sched._dcm_at(omega=om, t_step=t_step, t_ds=t_ds)
        sched._dcm_ref = dcm_ref
        sched._integrate_com_ref(dt=dt, omega=om, dcm_ref=dcm_ref)
        vel = sched._com_vel_ref + sched._in_place_return(com)
        return sched._com_ref_output(com), vel, dcm_ref.copy()

    def on_touchdown(self, sched) -> None:
        return None
