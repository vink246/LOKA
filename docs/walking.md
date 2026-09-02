# Phase B: the walking plant

Status: **the shuffle is fixed, balance is not.** The footstep plan and the DCM
reference are correct, sagittal tracking holds to a few centimetres, and the
swing foot executes its planned arc. Lateral CoM error compounds over roughly
8-16 steps until the robot topples sideways.

Standing is unaffected (8 s hold, 0.01° peak tilt, 0.1 mm mean CoM error).

Run `python -m loka.verify_gait` for current numbers. The pass/fail gates are in
`tests/test_gait.py`; the balance defect is a documented `xfail` there rather
than a loosened threshold.

## What was wrong originally

The robot stayed upright by shuffling: 0.019 m/s against 0.25 m/s commanded
(7.6%), with 2.9 mm of foot clearance against a 38.6 mm planned arc.

Five distinct defects, each individually sufficient to prevent walking:

1. **No term commanded forward progress.** The CoM reference was slaved to the
   measured foot midpoint while footholds were placed relative to the measured
   CoM. That loop is closed entirely on measurement: it has a fixed point
   wherever the robot happens to be standing, so "walk at 0.25 m/s" had no
   representation anywhere in it.
2. **The swing foot had no actuator.** `weight_swing_foot` was `0.0`, disabling
   the Cartesian foot task outright, and `stand.py` additionally zeroed
   `swing_acc[2]`. Footholds were computed and then never tracked; the leg
   moved only through a joint feedforward.
3. **The MPC braked every step.** `CentroidalReference.com_velocity` defaulted
   to zero while `weight_linear_velocity` was 60, so the force plan actively
   opposed whatever forward motion did occur.
4. **The MPC had no contact schedule.** The measured contact set was replicated
   across a 0.3 s horizon that spans most of a step, so the plan braced against
   feet that were in the air.
5. **Ad-hoc brake heuristics** in four places suppressed the residual motion,
   which is what made the shuffle look stable.

## Current architecture

`loka/control/gait.py` plans; `loka/control/stand.py` executes.

Footsteps form a **chain anchored at the current support foot**. Each link
advances by `speed × T_step` along the heading and crosses `stance_width` to the
other side. Commanded progress is therefore built into the geometry: the plan's
velocity is the commanded velocity by construction, whatever the robot does,
while the chain starts where the robot actually stands so footholds stay
reachable. Position error and speed error are decoupled — the plan absorbs an
overshoot in position and still commands the right speed.

The CoM reference is the DCM trajectory implied by that chain, treating each
footstep as a piecewise constant ZMP held for one step:

    ξ(τ)      = p_k + (ξ_eos,k − p_k) · e^{ω₀(τ − T_step)}
    ξ_eos,k−1 = p_k + (ξ_eos,k − p_k) · e^{−ω₀ T_step}
    ċ_ref     = ω₀ (ξ_ref − c_ref)

Measured state enters in exactly two bounded places: a clamped foothold
correction, and a saturation on the CoM tracking error handed to the QP. Neither
can cancel the feedforward.

Verified in isolation (`verify_gait.py` section 1, robot replaced by perfect
tracking): forward velocity settles on the command to within 0.3% at every speed
from 0.05 to 0.40 m/s; lateral motion is a bounded limit cycle with CoM sway
±22.9 mm and DCM sway ±69.6 mm, centred on the midline to within 0.3 mm. The
±69.6 mm matches the closed form `0.575 · stance_width/2` for `ω₀T = 1.31`.

## Bugs found and fixed while getting here

Worth recording because three of them are subtle and two are the same mistake
in different clothing.

- **The CoM-reference leash was positive feedback.** Clamping the reference to
  stay within 8 cm of the measured CoM meant a robot running ahead *dragged the
  target with it* — the reference reached 0.71 m when the plan said 0.31 m — and
  `com_vel_ref`, differentiated from that contaminated integrator, swung to
  −1.4 m/s. Fixed by keeping the integrator on plan state only and saturating
  just the error handed downstream. The plan may wait for the robot; it must
  never chase it.
- **A lateral ratchet in the foothold clamp.** The reachability clamp rebuilt
  the foothold as `support_foot + offset`, re-anchoring each step to the
  previous foot, so any lateral bias compounded: successive left footholds went
  +0.117, +0.178, +0.207, +0.217 m. A constraint must subtract only its own
  violation, never re-anchor.
- **The foothold correction was discarded at touchdown.** `_refresh_plan`
  rewrites `pos = nominal` every tick, and the freeze branch returned before
  re-applying the correction. The foot tracked the capture point for 70% of
  swing and then snapped back to nominal — landing up to 0.9 m behind the
  capture point, which *accelerates* divergence. Fixed by storing the
  correction on the `Footstep` instead of recomputing it.
- **No swing acceleration feedforward.** Asked to synthesise a 0.25 s arc from
  position error alone, `kp = 400` against a 20 m/s² clip saturates at 5 cm of
  error, so the task was bang-bang. The arc is analytic, so it now supplies its
  own velocity *and* acceleration and the gains only mop up residual error.
- **The swing leg's null space was undamped.** Dropping its joint posture to
  weight 0.5 left three redundant DOF regularised only by a 1e-6 ridge; near
  knee extension the leg Jacobian is ill-conditioned and the QP flung the foot
  0.7 m into the air. Now damping-only rows, as the planted legs already used.

## The open defect

Lateral CoM error compounds. The mechanism, from the traces:

Entering the first step the plan carries roughly half the limit-cycle lateral
velocity (−0.13 against −0.26 m/s), because the DCM reference is the steady
limit cycle from the first tick while the CoM reference starts wherever the
robot is standing and converges with a time constant of `1/ω₀ ≈ 0.27 s`. The CoM
therefore fails to travel far enough over the stance foot, leaves single support
with a surplus of outward velocity, and — since `c̈_y = ω₀²(c_y − p_y)` grows with
displacement — the next step pushes it further out still.

`OPENING_TRANSFER_STEPS` lengthens the initial double-support transfer to
address the entry condition. That removed the systematic bias toward the first
swing leg (drift direction is now mixed rather than always leftward) but did not
stop the compounding.

Ruled out by experiment, so as not to be re-litigated:

| Hypothesis | Result |
| --- | --- |
| Double-support ZMP mismatch (piecewise-constant ZMP over a 30% DS window) | Disconfirmed — `duty=0.52`, near-zero DS, is *worse* |
| MPC contact schedule | Neutral to mildly helpful; removing it does not fix it |
| MPC moving velocity reference | Clearly helpful — removing it makes the robot walk *backwards* |
| Cadence, duty, stance width, capture gain | All fall in 2-4 s across the swept envelope |
| Predictive touchdown-DCM correction instead of instantaneous | No improvement, possibly worse (amplifies error through `e^{ω₀ Δt}`) |
| Base orientation gain | Minor effect on roll amplitude, no effect on divergence |

## Where to look next

1. **Lateral foot placement authority is the prime suspect.** The correction is
   currently a single bounded offset from nominal, applied to the in-flight
   foothold. Textbook lateral stabilisation also adjusts **step timing** —
   landing early is a far stronger lateral correction than landing wide, because
   it cuts the divergent phase short. `T_step` is presently rigid.
2. **`MAX_COM_REF_ERROR` saturates the 2D error by norm**, so a legitimate
   sagittal lead shrinks the lateral component too. Saturating per-axis in the
   heading frame would decouple them.
3. **The lateral CoM reference amplitude is only ±23 mm** against a ±119 mm foot
   separation. Worth checking against a reference implementation whether that is
   right for `duty=0.65`, and whether the swing foot should be commanded to a
   *wider* stance during the transient.
4. **Instrument the CoP against the stance foot edge.** The traces showed the
   CoP tracking the stance foot centre to ±0.02 m, so the QP appears to be
   delivering the requested ZMP — but the foot is only 0.06 m wide, and it is
   worth confirming the lateral CoP is not saturating at the edge during the
   compounding phase.
