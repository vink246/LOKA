# Phase B: the walking plant

Two stacks, one contract (`compute_torque` never blocks). Select with `--stack`:

| Stack | Footholds | Force / torque |
|---|---|---|
| `legacy_dcm` (default) | DCM capture point | ConvexMPC + WBC |
| `alip_footstep` | ALIP stepping law (Xiong & Ames) on the same chain | same ConvexMPC + WBC |

Status of `legacy_dcm`: **8 s at 0.10–0.30 m/s holds. 30 s at 0.10 m/s holds; 30 s at 0.20 m/s stays up but drifts about 0.56 m sideways, so that limit-cycle bound fails.**
with the speed-scheduled high-DS gait (`gait_schedule_for_speed`, duty
~0.82 crawl → ~0.74 at 0.50 m/s). Isolation duty 0.65 still pitches over
at ~3 s — do not pin the yaml 0.65 default when asking for a walk. CoP
still runs near the sole edge (`cop95 ≈ 0.9`); the extra double-support
is what leaves enough spare to reject tracking error. The 8 s / 30 s
gates also require speed tracking and bounded lateral travel; pelvis-z
alone is not a walk.

Standing is unaffected (8 s hold, 0.01° peak tilt, 0.1 mm mean CoM error).

Run `python -m loka.verify_gait` for current numbers. The pass/fail gates are in
`tests/test_gait.py`.

## Architecture

```text
LLM  (0.5–5 s YAML)     gait.* Task_Targets only — never footholds
        │
        ▼
GaitScheduler  (500 Hz)  contacts, DCM / CoM xy, swing Bézier
        │                Englsberger DCM; ZMP inset 12 mm (Kajita / LIPM)
        ▼
ConvexMPC      (50 Hz)   finite-foot forces, 0.3 s horizon, OSQP QP
        │
        ▼
WBC            (500 Hz)  29 torques; the only layer that tracks the swing arc
```

The LLM brain stays a slow policy layer (`system_prompt.txt`). Locked QP costs
(`mpc.weight_*`, `wbc.weight_contact`, `mpc.cop_margin`) are not on the
allowlist. If the LLM only sets `gait.speed`, cadence and stance width follow
`gait_schedule_for_speed` so 0.10 and 0.50 m/s share one controller.

### Papers this plant is built from

| Layer | Paper | What we took |
|---|---|---|
| Convex SRBD QP | Di Carlo et al., *Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control*, IROS 2018 | State `x = [θ, p, ω, v, g] ∈ R¹³`, condensed QP, friction pyramids, 0.3 s horizon |
| Centroidal MPC, 6-DoF contacts | Sleiman, Farshidian, Minniti, Hutter, *A Unified MPC Framework for Whole-Body Dynamic Locomotion and Manipulation*, TRO 2021 / [arXiv:2103.00946](https://arxiv.org/abs/2103.00946) | CoP-in-sole instead of independent point feet; contact pose in the horizon (here: gait Bézier, not full joint kinematics) |
| Short-horizon NMPC with a gait reference | Galliker et al., *Bipedal Locomotion with Nonlinear Model Predictive Control*, Humanoids 2022 / [arXiv:2203.07429](https://arxiv.org/abs/2203.07429) | Cited for the *lesson* (a good reference gait is what makes 0.2–0.3 s work), not the solver. Ours is DCM + Bézier, not HZD. `ConvexMPC` is still Di Carlo's convex QP. We did **not** take the full-order torque NMPC or ocs2 — that would replace the WBC and the LLM/QP split. Implementation reference: [wb_humanoid_mpc](https://github.com/manumerous/wb_humanoid_mpc) (centroidal G1 example, not `launch-wb-g1-dummy-sim`) |
| DCM / capture point | Englsberger et al., *Three-dimensional bipedal walking control based on divergent component of motion*, IROS 2011; Pratt et al., *Capture Point*, Humanoids 2006 | Backward DCM recursion, clamped capture-point foothold correction |
| Whole-body QP | Herzog, Rotella, Mason, Grimminger, Schaal, Righetti, *Momentum control with hierarchical inverse dynamics on a torque-controlled humanoid*, Auton. Robots 2016 (and the OSQP WBC lineage used on ANYmal / MIT Cheetah) | Contact + posture + swing tasks at 500 Hz |

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

`loka/control/gait.py` plans; `loka/control/locomotion.py` executes.

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
- **The foothold correction was measured against a goal it had just moved.**
  `_refresh_plan` chains the tail off the corrected foothold, so the
  end-of-step DCM that the correction was compared with already included the
  previous tick's correction. Each tick computed `c = k (e − c_prev)` and
  settled at `k/(1+k)` of the error (0.375 at the old default 0.6). At
  `k ≥ 1` it oscillated and straight walking fell. The error is now the DCM
  offset past the target foothold, which that chain shift does not move, and
  `capture_gain` is the fraction of full placement (default 0.7).
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

## The remaining defect (and what closed the fall)

Duty was the lever. Isolation / yaml 0.65 at 0.20 m/s still dies at ~3 s
(CoP on the sole edge, then sagittal `land_f` grows). Pinning
`duty_factor ≈ 0.80` (more double-support, shorter single-support
exponential) is a bounded walk: 8 s and 30 s gates pass under
`gait_schedule_for_speed`. Duty 0.75 at the same period still falls at
~4.5 s; 0.78 holds 8 s but died at 19 s; 0.80 held 30 s.

What is *not* closed: CoP still runs at the edge (`cop95 ≈ 0.86–0.91` even
on a 30 s cycle); 0.50 m/s tracks ~68% of command with ~12 mm sagittal
short-plant; peak clearance is ~22 mm against a 45 mm Bézier (a real step,
not the old 3 mm shuffle). Extra DS is spare *time*, not spare *geometry*.

The opening ZMP sits on the upcoming swing-foot inset (APA). Each walking
step's ZMP is 12 mm inside the sole, toward the midline; the foothold itself
is still the foot centre. First-lift `v_l` dropped from ≈ −0.12 m/s (duty
0.65) to ≈ −0.02 m/s (duty 0.80) without touching `walk_accel`.

Ruled out by experiment, so as not to be re-litigated:

| Hypothesis | Result |
| --- | --- |
| Double-support ZMP mismatch (piecewise-constant ZMP over a 30% DS window) | Disconfirmed — `duty=0.52`, near-zero DS, is *worse* |
| MPC contact schedule | Neutral to mildly helpful; removing it does not fix it |
| MPC moving velocity reference | Clearly helpful — removing it makes the robot walk *backwards* |
| Cadence / duty below ~0.78 at 0.20 m/s | Still falls (0.65 at ~3 s, 0.75 at ~4.5 s). Duty ~0.80 is the walk |
| Predictive touchdown-DCM correction instead of instantaneous | No improvement, possibly worse (amplifies error through `e^{ω₀ Δt}`) |
| Base orientation gain | Minor effect on roll amplitude, no effect on divergence |
| `MAX_COM_REF_ERROR` 2-D clamp starving lateral | Disconfirmed as the *trigger* — clamp is 0% of samples through step 3, then a consequence of the crash |
| Capture-point clamp (`MAX_DCM_CORRECTION`) | 6% of samples, all after step 4; not the start |

## Where to look next

1. **Right-leg swing tracking on the way down.** The gait clock now waits
   until the swing-foot centre is within 8 mm of the ground, a sole site
   is in contact, *and* (for a swing that actually left the ground) the
   centre is within `TOUCHDOWN_RADIUS` (30 mm, the sole half-width) of the
   planned foothold (`TOUCHDOWN_HEIGHT` / `MAX_TOUCHDOWN_HOLD` in `gait.py`).
   A short scuff is not named as stance. Ankle pitch/roll are released after
   `s = 0.7` (`SWING_ANKLE_RELEASE_S`) and a world-flat sole orientation task
   takes over in that window so the four sites arrive together. Swing
   Cartesian / orientation tasks drop once the foot is accepted into the
   contact mask (same radius). Clock hold + limp ankles took touchdown height
   from 38 mm (right) to ~6 mm; the orientation task is the attempt to plant
   a flat sole rather than a flop or a toe.
2. **Short plant used to leave a phantom ZMP.** Traces landed 16–32 mm short
   of the first foothold; the clock then chained the next step from the
   *planned* support. On commit the support is now re-anchored at the
   measured plant. The leftover scrape (8 mm peak vs a 60 mm Bézier) was
   the WBC: stance contact is weighted 2000 and swing was 60, so the QP
   inverted the lift (`a*_z > 0`, realised `a_z < 0`). Swing weight 400
   (and `kp/kd` 500/40 to close the last centimetre) is the actuator; first
   plants are now on the foothold. Later `land_f` growth is a remaining
   tracking/capture problem, not a planner lie.
3. **Step timing as a stabilizer** is in both directions: the clock waits
   for a late or short foot and commits early once a swing that actually left
   the ground has seated near the foothold (`EARLY_PLANT_S`, same test as
   the late hold). A hop below `SWING_CLEARED_HEIGHT` (20 mm) is a scuff, not
   a swing — otherwise a 9 mm bounce early-plants at `s = 0.75` and cuts off
   the lagged Cartesian lift. Isolation tests keep z = 0, so they never look
   like a real swing and keep the nominal cadence.
4. **CoP spare authority.** The force plan keeps CoP in the *physical* sole
   (Sleiman); the 12 mm inset is only on the gait ZMP reference, so the outer
   sole is available for tracking. Dropping the Cartesian swing task on a
   pitched toe scuff (centre 11 mm up, 21 mm short, `s = 0.83`) was the
   remaining 0.20 m/s cascade — that accept now also requires the foot
   centre down. Capture corrections that outrun the remaining Bézier are
   clamped (Galliker: foot pose at contact is kinematic).
5. **Stance knee vault.** Joint traces showed the planted knee slamming
   through straight (18° → −10° in 32 ms) at first lift-off, 1.2 s before
   pitch grew. A light keyframe bias plus a one-sided kick below 15°
   (`STANCE_KNEE_*`) guards the stop without pinning the pelvis. Hip/ankle
   planted gains stay at zero.
6. **CoM-height vault.** Even with a bent stance knee the height task
   (WBC `kp` on z, MPC `weight_position[2]`) was asking single support to
   hold stand height. Both now scale with the planted-contact fraction, so
   standing is unchanged and single support is allowed to sit rather than
   vault. No crouch offset.

## Turning and transitions

`gait.heading` is a goal, not an instantaneous travel frame. The footstep
chain is a centre line that yaws by at most `TURN_CAP_*` (0.12 rad) per step
toward that goal, and forward speed is cut to 0.10 m/s while the heading
error is large. Once that cut has engaged (`TURN_SETTLE`), it stays on until
the foot-yaw error has been under 0.10 rad for two step boundaries and the
measured speed is back near the crawl. The stride is still the commanded
speed; the measurement only delays the return to it. Step period, duty,
stance width and swing height are latched
and slewed only at step boundaries. `gait.mode=stand` finishes the airborne
step instead of resetting mid-swing. Tread marches in place; limp is a
conservative cadence preset, not an asymmetric gait.

`python -m loka.walk_bench` is the pass/fail matrix. On the yaw-aware plant:

| Scenario | Result |
| --- | --- |
| T1 heading +0.3 rad at 0.20 m/s | pass |
| T2 90° at 0.20 m/s | pass (was a fall at 12.3 s, after the heading was already met) |
| T3 180° at 0.20 m/s | pass |
| T4 90° at 0.40 m/s | stays up; speed 0.20 m/s misses the 0.24 bar |
| T5 turn-in-place, T6 ramp | T5 passes. T6 stays up and misses the heading deadline |
| G2 cadence / width / swing height mid-walk | pass |
| G3 stop mid-swing, then walk again | pass |
| G4 crouch while walking | pass |
| G5b limp | pass |
| G1 speed step 0.10→0.50, G6 combined turn+speed | G1 passes. G6 stays up; speed 0.21 m/s misses the 0.24 bar |
| G5 tread | stays up (was a fall at ~4.3 s) but the later walk ends at −0.19 m/s |
| S0 straight 0.20 m/s for 30 s | bench passes (no fall, speed holds). The pytest limit cycle also bounds lateral travel, and this walk drifts ~0.56 m |
| P1/P2 pushes while walking or turning | no falls. The misses are backward pushes: 8 N·s late in swing, and 10–12 N·s, which do not get back to 0.12 m/s. The old swing-clock speed-up never runs: scheduled swing time stays under 0.16 s, and the speed-up requires 0.20 s |

Ruled out while getting here: a 0.25 rad/step cap (T2 fell earlier, roll
~4°), and duty 0.65 on a turn-in-place (the crawl schedule has to apply at
zero speed). Slowing the cap to 0.08 rad/step lets a 90° turn reach the
heading and then fall as speed recovers. That is a balance problem, not a
missing yaw reference. `TURN_SETTLE` is the hold that stops that recovery
from starting until the feet have caught the heading.

Also ruled out, from a pass at [wb_humanoid_mpc](https://github.com/manumerous/wb_humanoid_mpc) (ideas only):

| Idea | Result |
| --- | --- |
| Pelvis yaw only during double support (`YAW_IN_DS_ONLY`) | T1 fell at 6.3 s. The swing foot still yaws through single support, so freezing the pelvis fights it. Flag left off. |
| Schedule cadence from measured speed (`SPEED_HYSTERESIS`) | G1 already passes without it. Not what made T2 pass. Flag left off. |
| Freeze the speed ramp while the body lags (`RAMP_FREEZE`) | Same. Flag left off. The ramp is never pulled down by a measurement. |
| Soft spline touchdown (`SWING_PROFILE = "spline"`) | T5 touchdowns are already ~2 mm and ~2°. The falls are not impacts. Sine stays. |
| Yaw-moment penalty in the force QP | Stance-foot yaw rate on T5 has median 0.005 rad/s. The foot is not spinning. Not added. |
| Step-timing QP (`STEP_TIMING`) | Weights 0.02, 0.1 and 0.5 all walked S0 backwards (~−0.3 m/s) and turned survived pushes into falls. Flag left off. The foothold law alone is what cleared the falls. |
| `TURN_CAP_*` = 0.16 rad/step | Turn-in-place (T5) fell at 6.3 s. 0.12 did not, and it is the cap now. 0.25 was already ruled out. |

### Residual RL, scoped and not trained

A joint-torque residual is not justified. The failures that remain are
turn-in-place, tread, and the speed check on a turn whose crawl outlasts the
bench window. A 90° turn at 0.20 m/s now holds. If a later pass at
the CoP margin and the post-turn speed recovery does not clear them, the
cheap experiment is a per-step residual on the gait layer only: foothold xy
offset, a swing-clock scale, and optionally a yaw offset, bounded and added
to the capture correction, trained on the failing `walk_bench` scenarios.
`results/dr_rl*` is a different stack and is not a reason to skip that.

## Tools for the next attempt

`python -m loka.diagnose_walk` records a 125 Hz trace (CoM vs plan in the
heading frame, CoP against the sole edge, capture correction, swing height vs
command) and prints a per-step table aimed at these hypotheses. Default output
is `logs/walk/last.npz`; `--report` reprints from a saved file.

`python -m loka.dashboard --walk 0.25` drives this plant by hand: every
`gait.*` knob on a slider, the footstep plan and swing arc drawn in the viewer,
and the lateral CoM error plotted in the heading frame beside the forward speed
it is supposed to be trading against.

`python -m loka.verify_gait` is the planner-vs-closed-loop split. It first runs
the planner in isolation against a perfectly-tracking robot — the check to run
first after touching `gait.py` — then sweeps the closed loop.

## ALIP experiments (2026-09-28)

`alip_paper` and `alip_wbqp` were removed. The plant options are `legacy_dcm` and `alip_footstep`. The side-by-side bench lives in `docs/walk_bench.md`.

Isolation (`python -m loka.verify_gait --stack`) on the ALIP model with a perfect tracker: speed error is 0 after 3 steps at 0.05, 0.20, 0.40 and 0.60 m/s; a −30% momentum kick does not flip the sign of \(L\); the lateral cycle is centred to 0.0 mm. That is Gate 2, and it passed.

Closed loop, short smoke only (not the full bench):

| Trial | Result | Hypothesis |
|---|---|---|
| `alip_paper` quiet stand with `wbc.weight_force = 0` | OSQP: KKT not quasidefinite, fall | The force block is singular. Walking uses `1e-3`, not 0. |
| `alip_paper` quiet stand with CoP regularization left on | Fall at ~4 s, no solver error | The moment penalty fights the standing force plan. CoP weight is 0 while standing and 2 while walking. |
| `alip_paper` quiet stand after that split | 8 s, no fall | Standing path is the legacy WBC again. |
| `alip_footstep` walk at 0.20 m/s | Fall at ~2.1 s | First swing. Suspects, not yet separated: sagittal \(L\) sign, stance-width sign, or `capture_gain` 0.7 on a plant the law was not identified on. Not retuned to pass a test. |

Second pass on `alip_footstep` (0.20 m/s, 8 s). A per-touchdown trace checks the pre-impact prediction at the
freeze point against the real pre-impact state, and the law's step against
the landing. After the first two fixes both agree to about a centimetre, so
the remaining error is in the step-to-step map itself.

| Hypothesis | Result |
|---|---|
| LQR `Q = I` weights `L` ~(mH)² over `x` | True: every `capture_gain < 1` was deadbeat (eig 0.012). `Q = diag(1, 1/(mH)²)` now spans eig 0.12–0.32. Kept. |
| Deadbeat law asked for 0 → 0.2 m/s in one step | True: first foot 10 cm behind, second far ahead. Orbit speed now moves ≤ 0.05 m/s per step. Kept. |
| Orbit CoM reference with the DCM leash | The MPC brakes the pendulum; real lateral velocity is ~1/3 of the model's; law over-steps. Replaced. |
| Legacy DCM CoM reference for the whole walk | Pushes the CoM against the ALIP footholds (−0.25 m/s by step 2). Kept only for the opening step, where it supplies the weight shift. |
| Look-ahead pendulum velocity reference | The MPC pushes the CoM along the fall; reality diverges faster than the model. Replaced by the measured ALIP velocity. |
| Per-knot ALIP trajectory handed to the MPC | Falls at 2.7 s at all speeds and p95 rises to 7 ms. Reverted. |
| X&A double support (`L` constant) | Wrong on this plant: lateral velocity reverses within a 0.22 s double support. A CoP moving old → new contact over 80% of double support matches the traced steps. Kept (`DS_TRANSFER_FRACTION`). |
| Lateral cycle after that | Converges: 0.32 → 0.25 m steps, 6 s up. |
| Sagittal after that | Walks backward at ~0.25 m/s: a persistent ~+2 cm pivot offset the law did not know about. |
| Pivot-bias correction (online `L̇ = mg(x−d)` estimate, per-step EMA) | Reverses the sign (now accelerates forward), then overshoots and falls at 4–6 s. Grid over `capture_gain` 0.4/0.7/1.0 × bias rate 0.1/0.3: best is 8 s up but 1.7 m backward. Open. |
| Paper timing (0.8 s / duty 0.625) | Worse with every reference tried. |

At the end of this pass, `alip_footstep`, `alip_paper`, and `alip_wbqp` all still fell inside 8 s.
The ALIP model is right in single support; the convex MPC's 0.3 s horizon
changes double support and the sagittal pivot, so the step map the law
assumes is not the one the plant executes. Next step: identify the
step-to-step map online (least squares on logged `(X_k, u_k, X_{k+1})`,
model as prior) rather than fitting constants by hand.

Fixed in the same pass: the ALIP-MPC QP registered a dense Hessian
pattern, so every OSQP update after the first failed and returned the
previous answer. That planner left with `alip_wbqp`.

### Third pass: `alip_footstep` walks

| Hypothesis | Result |
|---|---|
| Identify the step-to-step map online (RLS on `(X_k, u_k, X_{k+1})`, model prior) | Stable, but still backward: the measured-velocity reference makes the MPC a velocity holder, so stepping cannot change speed. The 0.25 m/s outlier gate also rejected the sagittal samples. Kept for the `free` mode (`StepMapEstimator`). |
| Commanded speed as the MPC's sagittal velocity reference | The MPC and the law both regulate speed and fight; falls at 3 s. Partial pull (0.1–0.3) and weaker sagittal gain: falls at 3–7 s. |
| Nominal chain footholds + ALIP feedback, MPC on the DCM CoM reference | Error vs the ALIP model orbit: biased, falls at 2.5–3.5 s. Error vs the gait's own CoM reference: 0.2 and 0.3 m/s hold 8 s. |
| LQR/deadbeat gain on that error | 0.1 m/s crawl (two-thirds double support) walks backward into a fall. Capture-point shape `capture_gain*(dx + dv/omega)`: holds. |
| Correct in double support too | Worse at 0.1 m/s. Swing only, like the DCM nudge. |
| `L` about the contact vs `m H v_com` for the error | `L` reads the swing leg's and arms' momentum as a speed error; 0.1 m/s raced forward at 0.28 m/s. CoM velocity: 0.1/0.2/0.3 m/s all hold 8 s. The ALIP model still carries the error to impact. |
| 0.25 rad/step turn cap | Falls on T2, T5, N1, P2. Chain cap restored. |
| ALIP error while turning or treading | Still falls on T2, T5, N1 and B1's tread (the leashed CoM reference does not rotate with the chain). The DCM nudge while turning or treading: T2, T5, N1 pass. |
| Correction clamp 0.1 / 0.2, large-error handoff to the DCM nudge | No change on the 10 N·s pushes. Off by default. |
| `capture_gain` 0.4 / 0.7 / 1.0 | 1.0 passes P1_left_ds_10 but loses G3 and S1 (13/17). 0.7 is the default (14/17). |

Bench subset (17 scenarios), `legacy_dcm` vs `alip_footstep` at the
defaults: 14 vs 14. ALIP passes S1 (30 s lateral drift ≤ 0.15 m), which
legacy fails at 0.26 m; legacy survives B1 (tread → walk),
P1_back_late_10 and P1_left_ds_10, where ALIP falls. Both pass T1, T2, T5,
G1, G3, S0, B2, N1, P2 and the 8 N·s mid-swing pushes in all four
directions. Current numbers are in `docs/walk_bench.md`.

Open: large (10 N·s) backward pushes and the tread-to-walk start. After a
big push the ALIP error keeps the pre-push reference speed; the DCM nudge
re-anchors on the corrected foothold instead.

### Turning in place without walking off

A 90° zero-speed turn used to move the pelvis 0.71 m. The capture correction
put the swing foot 8–11 cm behind the plan, and the chain rebuilt its centre
on that foot, so the DCM reference followed.

The zero-speed chain now matches convex-mpc-biped's pure yaw: the footprint
centre is latched, and each foot is only the lateral offset rotated to that
step's yaw. Capture still moves the foot in the air. It does not move the
centre. A capped return `2/s × (centre − CoM)`, at most 0.10 m/s, is added
to the MPC velocity so the body walks back onto that centre. Measured on
`legacy_dcm`: a 90° turn ends about 0.11 m away (peak about 0.12 m) and a
straight 8 s walk at 0.20 m/s is unchanged (1.29 m forward, −0.16 m lateral).
A walking turn (speed 0.10 m/s and above) stays on the forward chain.

The earlier chain-pull of 5–10 cm/step did not hold (0.41–0.76 m): the
correction re-added what the pull removed. That ALIP-only rewrite is gone;
both stacks use the latched centre.

`legacy_dcm` remains the default. It is not retired.

`tests/test_gait.py` holds the planner invariants plus
`test_closed_loop_walk_stays_up` (8 s, 0.10/0.20/0.30) and
`test_closed_loop_walk_is_a_limit_cycle` (30 s, 0.10/0.20/0.35/0.50, marker
`slow`). The 0.35 and 0.50 rows are the same run: `gait.speed` clamps at
0.30, and that walk averages 0.20 m/s over 30 s, so both miss `0.6 ×` the
requested speed. That miss is unchanged with `TURN_SETTLE` off. Pelvis-z
alone is not a pass.
