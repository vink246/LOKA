# LOKA

**LLM-Orchestrated Kinematic Adaptation** — an immune system for dynamic locomotion, with a Unitree G1 balance controller as the plant.

Foundation models reason about the world in language; bipeds are dynamically unstable and fall easily. Bridging that gap is hard: VLAs struggle to compose novel skills online, Domain Randomization cannot isolate discrete unmodeled faults at runtime, and code-as-policy LLM loops are too slow (and brittle) for balancing underactuated robots. LOKA resolves this latency–stability paradox with a hybrid neuro-symbolic architecture: keep a mathematically transparent controller in the fast loop, and let a language model act as a slow, online system-identification engine that rewrites the controller's objective landscape and physical self-awareness by mutating typed parameters mid-flight.

When motors seize, friction collapses, mass shifts, or an operator says *"crouch and walk at 1 m/s"*, LOKA compresses proprioceptive anomalies into semantic tags, diagnoses in the background, and applies surgical edits—cost weights, planner numerics, task goals, mission criteria, even virtual amputations (zero gear on a dead actuator)—so the robot can rediscover a gait before collapse.

## What's here

| | Entry point | What it is |
|---|---|---|
| **Plant** | `python -m loka.main` | The G1 locomotion controller: centroidal MPC feeding a whole-body QP, with a gait layer on top. Never waits on anything. |
| **Manual tuning** | `python -m loka.dashboard` | The same plant with every knob on a slider and the footstep plan drawn in the viewer. No LLM in the loop. |
| **Orchestration** | `python -m loka.run_loka` | The research loop: telemetry compression → LLM diagnosis → typed parameter mutation, driving that plant. |

All three drive the *same* controller object through the *same* two typed
surfaces — `update_weights()` for costs and gains, `set_task_targets()` for
setpoints and gait policy. A knob you can reach by hand in the dashboard is a
knob the model can reach later, by the same name, with the same clamp.

Standing is solid and measured below. **Walking is not finished**: the robot
steps at the commanded speed and then topples sideways after eight to sixteen
steps. `docs/walking.md` has the diagnosis and what to try next; the failure is
pinned by an `xfail` in `tests/test_gait.py` rather than hidden.

## The G1 locomotion controller

Two layers, both quadratic programs, running in one process with MuJoCo:

```text
(qpos, qvel) ──► centroidal MPC  (50 Hz) ──► desired contact forces
             └─► whole-body QP  (500 Hz) ──► 29 joint torques ──► plant
```

The **MPC** treats the robot as a single rigid body and optimises contact forces
over a 0.3 s horizon, so it can anticipate where the centre of mass is heading
rather than only reacting to where it is. The **whole-body QP** runs ten times
faster and answers a different question: what torques realise those forces while
keeping the feet planted, the torso upright, and the posture near nominal. It
solves over `[q̈, f]` subject to the floating base's unactuated equations of
motion, friction cones, and actuator limits.

Nothing in `loka/control/` imports MuJoCo for anything but kinematics queries, and
the controller's interface is `compute_torque(qpos, qvel) -> torque`. Swapping the
simulator for a hardware bridge does not touch the control code.

### Measured behaviour

From `python -m loka.evaluate` (10 s hold, then disturbance sweeps):

| | |
|---|---|
| CoM error, undisturbed | 0.10 mm mean, 0.16 mm max |
| Torso tilt, undisturbed | 0.007° mean |
| Peak torque, undisturbed | 6.5 Nm (leg limits run to 139 Nm) |
| Push recovery, fore/aft | 9.4 / 10.0 N·s |
| Push recovery, lateral | 18.8 N·s both sides |
| Crouch tracking, 0.70 → 0.55 m | within 3.2 mm |
| Solve time | 0.68 ms mean, 1.77 ms p95, against a 2 ms budget |

The honest ceiling for a controller that cannot step is the **capture point**: once
`c + ċ/ω` leaves the support polygon, no contact force can arrest the fall. With
this stance that limit is 0.318 m/s of CoM velocity, and the controller absorbs
0.344 m/s — slightly past the point-mass bound, because the torso and arms
contribute angular momentum. Recovering meaningfully more would require a stepping
controller, not better tuning.

### Layout

```text
loka/control/
├── robot.py       # G1 model wrapper: indices, contact sites, dynamics, stance margins
├── qp.py          # OSQP front-end with pinned sparsity patterns for warm starts
├── mpc.py         # convex single-rigid-body MPC over contact forces
├── wbc.py         # whole-body QP: accelerations + forces -> torques
├── gait.py        # gait clock, footstep plan, DCM reference, swing arcs
├── locomotion.py  # ties them together; the LOKA-facing command surface
└── tuning.py      # the typed, clamped parameter registry (GUI + LLM)
loka/agent/       # the slow layer: compress -> diagnose -> typed apply
loka/sim.py       # MuJoCo loop, scripted pushes, run statistics
loka/dashboard.py # manual tuning GUI
loka/viz.py       # viewer overlays: footstep plan, swing arc, pushes, mass
loka/evaluate.py  # robustness suite (hold / push / crouch)
loka/config/g1.yaml  # static tuning, mirroring the dataclasses field for field
models/g1/        # vendored 29-DoF MJCF + stance scene
docs/             # walking.md (the gait layer), orchestrator.md (the slow layer)
```

## Setup

```bash
conda env create -f environment.yml
conda activate loka
```

`osqp>=1.1` is required — the controller reads `result.info.status_val` to reject
infeasible solves, and older versions report status differently.

Only the orchestrator needs an API key; the plant and the dashboard run without one:

```bash
cp .env.example .env   # then set OPENAI_API_KEY
```

## Run

```bash
python -m loka.main                  # interactive viewer
python -m loka.main --push 4         # shove the pelvis every 3 s
python -m loka.main --height 0.60    # crouch
python -m loka.main --headless -T 10 # scripted run, prints a summary

python -m loka.dashboard             # manual tuning GUI + viewer
python -m loka.dashboard --walk 0.25 # ... starting in a walk
python -m loka.dashboard --no-robot  # GUI only (faster on software GL)

python -m loka.evaluate              # plant robustness suite
python -m loka.verify_gait           # gait diagnostics across a parameter matrix

python -m loka.run_loka                          # the orchestration loop
python -m loka.run_loka --operator "crouch a bit" # ... driven by language
python -m loka.run_loka --no-llm --fault mass     # ... ablated, plant only
```

Static tuning lives in `loka/config/g1.yaml`, which mirrors the dataclasses in
`loka/control/` field for field; the dataclass defaults are what runs when no
`--config` is passed.

## The tuning surface

There are two, and everything that can change at runtime goes through one of
them. `loka/control/tuning.py` declares the costs and gains — 30 of them, each
with a range and a one-line rationale — and `loka/control/gait.py` declares the
`gait.*` policy the same way in `GAIT_KNOBS`. The dashboard renders those
registries; the orchestrator mutates the same entries by the same names:

```python
controller.update_weights(**{"wbc.kp_base_position": 30.0, "mpc.friction_mu": 0.35})
controller.set_task_targets({"gait.mode": "walk", "gait.speed": 0.25, "height": 0.65})
```

Footstep coordinates and per-tick contact flags are *not* on either surface.
Walking geometry stays classical and outside the model's vocabulary; the model
picks a speed, a cadence and a stance width, and `gait.py` decides where the
feet go.

Two properties make this the right seam for an LLM:

- **Paths are unambiguous.** `weight_force` and `friction_mu` exist on *both*
  layers with values three orders of magnitude apart, so a bare field name is
  not a safe address. Only `mpc.weight_force` and `wbc.weight_force` resolve.
- **Values are clamped** into a band that keeps the QPs well posed, so a
  hallucinated exponent degrades the stance instead of destroying the solver.

Applying an update costs ~0.1 ms and rebuilds nothing: both layers re-read
their weights every solve, and the handful of values baked into a constraint
matrix are refreshed in place. You can drag a slider while the robot is
standing on the result. `tuning.catalogue()` renders the whole surface as
prompt-ready text.

## Tests

```bash
pytest                      # ~30 s
pytest -m 'not slow'        # invariants only, no closed-loop runs
```

The suite splits into physics identities checked at a single state (contact forces
sum to body weight, torques match the equation of motion, `J̇q̇` agrees with finite
differences) and short closed-loop runs for drops, tilts, crouches and pushes. The
capture-point test asserts recovery *and* asserts that a kick 50% past the limit
still falls, so the suite cannot be satisfied by loosening what counts as a fall.

The one known failure is marked, not hidden: `test_closed_loop_walk_stays_up` is
an `xfail`, so it reports if the lateral divergence is ever fixed by accident,
and the gait planner's own invariants are tested in isolation either way.

## Notes on the design

A few choices are load-bearing and easy to undo by accident:

- **Contact detection is absolute, not relative.** A sole point counts as planted
  when it is near the estimated ground height. Comparing against the *lowest* sole
  point instead makes an airborne robot read as fully planted, and the QP then
  computes torques bracing against support that does not exist.
- **Keeping the feet planted is a cost, not a constraint.** As a heel lands, a hard
  "hold this point still" row fights the velocity the foot already has and the
  program can go infeasible in the one millisecond where an answer matters most.
- **Base position gains are deliberately low.** A planted biped accelerates its CoM
  only by shifting the centre of pressure inside its soles, capping authority near
  1.2 m/s². Gains asking for more saturate, turn the PD law bang-bang, and overshoot
  the robot out of its own support polygon.
- **The stance keyframe in `models/g1/scene.xml` is tuned, not arbitrary.** Knees
  are bent clear of the straight-leg singularity, soles are exactly flat, and the
  legs are pitched forward ~0.0176 rad so the CoM lands mid-sole — which is what
  makes the forward and backward push margins equal.
- **Dashboard widget payloads go through DearPyGui's `user_data`.** DearPyGui
  decides how many arguments to pass by reading `co_argcount`, which counts
  parameters that merely *have* defaults. The usual Python idiom for capturing
  a loop variable — `lambda s, v, path=entry.path: ...` — therefore reads as a
  three-argument callback and gets `path` overwritten with `user_data`. Every
  callback takes the full `(sender, app_data, user_data)` triple so this cannot
  recur quietly; `tests/test_dashboard.py` fires each widget through the real
  dispatch path rather than calling the handlers directly.
- **The walk plan leads the robot; measurement only nudges it.** Footholds chain
  off the current support foot, one commanded stride apart, and the CoM
  reference is the DCM trajectory that chain implies. Measured state enters in
  exactly two clamped places. The version before this derived the CoM reference
  from the measured feet while placing the feet relative to the measured CoM —
  a loop with no term commanding forward progress, which is why it could only
  shuffle.
- **The viewer redraws at 60 Hz, not once per control tick.** `viewer.sync()`
  costs ~3 ms; calling it every 2 ms tick spends more time drawing than
  simulating and drops the window to 0.41x realtime. Batching a frame's worth of
  control steps between redraws restores 0.87x. Do not move the physics to a
  background thread to go faster: `sync()` reading `MjData` mid-write segfaults,
  and guarding it with `viewer.lock()` contends so badly it lands at 0.59x.
  `--headless` remains the fast path at 2.4x.
