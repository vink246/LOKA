# The LOKA orchestrator (implemented)

**Last revised:** 2026-09-02  
**Scope covered:** the `loka/agent/` package — telemetry compression, LLM
diagnosis, typed apply — over the G1 plant  
**Not covered yet:** classical reflex catalog, cross-session memory,
paper-scale ablation matrix (Phases C–D in the integration plan)

The plant this drives is documented separately: `docs/walking.md` for the gait
layer and its open defect, and the README for the control law. The orchestrator
was built when the plant could only stand, and §19 lists what has to change now
that it can walk.

---

## 1. What LOKA is (this repo)

**LOKA** (*LLM-Orchestrated Kinematic Adaptation*) is a **slow adaptive /
immune layer** over a **fast certified balance plant**.

| Layer | Rate | Job |
|---|---|---|
| Plant (centroidal MPC + whole-body QP) | ~50 Hz MPC / ~500 Hz WBC | Stay upright *now* |
| LOKA orchestrator | ~0.5–5+ s (async) | Diagnose from compressed telemetry; mutate typed YAML (costs, tasks, mission criteria, model belief) |
| Operator language | sparse | New objectives + updated `Error_Tracking` |

**Hard rule:** anything that must act in &lt;300 ms must not wait on the LLM.
The control thread never blocks on an API call; the LLM runs in a background
worker and results are drained on the next control ticks.

**Paper framing (v1 intent):** humanoid stand → walk with adaptive recovery under
discrete faults + operator language — *not* VLA / code-as-policy. The reflex
catalog is planned but **not implemented** in the current code path.

There is one entry point: `python -m loka.run_loka`. (The original MuJoCo-MPC
Walker stack this grew out of has been deleted; if you find a reference to
`mujoco_mpc`, `models/walker/` or a root `main.py`, it is stale.)

---

## 2. Architecture overview

```text
                    ┌─────────────────────────────────────────┐
  stdin / keys /    │           LokaRuntime.step()       │
  --fault / --op    │                                         │
        │           │  1. inject timed / interactive faults   │
        │           │  2. drain LLM queue (apply scratchpad)  │
        │           │  3. re-apply belief mutations           │
        │           │  4. LocomotionController.compute_torque → τ  │
        │           │  5. MuJoCo step (sim plant)             │
        │           │  6. build telemetry frame               │
        │           │  7. mission Error_Tracking + anomaly    │
        │           │  8. plateau gate OR dispatch LLM        │
        │           └───────────────┬─────────────────────────┘
        │                           │
        │              ┌────────────┴────────────┐
        │              │                         │
        ▼              ▼                         ▼
   sim.model      controller.robot.model    OpenAI (async)
   (hidden        (LOKA *belief* prior)     YAML scratchpad
    plant faults)  gear / μ / mass          → apply_stand_scratchpad
```

**Plant vs belief:**

- **Plant** (`Simulation.model` / `sim.model`): physics the robot *actually*
  experiences. Interactive / scripted faults mutate this model (extra mass,
  ice friction, dead actuator gear, external push).
- **Belief** (`LocomotionController.robot.model`): private MuJoCo model the
  controller uses for dynamics / planning. `Model_Mutations` from LOKA write
  here only. Re-applied every control tick via `apply_loka_mutations`.

The LLM is never given a privileged fault label; it diagnoses from telemetry
and tags only.

---

## 3. Package map

### 3.1 The orchestrator package (`loka/agent/`)

| Module | Responsibility |
|---|---|
| `runtime.py` | Dual-rate loop owner: `LokaRuntime`, `LokaConfig`, dispatch/drain, mission baseline, plateau accept |
| `anomaly.py` | Early plant-health scoring + semantic clues; gated so healthy crouch residuals do not open episodes |
| `compress.py` | Telemetry compression (sections 0–3) for the LLM user turn |
| `apply.py` | Parse/apply YAML scratchpad → allowlisted weights, tasks, Error_Tracking, belief mutations |
| `policy.py` | Hard LLM allowlist (9 adaptive knobs); rejects locked QP costs |
| `plateau.py` | Round-to-round severity, intervention cap, accept-residual scratchpad + mass inference |
| `faults.py` | `FaultSpec` + plant-side mass / friction / actuator_dead apply & clear |
| `interactive.py` | Stdin fault command parsing (rendering lives in `loka/viz.py`) |
| `llm.py` | System prompt assembly + background OpenAI worker (`system_prompt.txt`) |
| `context.py` | Capabilities block, robot context, current configuration formatting |
| `error_defaults.py` | Default standing `Error_Tracking` (pelvis height, drift, tip/roll rates) |
| `session_log.py` | Per-run `logs/loka/session.log` + `.jsonl` (rewritten each run) |
| `session.py` | `FailureEpisode` / `OperatorSession` multi-turn conversation state |
| `error_spec.py` | `ErrorSpec` / `ErrorTerm`, tracking error, YAML parse |
| `model_state.py` | MJCF name resolution; apply belief mutations; zero dead-actuator commands |
| `robot_context.py` | Static kinematic reference text for the prompt |

### 3.2 Plant modules the orchestrator drives

| Module | Role |
|---|---|
| `loka/control/locomotion.py` | `LocomotionController`, `LocomotionCommand` (height / lean / yaw), telemetry |
| `loka/control/gait.py` | Gait clock, footstep plan, DCM reference, swing arcs; `gait.*` policy knobs |
| `loka/control/mpc.py` | Centroidal convex MPC |
| `loka/control/wbc.py` | Whole-body QP |
| `loka/control/robot.py` | `G1Model` kinematics / contacts / support margin |
| `loka/control/tuning.py` | Full tunable catalogue (GUI + allowlist subset) |
| `loka/sim.py` | MuJoCo simulation wrapper, pushes, fall detection |
| `loka/viz.py` | Viewer overlays: footstep plan, swing arc, pushes, added mass |

### 3.3 Entry points

| Command | Purpose |
|---|---|
| `python -m loka.run_loka` | Interactive / headless plant + LOKA |
| `python -m loka.evaluate_loka` | Headless scenario skeleton (mostly no-LLM ablations) |
| `python -m loka.dashboard` | The same plant driven by hand, no LLM in the loop |
| `python -m loka.main` | Plant alone, viewer or headless |
| `python -m loka.evaluate` | Plant-only disturbance / push eval |
| `python -m pytest tests/test_agent.py` | Orchestrator unit / integration tests |

### 3.4 Prompts

`system_prompt.txt` at the repo root, loaded by `loka/agent/llm.py`. Live model
context (objective, capabilities, robot reference) is appended at assembly
time, so the file itself stays plant-agnostic.

---

## 4. The plant

### 4.1 Control law

```text
(qpos, qvel) ──► centroidal MPC (~50 Hz) ──► desired contact forces f*
             └─► whole-body QP (~500 Hz) ──► 29 joint torques τ ──► MuJoCo
```

- **MPC:** single rigid-body CoM dynamics; optimises contact forces over a short
  horizon subject to friction cones.
- **WBC:** realises those forces with base / posture / contact tasks under
  floating-base dynamics, friction, and torque limits (OSQP).
- **API:** `compute_torque(qpos, qvel) → τ`. No EKF; sim uses ground-truth state
  (decision D7 in the integration plan).

Typical undisturbed numbers (from plant eval / README): sub-millimetre CoM
error, sub-degree tilt, sub-millisecond solves vs a ~2 ms budget. The hard
ceiling without stepping is the capture point; large pushes need a future
reflex / step module (Phase C / stretch), not more LLM latency.

### 4.2 Task / command surface (`Task_Targets`)

Exposed to LOKA and ablations via `LocomotionController.set_task_targets`:

| Name | Meaning | Limits (approx.) |
|---|---|---|
| `height` | CoM height above foot plane [m] | band ~`[0.50, 0.78]` |
| `yaw` | Facing [rad] | `|yaw| ≤ 0.8` |
| `lean_x` | CoM offset forward (+) [m] | clamped by support margin (~55% of margin) and `|lean| ≤ 0.06` |
| `lean_y` | CoM offset left (+) [m] | same |

Aliases `com_offset_x` / `com_offset_y` are accepted. Lean is implemented as
`LocomotionCommand.com_offset_xy`.

**Height convention (easy to confuse):**

- `Task_Targets.height` ≈ CoM height above soles (~0.70 nominal).
- `Error_Tracking` `pelvis_height` uses `qpos[2]` (pelvis world Z, ~0.78 when tall).
  When crouching, the LLM must lower the pelvis Error_Tracking target; it must
  **not** set pelvis target equal to CoM height.

### 4.3 Controller tunables vs LLM allowlist

The full catalogue lives in `loka/control/tuning.py` (also for dashboards).
The orchestrator may only set the **adaptive allowlist** in
`loka/agent/policy.py`:

| Path | Intent |
|---|---|
| `mpc.friction_mu` | Friction belief in MPC |
| `wbc.friction_mu` | Friction belief in WBC |
| `wbc.kp_base_position` | CoM position stiffness |
| `wbc.kd_base_position` | CoM position damping |
| `wbc.kp_base_orientation` | Torso orientation stiffness |
| `wbc.kd_base_orientation` | Torso orientation damping |
| `wbc.weight_posture_legs` | Leg posture task weight |
| `wbc.kp_posture_legs` | Leg posture stiffness |
| `wbc.kd_posture_legs` | Leg posture damping |

**Locked on purpose** (rejected if the LLM emits them): `mpc.weight_*`,
`wbc.weight_contact` / base task priorities, `contact_kd`, `stand.max_*_acc`,
upper/swing posture knobs, etc. Early experiments showed wholesale QP retunes
could destabilise the certified plant.

Bare names like `weight_force` are ambiguous across layers and are rejected.

---

## 5. Dual-rate runtime loop

### 5.1 `LokaConfig`

| Field | Default | Meaning |
|---|---|---|
| `enable_llm` | `True` | If false: plant + faults only; no API dispatch |
| `cooldown_s` | `4.0` | Min seconds between LLM dispatches |
| `anomaly_collection_s` | from compressor (~1 s) | Collection window for **mission** breaches |
| `heartbeat_s` | `30.0` | Reserved |
| `objective` | default stand objective string | Primary mission text in the prompt |
| `print_compressor` | `True` | Print full user turn to stdout |
| `enable_session_log` | `False` | Write `session.log` / `.jsonl` |
| `max_interventions_per_episode` | `5` | Hard cap before accept-residual |
| `plateau_min_interventions` | `2` | Min fixes before plateau can fire |
| `plateau_stall_rounds` | `2` | Consecutive non-improving rounds |
| `plateau_rel_improve` | `0.08` | Relative severity improvement threshold |
| `plateau_abs_score_eps` | `0.03` | Absolute anomaly-score drop counts as improve |

CLI: `--max-interventions` maps to `max_interventions_per_episode`.

### 5.2 Per-tick sequence (`LokaRuntime.step`)

1. Fire any timed `FaultSpec`s whose `time` has elapsed; handle interactive injects.
2. `_drain_llm()` — if a worker result is ready, `apply_stand_scratchpad`, optionally
   invalidate mission baseline on `Task_Targets` changes.
3. Re-apply belief mutations on `controller.robot.model`.
4. Control step: `compute_torque` → zero commands for belief-dead actuators →
   MuJoCo substeps; record push viz wrench.
5. Build a telemetry **frame** (qpos/qvel, CoM error, rpy, contacts, forces,
   margins, util, cmd lean/height, QP failure counts, …).
6. Until `MissionNominal` baseline is full (~2 s of healthy frames), only collect
   baseline and return.
7. Evaluate **mission** error via `Error_Tracking` and **early anomaly** via
   `assess_stand_anomaly`.
8. Operator queue (if idle + LLM on): rebuild prompt, invalidate baseline, dispatch
   `OperatorSession`.
9. Else if mission breach or plant anomaly (after collection + cooldown):
   plateau/cap gate, else dispatch `FailureEpisode`.
10. Else healthy: refresh / rebase baseline; close failure episode if recovered.

Constants of note:

- `EARLY_ANOMALY_COLLECTION_S = 0.4` — short window before early-anomaly dispatch.
- `TASK_SETTLE_SUPPRESS_S = 5.0` — ignore early clues / mission lag after
  Task_Targets / operator / accept-residual while the pose settles.
- `NOMINAL_BASELINE_S = 2.0` — length of MissionNominal buffer.

### 5.3 MissionNominal re-baseline

`MissionNominal` in the compressor is **empirical healthy telemetry under the
current mission**, not a forever-frozen initial stand and not LLM-invented
numbers.

- On operator request or applied `Task_Targets` (and on accept-residual):
  `_invalidate_mission_baseline` clears the buffer, marks `_baseline_stale`,
  suppresses early triggers for the settle window.
- After settle, when `balance_healthy`, frames are re-collected until the buffer
  is full → “mission nominal baseline rebased”.
- While stale, **mission** Error_Tracking breaches do not dispatch (criteria may
  still describe the old objective). Early plant anomalies are also suppressed
  during settle.

### 5.4 Sessions

- **`FailureEpisode`** (`loka/agent/session.py`): one multi-turn conversation from first
  fault/anomaly until recovery, plateau accept, operator objective change, or
  close. Tracks `interventions`, `metric_history`, `stall_count`,
  `accepted_residual`.
- **`OperatorSession`**: multi-turn operator requests (mission changes). Requires
  LLM; ablations use structured `--task` instead (no keyword NLP).

`prior_turn_count` = completed user/assistant pairs. `round_number` =
`len(interventions) + 1`.

---

## 6. Triggers: mission vs early anomaly

### 6.1 Mission — `Error_Tracking`

Default (`error_defaults.py`), trigger threshold `0.20`:

| Term | Signal | Mode | Notes |
|---|---|---|---|
| `pelvis_height` | `qpos[2]` | `below_target` | tol 0.08, weight 2 |
| `lateral_drift` | `qpos[1]` | `abs_deviation` | tol 0.12 |
| `forward_drift` | `qpos[0]` | `abs_deviation` | tol 0.15 |
| `tip_rate` | `qvel[4]` (wy) | `abs_above` | tol 0.8 |
| `roll_rate` | `qvel[3]` (wx) | `abs_above` | tol 0.8 |

The LLM may rewrite `Error_Tracking` entirely (especially on operator mission
changes). Weighted excess over tolerances produces a scalar tracking error;
breach when above `trigger_threshold`.

### 6.2 Early plant-health — `anomaly.py`

Separate from mission criteria. Scores **balance distress** so LOKA can act
before a fall / before Error_Tracking trips.

**Distress clues (always eligible):**  
`[TILT_ELEVATED]` / `[TILT_HIGH]`, `[SUPPORT_MARGIN_LOW]`, `[FORCE_MISMATCH]`,
`[CONTACT_SLIP]`, `[ACTUATOR_STRAIN]`, `[QP_FALLBACK]`.

**Tracking clues (gated):**  
`[COM_BIAS_LEFT/RIGHT]` can trigger even when otherwise calm (asymmetric mass).  
Fore-aft / overall CoM elevation only counts when balance is already unhealthy;
healthy deep crouch may tag `[COM_TRACKING_RESIDUAL]` informatively without
triggering.

**Trigger policy:** distress score ≥ `0.12`, or strong lateral bias, or
unhealthy + total score ≥ `0.12`. Vertical CoM residual alone does **not** open
a failure episode (that was a crouch false-alarm class).

Key thresholds (see source for exact values): CoM warn/strong 8/25 mm,
tilt warn/strong ~0.6°/2°, force mismatch warn 40 N, margin warn 35 mm, etc.

### 6.3 Residual floor (after plateau accept)

After accept-residual, `_residual_floor` stores the accepted `EpisodeMetrics`.
Subsequent early anomalies that do **not** exceed that floor by a margin
(score / force / CoM) are suppressed so LOKA does not thrash the same
steady-state offset. Clearing interactive faults clears the floor.

---

## 7. Telemetry compressor

`synthesize_stand_telemetry` builds the LLM user-turn body:

0. **TRACKED STATE** — each Error_Tracking term: MissionNominal vs current window
   vs snapshot; in-band / out-of-band labels.
1. **BALANCE / CONTACTS** — CoM error, torso rpy, support margins, contacts,
   force tracking ‖f*-f‖, mpc_cost, QP fails, command setpoints, semantic tags,
   early anomaly score.
2. **MOTOR LOAD** — mean/peak |τ|, utilisation.
3. **DIRECTIVE** — short instruction (plant anomaly vs operator, etc.).

Compressor semantic tags (examples): `[CAPTURE_MARGIN_LOW]`, `[SLIP]`,
`[LEAN_FORWARD/BACK/LEFT/RIGHT]`, `[SINGLE_SUPPORT_RISK]`,
`[SATURATED_SEIZED:jointk]`, `[SATURATED_FLAILING:jointk]`. Early-anomaly clues
from `anomaly.py` are also surfaced in the directive / balance tags.

---

## 8. Scratchpad apply path

### 8.1 Schema (stand)

```yaml
Semantic_State:
  Hypothesis: "..."
  Analysis: "..."
Controller_Targets:          # allowlisted dotted paths only
  wbc.kp_base_position: 40.0
  mpc.friction_mu: 0.35
Task_Targets:
  height: 0.65
  lean_x: 0.0
  lean_y: 0.0
  yaw: 0.0
Error_Tracking:
  trigger_threshold: 0.2
  terms: [...]
Model_Mutations:
  - object_type: actuator   # or geom | body
    name: right_knee
    attribute: gear         # or friction | mass
    value: 0.0              # mass is absolute kg on the body
```

`Planner_Targets` are ignored with a log note: they addressed a sampling
planner's metaparameters, and this plant has none.

### 8.2 Apply behaviour (`apply_stand_scratchpad`)

1. Require `Semantic_State`.
2. Record intervention on the active `FailureEpisode` / `OperatorSession` if provided.
3. Filter `Controller_Targets` through the allowlist; apply via
   `controller.update_weights`.
4. `set_task_targets` with clamps.
5. Parse `Error_Tracking` into `loka_state["error_spec"]`.
6. Queue `Model_Mutations` onto `loka_state["mutations"]` (resolve MJCF names on
   belief model, fallback plant for name resolution only) and call
   `apply_loka_mutations`.

Supported belief attributes today: **`gear`**, **`friction`**, **`mass`**.
No `body_ipos` / inertia belief yet.

On accept-residual mass updates, the runtime also refreshes
`robot.total_mass` and `mpc.mass` so gravity/force scaling tracks belief mass.

### 8.3 LLM worker

- Model: `LOKA_STAND_MODEL` env (default `gpt-5.4-mini`), temperature `0.1`.
- Requires `OPENAI_API_KEY`.
- Strips markdown fences; `yaml.safe_load` → scratchpad dict on the result queue.
- Failures put `None`; runtime logs and continues plant control.

Prompt assembly: `system_prompt.txt` + objective + capabilities catalogue +
current Error_Tracking + robot context (G1 stand notes, lean limits, qpos layout).

---

## 9. Plateau / intervention-cap gate

### 9.1 Problem this solves

Under lasting model mismatch (e.g. 5 kg on a shoulder), the plant often reaches a
**stable residual**: upright, planted, low util, but permanent CoM / force-tracking
offset that keeps early-anomaly above threshold. Naïve multi-turn LLM behaviour
then thrashing `kp`/`kd`/height (crouch ↔ uncrouch) without improving severity.

### 9.2 Detection

Before each failure-episode dispatch:

1. Snapshot mean metrics over the anomaly buffer → `EpisodeMetrics`
   (anomaly score, CoM xy/lat, force mismatch, tilt, mpc_cost).
2. Compare to previous snapshot via `severity()` /
   `is_improved(..., rel_improve, abs_score_eps)`.
3. Update `stall_count` (consecutive non-improvements).
4. Stop if:
   - `len(interventions) >= max_interventions_per_episode` → `max_interventions`, or
   - `interventions >= plateau_min_interventions` and
     `stall_count >= plateau_stall_rounds` → `plateau`.

### 9.3 Accept residual

`_accept_residual` builds and applies a scratchpad **without** another LLM call:

- **Task_Targets:** lean toward persistent CoM bias (clamped); keep current height/yaw.
- **Model_Mutations:** infer absolute body mass from lateral CoM
  (`Δm ≈ |Δy| · M / lever`, shoulder body) or force mismatch → torso.
- **Error_Tracking:** retarget pelvis floor / widen drift tols for the new pose.
- Invalidate MissionNominal, close the failure episode, set `_residual_floor`,
  refresh MPC mass if mutated.

Logged as `[LOKA] accepting residual at t=... (reason=plateau|max_interventions, ...)`.

---

## 10. Plant faults

### 10.1 Kinds (`FaultSpec`)

| Kind | Effect on **plant** | Typical params |
|---|---|---|
| `mass` | Add `delta_kg` to a body mass | `body` (`torso_link`, `left_shoulder_roll_link`, …), `delta_kg` |
| `friction` | Set floor geom sliding friction | `mu` (ice ~0.2) |
| `actuator_dead` | Zero actuator gear + ctrlrange | `actuator` name (default `right_knee`) |
| `push` | Impulse on pelvis via `Simulation.pushes` | `impulse` [N·s], `direction` |

Belief model is **not** changed by plant fault injection (verified in tests).

### 10.2 Interactive use

Stdin commands and viewer keys (see `interactive.FAULT_HELP`):

- `mass [kg] [torso|left|right]`, `mass left|right [kg]`
- `friction [mu]` / `ice`
- `push [N.s] [dir]`
- `dead [actuator]`
- `clear` — restore mass/friction/dead (not pushes); clear residual floor
- any other text → operator request (LLM on only)

Keys: `1` torso mass, `2` ice, `3` push, `4` dead knee, `5` clear,
`6`/`7` shoulder mass, `H` help.

Overlays: cyan push arrow; amber mass marker; strong cyan floor tint for ice.

### 10.3 Scripted / CLI

```bash
python -m loka.run_loka --fault mass --fault-at 4 --fault-mass 5
python -m loka.run_loka --fault friction --fault-mu 0.25
python -m loka.run_loka --fault actuator_dead --fault-actuator right_knee
python -m loka.run_loka --fault push
```

---

## 11. Operator language vs ablations

| Mode | How | LLM? |
|---|---|---|
| Free-text operator | `--operator "..."` or stdin non-fault text | **Required** |
| Structured ablation | `--no-llm --task height=0.60,lean_x=0.03` | No |
| Keyword NLP for crouch/lean | **Not implemented** (by design) | — |

Operator turns instruct the model to update `Task_Targets` **and**
`Error_Tracking` to match the new mission. Runtime closes any open failure
episode and invalidates MissionNominal when an operator request is accepted.

---

## 12. Logging

When session logging is enabled (default in `run_loka` unless `--no-log`):

- `logs/loka/session.log` — human-readable compressor turns, LLM YAML,
  apply summaries, notes (rewritten each run).
- `logs/loka/session.jsonl` — structured events for tooling.

Directory is gitignored; paths print at startup.

---

## 13. Evaluation harness (skeleton)

`python -m loka.evaluate_loka [--scenario NAME] [--json-out path]`

| Scenario | LLM | What it does |
|---|---|---|
| `hold` | off | Quiet stand baseline |
| `lean_operator` | off | Direct Task_Targets lean/crouch ablation |
| `mass` | off | Torso mass fault, no recovery policy |
| `friction` | off | Floor μ step-down, no recovery |
| `mass_recovery` | off | Two **scripted** scratchpads (soften gains, then crouch) |

This is a Phase A skeleton, not the full paper ablation matrix (TSR, \(T_s\),
SAP, reflex latency, etc. — Phase D).

---

## 14. Tests

`tests/test_agent.py` (run under the `mjpc` conda env):

| Test | Covers |
|---|---|
| `test_lean_and_height_are_clamped` | Task clamps |
| `test_apply_scratchpad_task_and_weights` | Happy-path apply |
| `test_apply_rejects_ambiguous_bare_weight_name` | Bare name rejection |
| `test_apply_locks_robot_tuned_mpc_weights` | Allowlist locks |
| `test_compressor_emits_balance_section` | Compressor sections |
| `test_mass_fault_injection_changes_plant_not_belief` | Plant ≠ belief |
| `test_runtime_survives_scripted_lean_without_llm` | Runtime + lean |
| `test_evaluate_lean_scenario` / `…_scripted_multi_turn_recovery` | Eval scenarios |
| `test_interactive_fault_commands_parse` | Stdin parser |
| `test_inject_and_clear_faults_update_viz` | Viz state |
| `test_mission_baseline_rebases_after_task_change` | MissionNominal rebase |
| `test_early_anomaly_fires_on_shoulder_mass_not_quiet_stand` | Anomaly gating |
| `test_plateau_improvement_and_stop_reasons` | Plateau helpers |
| `test_accept_residual_updates_mass_belief_and_lean` | Accept residual side effects |
| `test_gate_failure_dispatch_detects_plateau` | Stall → plateau |
| `test_session_log_rewrites_each_session` | Log rewrite semantics |

Plant-only tests also exist under `tests/test_control.py` etc.

---

## 15. How to run (cheat sheet)

```bash
conda activate mjpc
cd /path/to/LOKA

# Interactive stand + LOKA (viewer + stdin faults / operator)
python -m loka.run_loka

# Operator language (needs OPENAI_API_KEY)
python -m loka.run_loka --operator "lean left and crouch slightly"

# Ablation without LLM
python -m loka.run_loka --no-llm --task height=0.60,lean_x=0.03

# Headless timed fault
python -m loka.run_loka --fault mass --headless -T 12 --max-interventions 5

# Eval skeleton
python -m loka.evaluate_loka

# Tests
python -m pytest tests/test_agent.py -v
```

Environment:

- `OPENAI_API_KEY` — required for LLM modes
- `LOKA_STAND_MODEL` — optional model override (default `gpt-5.4-mini`)

---

## 16. Design decisions baked into the code

1. **Dual-rate only** — control never waits on LLM.
2. **Typed YAML only** — no code generation, no footholds, no contact schedules.
3. **Hard allowlist** — certified QP costs stay human/robot-tuned.
4. **Plant ≠ belief** — faults are hidden; belief updates are explicit mutations.
5. **No operator keyword NLP** — language requires LLM; ablations use `--task`.
6. **Mission vs plant-health** — Error_Tracking is editable mission criteria;
   anomaly.py is fixed early distress detection.
7. **MissionNominal is empirical** — re-baselined after objective / accept changes.
8. **Plateau / cap** — stop thrashing stable residuals; absorb into lean + mass belief.
9. **Sim ground truth** — no state estimator in the loop yet (hardware later).

---

## 17. Explicitly not implemented yet

| Area | Status |
|---|---|
| Classical reflex catalog (`slip_mu`, `capture_step`, …) | Phase C |
| Session / cross-session memory store | Phase C |
| Auto-step past capture | Stretch |
| Body CoM offset / inertia belief attrs | Not in `apply_loka_mutations` |
| Full paper metrics harness (TSR, \(T_s\), SAP, …) | Phase D |
| Hardware / estimator bridge | Deferred |

---

## 18. Relationship to other docs

| File | Tracked? | Purpose |
|---|---|---|
| **`docs/orchestrator.md`** (this file) | **Yes** | What the slow layer does and how |
| `docs/walking.md` | Yes | The gait layer, and the open lateral-stability defect |
| `LOKA_INTEGRATION_PLAN.md` | No (gitignored) | Working roadmap, decisions, phase checkboxes |
| `README.md` | Yes | High-level project intro + plant metrics |
| `system_prompt.txt` | Yes | Live orchestrator instructions |

When you change behaviour in `loka/agent/`, update **this** file in the same
PR/commit when practical so GitHub stays the source of truth for implementers.

---

## 19. What still has to change now that the plant walks

This package was deliberately left alone during the Phase B walking work, so
the plant and the slow layer could be debugged separately. It is *not*
stand-only: `error_defaults.py` already swaps in `default_walk_error_spec` when
the mode changes, `anomaly.py` carries a second set of `WALK_*` thresholds, and
`context.py` warns the model that slip and force mismatch are normal during
swing. What follows is what is genuinely still owed.

**The compressor is blind to the gait.** This is the load-bearing gap.
`compress.py` formats CoM error, tilt, support margin, contacts and motor load
— and mentions the gait nowhere. `LocomotionTelemetry` already carries
`gait_phase`, `swing_clearance` and `swing_error`, and `GaitScheduler` exposes
`cmd_speed` and `step_index`, but `LokaRuntime._frame` copies only the
`walking` flag, so none of it reaches the model. A diagnosis of the lateral
divergence in `docs/walking.md` needs at least: phase within the cycle,
commanded versus achieved forward speed, lateral CoM error resolved in the
*heading* frame rather than world y, and swing tracking error per step. The
dashboard plots exactly these (`loka/dashboard.py`, `PLOTS`), which is a
reasonable specification to copy.

**Naming.** Every public symbol here still says `stand`:
`apply_stand_scratchpad`, `synthesize_stand_telemetry`, `default_stand_error_spec`,
`build_stand_capabilities`, `build_stand_robot_context`, `format_stand_configuration`,
`load_stand_system_prompt`, `stand_llm_worker`, `StandTelemetrySections`,
`assess_stand_anomaly`, `LLM_STAND_CONTROLLER_ALLOWLIST`, `stand_llm_catalogue`,
and the `LOKA_STAND_MODEL` environment variable. The plant classes were renamed
to `Locomotion*` and the package to `loka/agent/`; the functions inside were
not, because renaming them without rethinking what they report would only move
the problem.

**The gait capability list is hand-maintained.** `context.py` writes the
`gait.*` knobs out as prose, and has already drifted — `gait.foothold_retarget_s`
is missing, and no knob carries its range. `loka/control/gait.py` now exports
`GAIT_KNOBS`, each with a summary and a band, in the same shape
`tuning.catalogue()` uses; rendering from that removes a class of drift.

**Plateau accept assumes the residual is a mass offset.** `_infer_mass_belief`
resolves a stable lateral CoM bias into a believed shoulder or torso mass.
Standing, that is a good inference. Walking, a persistent lateral bias is at
least as likely to be foothold placement, and writing it into the belief model
would bury the very defect that is open.

**Walk faults are untested.** `tests/test_agent.py` exercises faults from a
stand. Nothing injects a fault mid-stride, which is where the interesting
recoveries are — and where the plant currently falls over anyway.
