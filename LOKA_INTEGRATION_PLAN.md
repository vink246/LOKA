# LOKA × G1 Integration Plan (working doc)

> **Local tracking only** — gitignored. Edit this as we revise priorities and mark progress.
> Last revised: 2026-07-29

## North star

LOKA is a **slow adaptive / immune layer** over a **fast certified MPC+WBC (+ gait) plant**.

| Layer | Rate | Job |
|---|---|---|
| Plant (MPC / WBC / gait / reflexes) | 50–500 Hz | Stay up *now* |
| LOKA orchestrator | ~0.5–5+ s | Diagnose, retune belief/mission/gait *policy*, promote reflexes |
| Operator language | sparse | New objectives + `Error_Tracking` |

**Hard rule:** anything that must act in &lt;300 ms cannot wait for the LLM.

**Paper target:** humanoid locomotion (stand → walk) with adaptive recovery under paper faults + operator language — *not* VLA / code-as-policy.

---

## Decisions (revise here)

| # | Topic | Current call | Open? |
|---|---|---|---|
| D1 | First milestone | Standing + compressor + operator lean/crouch/yaw **before** walking | |
| D2 | Large push | Classical auto-step / capture reflex at control rate; LOKA not in the catch path | |
| D3 | Sudden friction | Fast slip **reflex** + LOKA confirmation / retune | |
| D4 | Reflex authorship | **Hybrid:** ship a small catalog of classical reflexes that guarantee demos; LOKA may **tune and promote** them into session/cross-session memory (meta path). Pure LLM-invented reflexes are stretch / Phase C | yes — revisit after Phase A |
| D5 | LLM × gait | High-level gait *policy* knobs only (speed, step length/period, duty, swing height, stance width, mode). **Not** per-tick foothold XY or contact flags | |
| D6 | Paper scope (v1) | Stand + walk + faults + operator; corridor/crawl = stretch | yes |
| D7 | State estimate | Sim ground truth for now; estimator = hardware bridge later | |

---

## Priority order (accelerated)

1. **Phase A** — Standing LOKA loop (compressor, faults, operator, lean setpoints, eval harness skeleton)
2. **Phase B** — Walking plant + gait policy surface for LLM
3. **Phase C** — Reflex catalog + memory (meta-tune); classical defaults that always work
4. **Phase D** — Eval tasks / demos one-by-one against paper metrics
5. **Stretch** — Auto-step recovery, richer belief mutations, sim-to-real memory transfer

Work A → B in parallel only where interfaces are stable (e.g. scratchpad schema designed once for stand+walk).

---

## Phase A — Standing + LOKA loop

**Goal:** Language and fault recovery on the *existing* stand controller, with richer task setpoints than height-only.

### A.1 Task / command surface (expose to LLM)

Extend beyond crouch:

| Setpoint | Meaning | Notes |
|---|---|---|
| `height` | CoM height above feet | exists |
| `yaw` | facing | exists |
| `com_offset_xy` / lean | forward-back, left-right CoM bias | partially exists as `com_offset_xy`; expose as lean_x / lean_y with safe clamps (support-margin aware) |
| `Recovery_Mode` | HOLD / ABSORB / CROUCH / LIMP | optional early |

- [x] Lean forward/back and lateral as first-class `Task_Targets` (clamped by support polygon / capture margin)
- [x] Document lean limits in `tuning` / system prompt (physics-aware affordance)
- [x] Operator phrases: “lean left”, “shift weight forward”, “crouch”, “stand tall”, “face me”

### A.2 Telemetry compressor (G1)

Reuse Stack A compressor ideas; specialize signals:

- CoM error, torso rpy / lean, capture + support margins
- Contact mask / heel peel, MPC vs WBC force mismatch, QP fallbacks
- Motor load, saturation + near-zero velocity → seized / flailing tags
- Mission excess vs live `Error_Tracking`

- [x] G1-specific compression sections (mirrors paper §0–3)
- [x] Semantic tags: `[CAPTURE_MARGIN_LOW]`, `[SLIP]`, `[SATURATED_SEIZED]`, `[LEAN_LEFT]`, …
- [x] Trigger: Error_Tracking, effort/force mismatch, heartbeat, operator request

### A.3 Orchestrator apply path

- [x] Dual-rate worker (background LLM, main loop never blocks)
- [x] Scratchpad → `update_weights` / command / Error_Tracking / Model_Mutations (belief)
- [x] Dotted paths only (`mpc.*`, `wbc.*`, `stand.*`, `task.*`) — no ambiguous bare names
- [x] System prompt + `tuning.catalogue()` for G1 stand
- [x] Multi-turn failure episodes (existing `session.py` pattern)
- [x] Plateau / intervention-cap gate: stop thrashing when residuals do not improve; accept residual → lean Task_Targets + mass belief + MissionNominal rebase (`StandLokaConfig.max_interventions_per_episode=5`)

### A.4 Fault detection (standing)

Minimum detectable faults for demos:

- [x] Friction step-down (with note: reflex required for harsh ice — Phase C)
- [x] Asymmetric mass / CoM bias
- [x] Actuator disable (virtual amputation path)
- [x] Operator mission change mid-stand

### A.5 Exit criteria (Phase A)

- [x] `python -m …` entry: stand + LOKA + operator lean/crouch without falling
- [x] At least one automated fault trial with multi-turn recovery
- [x] Ablation baseline: fixed controller (no LOKA)

---

## Phase B — Walking + LLM gait policy

**Goal:** Humanoid locomotion as the paper’s main plant; LLM sets gait at a **high level**.

### B.1 Walking plant (engineering)

Minimal viable walker on top of current MPC+WBC:

1. Contact / gait clock (stance–swing schedule per foot)
2. Capture-point or Raibert-style footstep placement
3. Swing foot Cartesian task in WBC (reuse airborne leg hooks)
4. MPC: planned `contact_mask` + contact positions from schedule (not only measured soles)

- [x] Gait generator (periodic schedule)
- [x] Footstep policy (capture-point + Raibert velocity term)
- [x] Swing trajectory + WBC Cartesian foot-center task (Bézier; no separate foot IK)
- [x] Stand ↔ walk mode transition (`gait.mode` Task_Targets)
- [ ] Push / velocity tracking smoke tests (extend)

### B.2 What the LLM may tune (gait policy)

**In scope (typed, clamped):**

| Path (draft) | Role |
|---|---|
| `gait.mode` | `stand` / `walk` / `tread` / `limp` |
| `gait.speed` | commanded forward speed |
| `gait.heading` | world-frame travel direction |
| `gait.step_period` | cadence |
| `gait.duty_factor` | fraction in stance |
| `gait.step_length_max` | cap |
| `gait.swing_height` | clearance |
| `gait.stance_width` | lateral foot separation |
| `gait.capture_gain` | how aggressively foothold tracks ξ |
| existing MPC/WBC weights | priorities under that gait (`wbc.*_swing_foot` allowlisted) |

**Out of scope for LLM:**

- Per-horizon contact binary sequences
- Continuous foothold XY each MPC solve
- Raw swing knot waypoints every tick

**Rationale:** Language is good at “walk slower / higher steps / wider stance / limp.” Geometry of the next foothold stays classical so the QP stays well-posed.

- [x] Freeze gait tunable registry (`loka/control/gait.py` + Task_Targets)
- [x] Prompt section: AVAILABLE GAIT PARAMETERS
- [x] Operator: “walk at 0.3 m/s”, “shorten steps”, “raise feet”

### B.3 Exit criteria (Phase B)

- [x] Stable periodic walk in sim (no LOKA) — short smoke test; tune further as needed
- [x] LOKA can switch stand↔walk via `gait.*` Task_Targets / `--mode walk`
- [ ] One walking + operator demo recorded

---

## Phase C — Reflexes: classical core + LLM meta-memory

### C.0 How SOTA handles robustness (context)

| Approach | Idea | Fit for us |
|---|---|---|
| Domain randomization / privileged RL | Train under noise so policy is wide | Offline; not online discrete faults |
| Robust / tube / chance-constrained MPC | Plan with uncertainty sets | Heavy; complementary |
| Reflex layers (ANYmal, humanoid stacks) | Fast heuristic on IMU/force/slip | **What we need under 300 ms** |
| Adaptive / online system ID | Update model params | LOKA belief mutations |
| Contact / fall state machines | Detect fall risk → brace / step | Auto-step module |
| Learning residual / student of MPC | Speed + robustness | Out of scope for this paper |

**Call:** guarantee demos with a **fixed reflex catalog**; use LOKA to **tune and remember** which settings worked (sim-to-real story), not to invent new control laws from scratch on a falling robot.

### C.1 Reflex catalog (classical, always available)

Each reflex: trigger → action → safety bounds → telemetry tag.

| ID | Trigger | Action (fast) |
|---|---|---|
| `slip_mu` | loaded contact sliding | lower μ belief / shear authority |
| `capture_absorb` | margin critically low | soften CoM PD, raise damping, optional crouch |
| `capture_step` | margin &lt; 0 (if stepping exists) | emergency step |
| `sat_disable` | τ sat + \|q̇\|≈0 | locally zero that actuator command |
| `force_track_collapse` | ‖f_des−f_wbc‖ large | fade base task weight / reduce fz aggression |

- [ ] Implement catalog with deterministic defaults that pass harsh friction / sat demos
- [ ] Expose **tunable gains/thresholds** of each reflex to LOKA (not the structure)
- [ ] Logging: which reflex fired, when, outcome

### C.2 Meta-memory (your idea — phased)

**Session memory:** after LOKA retunes a reflex / belief and the same fault class recurs, prefer the last successful bundle.

**Cross-session memory:** persist `{fault_class → parameter bundle, success count}` for sim-to-real transfer.

- [ ] Fault taxonomy aligned with tags + paper scenarios
- [ ] Memory store (YAML/JSON); load on start
- [ ] Orchestrator prompt: ATTEMPTED FIXES + MEMORY HITS
- [ ] Promotion rule: only store if post-fix TSR / no-fall for N seconds
- [ ] Ablation: memory on vs off

**Risk to manage:** catastrophic forgetting / overfitting to sim. Keep clamps; never let memory bypass affordance limits.

### C.3 Exit criteria (Phase C)

- [ ] Sudden μ drop: survive with reflex-only; LOKA improves steady walking/standing after
- [ ] Memory replay improves second occurrence \(T_s\) or turns
- [ ] Document: LOKA does not replace reflexes for millisecond events

---

## Phase D — Evaluation harness + demos

### D.1 Infrastructure

- [ ] Scenario runner (seed, duration, fault injection time, operator script)
- [ ] Metrics: TSR, \(T_s\), \(E_{dev}\) or force-tracking integral, outer-loop turns, SAP, YAML validity
- [ ] Extra: reflex latency vs LOKA latency; min capture margin; fall rate by ablation
- [ ] Baselines: fixed plant; reflex-only; LOKA-only; full
- [ ] Artifact dump: telemetry, scratchpads, video path

### D.2 Task backlog (add one-by-one)

Paper draft:

| ID | Scenario | Depends on | Status |
|---|---|---|---|
| T1 | Mass shift (asymmetric) | A | pending |
| T2 | Friction shift | A + C reflex | pending |
| T3 | Hardware failure / amputation | A (+ C sat_disable) | pending |
| T4 | Operator semantic (crouch / lean / hold) | A | pending |
| T5 | Walk + speed/gait language | B | pending |
| T6 | Walk onto low friction | B + C | pending |

From integration discussion:

| ID | Demo | Status |
|---|---|---|
| D1 | Mid-stand friction + reflex + LOKA | pending |
| D2 | Seized knee limp | pending |
| D3 | Operator lean L/R + F/B + crouch | pending |
| D4 | Push past capture → auto-step (stretch) | pending |
| D5 | Ablation matrix figure | pending |

Corridor / crawl / rubble: **stretch**, not v1 blocker.

### D.3 Exit criteria (Phase D)

- [ ] One command runs the ablation matrix for T1–T4
- [ ] Numbers fill paper Table 1 placeholders
- [ ] Recorded demos for D1–D3

---

## Scratchpad schema (draft — evolve with code)

```yaml
Semantic_State:
  Hypothesis: "..."
  Analysis: "..."
Recovery_Mode: stand_hold   # stand_hold | stand_absorb | crouch | walk | limp
Controller_Targets:
  wbc.kp_base_position: 40.0
  mpc.friction_mu: 0.35
Task_Targets:
  height: 0.62
  yaw: 0.0
  lean_x: 0.02          # forward (+)
  lean_y: -0.01         # left (+)
  gait.mode: walk
  gait.speed: 0.25
  gait.step_period: 0.4
Error_Tracking:
  trigger_threshold: ...
  terms: [...]
Model_Mutations:
  - object_type: actuator
    name: ...
    attribute: gear
    value: 0.0
Reflex_Targets:            # Phase C
  slip_mu.mu_floor: 0.2
  capture_absorb.margin_enter: 0.02
Memory_Update:             # optional
  store_as: friction_low
```

---

## Feasibility cheat sheet (do not oversell)

| Claim | Verdict |
|---|---|
| Operator lean / crouch / walk-speed via LOKA | Yes |
| Mild friction / mass / limp via LOKA | Yes |
| Sudden ice via LLM alone | No — need reflex |
| Mega-push via LOKA alone | No — need step/absorb at control rate |
| LLM sets high-level gait | Yes |
| LLM invents footholds each step | No (out of scope) |
| Meta-memory tunes reflexes | Yes as Phase C; classical defaults first |

---

## Working log

| Date | Change |
|---|---|
| 2026-07-29 | Initial plan from accelerated-timeline discussion (stand→walk→reflex memory→eval) |

---

## Next concrete actions (update as we go)

1. [x] Freeze Phase A lean setpoint API + clamps
2. [x] Wire compressor + orchestrator apply onto `StandController`
3. [x] Skeleton eval harness + T4 operator lean demo
4. [x] Design walk gait registry (B.2) before coding walker so prompts stay stable
5. [x] Implement Phase B gait clock + CP/Raibert + Bézier Cartesian swing
6. [ ] Implement `slip_mu` + `sat_disable` reflexes early (unblock T2/T3)
7. [ ] Record walking + operator demo; tune velocity tracking

**When starting implementation:** switch chat to the active phase checkbox set and tick items as they land.
