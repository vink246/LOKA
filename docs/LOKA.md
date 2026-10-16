# LOKA: LLM-Orchestrated Kinematic Adaptation for Humanoid Locomotion

**Working methodology document.** This is the long-form source for the Methodology section of the paper in `paper/`. It is deliberately more detailed than the paper will be. The paper version should keep the structure and equations and drop most of the numeric detail, which is kept here so that every claim in the paper can be traced to the implementation.

**Scope of this revision:** motivation, methodology (architecture, low-level controller, anomaly detector, telemetry compressor, orchestrator, scratchpad, belief search) and experimental setup. Results and discussion are not yet covered.

**Source of truth:** the implementation as of October 2026, in its default configuration (Unitree G1, 29 actuated degrees of freedom, MuJoCo plant, convex-MPC + whole-body-QP stack with DCM footholds). Where earlier project notes or the current paper draft disagree with the implementation, this document follows the implementation; the disagreements are listed in Section 12.

---

## 1. Motivation

### 1.1 Robust humanoid locomotion and its current trade-off

Humanoid robots are built to operate in environments designed for people, and locomotion is the capability on which every other task depends. Walking in such environments means coping with conditions that the designer of the controller did not anticipate: a payload of unknown mass strapped to the torso, an actuator that has lost part of its strength, a floor that is more slippery than the one the gait was tuned on, an obstacle in the path, or a new goal given in natural language. Existing approaches handle these conditions in four broad ways, and each pays for its robustness in a different currency.

**Model-based control.** Classical humanoid controllers plan footsteps and centre-of-mass motion on reduced models such as the zero-moment point and the divergent component of motion [1], [2], [3], and track them with model-predictive and whole-body optimisation over the robot's rigid-body dynamics [4], [5], [6], [7]. These controllers need no training data, enforce contact and actuation constraints explicitly, and are interpretable: every decision can be traced to a cost, a constraint or a model parameter. Their weakness is the model itself. Their stability margins are derived for the model they were given, and a mismatch between that model and the robot, such as an unmodelled payload, a weakened motor or a wrong friction assumption, degrades performance or causes a fall. Inertial parameters can be identified offline from dedicated excitation experiments [8], [9], and state estimators and contact detectors can compensate for slippage and uneven contact [10], [11], but deciding *which* part of the model has become wrong while the robot is in use, and changing it, is in practice left to a human engineer. Model-based controllers are therefore precise but brittle outside the conditions for which they were built.

**Reinforcement learning with domain randomisation.** Learned locomotion policies obtain robustness by training in simulation over randomised dynamics, so that the policy is exposed during training to the variations it will meet at deployment [12], [13]. This approach has produced robust and agile legged and humanoid locomotion on real hardware [14], [15], [16], [17], and massively parallel simulation has reduced training to minutes or hours per policy [18]. The robustness, however, is bounded by the randomisation the designer chose: a perturbation outside the training distribution is not handled, and the gap between simulation and the real robot remains an open problem [19]. Each new robot, task or class of perturbation requires reward design, a new training run and validation, and the resulting policy is a black box that cannot be told, after the fact, what has changed about the world.

**Vision-language-action models.** Vision-language-action (VLA) models fine-tune large pretrained vision-language models to output robot actions directly, transferring web-scale knowledge to robot control [20], [21], [22]. Recent humanoid foundation models pair a slow vision-language module with a fast learned motor policy [23]. These models generalise across instructions and scenes, but they require large robot datasets, substantial post-training for every embodiment, and, for whole-body balance, a learned low-level controller that inherits the limitations of the previous paragraph.

**Language-model agents.** A fourth line of work keeps the foundation model frozen and uses it as a planner or programmer on top of a library of skills [24], [25], [26], [27], [28], or as an author of a policy interface, a reward or a cost for a lower-level controller [29], [30], [31]. These systems need no robot-specific training of the language model, but the skills they compose are fixed: when a skill fails because the robot or the environment has changed, the agent can choose another skill but cannot repair the one that failed. As discussed in Section 2.1, most of them also target platforms that remain safe while the model is thinking, which a walking humanoid is not.

In short, existing approaches to robust humanoid locomotion are either brittle outside their design conditions or obtain robustness through large amounts of training that does not transfer to a new robot, task or perturbation.

### 1.2 Foundation models as a source of general reasoning

Large language and vision-language models trained on internet-scale data now display broad, general-purpose reasoning that is available without task-specific training [32], [33]. Elicited through prompting alone, they can decompose problems step by step [34], interleave reasoning with actions [35], and keep an explicit working memory of intermediate results [36]. Robotics work has shown that, without fine-tuning, such models can describe and explain robot failures [37], [38], write reward functions and cost weights for controllers [31], [39], [30], and plan multi-step tasks from natural-language goals [24], [26].

These are precisely the abilities that the human engineer contributes when a model-based controller stops fitting its robot: describing what has changed in the robot or its environment ("the torso is heavier than the model believes", "the floor is slippery", "there is a chair in the way"), planning what to do about it, and configuring the controller's goals and parameters accordingly. The premise of this work is that foundation models are now capable enough to play that role. If they can, a model-based controller, which needs no training, can be made robust to conditions outside its design envelope by a foundation model, which also needs no training, without any reinforcement learning, post-training or fine-tuning.

Two obstacles stand in the way. First, timing: a foundation model answers in seconds, whereas a humanoid's balance must be corrected every few milliseconds, so the model can never be in the balance loop. Second, grounding: the model cannot observe forces, torques and contact states directly, and an unconstrained model that rewrites a balance controller can destabilise it.

### 1.3 LOKA

We introduce LOKA (LLM-Orchestrated Kinematic Adaptation), a hybrid neuro-symbolic, dual-rate (slow–fast) controller for humanoid locomotion that exploits the general knowledge of foundation models to make a humanoid robust to changes in itself and its environment, without any additional training of either the controller or the model.

The *fast* part of LOKA is symbolic: a model-based locomotion controller with an explicit physical model, explicit constraints and explicit task parameters, which keeps the robot balanced at the control rate and never waits for the slow part (Section 4). The *slow* part is neural: an off-the-shelf language or vision-language model that is called only when needed, reasons about the situation, and acts on the fast part solely by editing its task targets, its gait parameters and its internal physical model through a narrow, typed interface (Sections 7 and 8). The division mirrors the distinction between fast, automatic and slow, deliberative processing in human cognition [40], and follows the neuro-symbolic principle of combining learned, general-purpose reasoning with explicit, verifiable structure [41]. Between the two parts, LOKA adds the components that resolve the timing and grounding obstacles: an anomaly detector that recognises, in terms of forces, when the controller's model no longer fits the robot (Section 5); a compressor that turns the robot's high-rate state into a short report that a language model can reason about (Section 6); an orchestrator that is invoked asynchronously and schedules its own next invocation (Section 7); and a search routine that turns the model's qualitative diagnosis into an identified physical parameter (Section 9).

With this structure, LOKA allows a humanoid to walk to goals, navigate around obstacles, and recover from model errors such as added payloads, weakened actuators and low-friction floors. Our claim, evaluated against the same model-based controller without LOKA and against an open-source reinforcement-learning locomotion policy for the same robot [42], is that this robustness is comparable to that of existing methods while requiring no reinforcement learning, post-training or fine-tuning.

---

## 2. Overview

### 2.1 Problem setting

A humanoid robot is a floating-base, underactuated system whose support polygon is a pair of soles roughly 0.17 m long and 0.06 m wide. Balance has to be maintained continuously on a millisecond timescale: the centre of pressure can only move within the soles, and while walking the robot spends most of every step on one foot. Any component that is in the balance loop must therefore answer within one control period (2 ms in our system).

Large language models (LLMs) and vision-language models (VLMs) answer in seconds. In our own logs the wall-clock round trip of one orchestrator call ranges from roughly 2 s to 11 s. A language model can therefore never be in the balance loop of a humanoid. This is the main reason why most work that places a language model in a robot's control stack targets platforms that remain safe while the model is thinking: manipulators and mobile bases that can simply stop [24], [25], [26], [27], or legged systems driven by a learned policy whose interface the model writes [29]. Work that lets a language model modify the parameters of a model-based controller has so far addressed navigation cost weights for wheeled robots [30] and reward code for a sampling-based predictive controller [31]. Runtime anomaly detection followed by slow language-model reasoning, with a model-predictive fallback that keeps the system safe in the meantime, has been demonstrated by AESOP [37]; turning robot logs into language summaries for failure explanation has been demonstrated by REFLECT [38], offline and for manipulation. Humanoid agents that use a VLM as a high-level planner compose learned skills rather than adapting a model-based controller [28].

LOKA addresses a different question: **while a dynamically unstable humanoid keeps walking under a classical, certified model-based controller, can a language model detect that the controller's internal model or task no longer fits the world, decide what to change, and change it through a narrow, typed interface, without ever blocking the controller?**

### 2.2 Design principles

The architecture follows six principles. Each is enforced by the implementation rather than by prompting alone.

1. **The fast loop never waits.** Balance, footstep placement and contact scheduling run at 50 to 500 Hz and never block on the language model. The language model runs on a background thread, and its output is applied between two control ticks.
2. **A typed, bounded interface instead of code.** The language model writes a structured YAML document (the *scratchpad*), never executable code, footholds, contact schedules or optimisation costs. Every field it may write is enumerated, typed and clamped to a band in which the controller is known to remain stable.
3. **Plant and belief are separate.** The controller plans with its own private copy of the robot model (its *belief*). Faults are applied only to the physical plant and are never revealed to the language model. LOKA can only change the belief, and must infer the fault from its consequences.
4. **Diagnose from dynamic consistency, not tracking error.** The anomaly detector compares measured generalised forces with the forces the belief model predicts for the measured motion. A correct model explains any motion, including intentional crouching, leaning and walking, so the detector responds to model error rather than to commanded behaviour, and it responds before the error has become a kinematic deviation.
5. **The language model decides what to identify; a deterministic routine decides the number.** The language model chooses which physical parameter is wrong (for example, the mass of the torso). A closed-loop search driven by the same residual finds its value.
6. **Self-scheduled attention and persistent intent.** The language model decides when it next wants to be called (through *listeners*) and keeps its own editable plan, which survives after the conversation that produced it has scrolled out of its context window.

### 2.3 Architecture overview

LOKA has five components (Fig. 1).

1. **Low-level model-based locomotion controller.** A gait planner based on the divergent component of motion (DCM), a centroidal convex model-predictive controller (MPC) and a whole-body quadratic-programming controller (WBC). It turns joint positions and velocities into 29 joint torques. This component is adapted from prior work (Section 4); everything else is new.
2. **Anomaly detector.** Runs at the control rate. It computes three signals: the *effort bias* (a model-consistency residual expressed per actuator), the *motor deficit* (commanded minus delivered actuator torque) and the *stance slip* (tangential speed of planted sole points). A persistent excursion of any of them opens a failure episode and invokes the orchestrator.
3. **Telemetry compressor.** Turns the last two seconds of high-rate state into a compact, labelled text report, ordered by relevance, together with the controller's live tunables, the current belief and the mission context.
4. **High-level LLM/VLM orchestrator.** An API-hosted language model, optionally given an egocentric camera image, invoked asynchronously by the anomaly detector, by its own listeners, by operator directives, or by a periodic heartbeat. It replies with a scratchpad.
5. **Editable scratchpad for belief state and planning.** The typed, persistent state that the orchestrator edits: task targets, gait policy, belief-parameter searches, the internal model, wake-up listeners, and the self-assigned plan. A deterministic *belief-search executor* (Section 9) turns queued searches into identified model parameters.

```
   operator text ──► locked mission directives
                                   │
     ┌─────────────────────────────┼─────────────────── slow loop (event-driven, seconds) ─┐
     │                             ▼                                                        │
     │   telemetry ──► Telemetry Compressor ──► Orchestrator (LLM/VLM, background thread)  │
     │   buffer (2 s)        ▲                         │                                    │
     │                       │                         ▼                                    │
     │   Anomaly Detector ───┘ trigger          YAML Scratchpad ──► typed apply + clamps    │
     │   (bias, motor, slip)   listeners ◄──────────── │ ──► belief-search executor         │
     │          ▲              heartbeat               │                                    │
     └──────────┼──────────────────────────────────────┼────────────────────────────────────┘
                │                                      ▼  applied between two ticks
     ┌──────────┼─────────────────────────── fast loop (500 Hz) ────────────────────────────┐
     │  measured forces                 belief model ◄─ mass / CoM / friction / gear         │
     │          │                       task targets, gait policy                           │
     │   plant (MuJoCo) ◄── torques ── WBC (500 Hz) ◄── MPC (50 Hz) ◄── gait / DCM (500 Hz) │
     └──────────────────────────────────────────────────────────────────────────────────────┘
```

*Fig. 1. LOKA's dual-rate architecture. Solid paths in the lower box run every control tick; the upper box runs only when an event invokes the orchestrator.*

| Component | Rate | Latency budget | Blocks control? |
|---|---|---|---|
| Whole-body QP | 500 Hz | < 2 ms | is the control loop |
| Gait planner (DCM, footholds, swing) | 500 Hz | < 2 ms | is the control loop |
| Centroidal convex MPC | 50 Hz (every 10th tick) | < 2 ms | is the control loop |
| Anomaly detector | 500 Hz, 0.2 to 1.5 s windows | negligible | no |
| Belief-search executor | 500 Hz, probes every 0.4 to 1.7 s | negligible | no |
| Telemetry compressor | per orchestrator call | milliseconds | no |
| Orchestrator | event-driven; at most one call in flight; ≥ 2 s between applied edits; heartbeat every 15 s | 2 to 11 s observed | never |

### 2.4 Contributions

The low-level controller is an engineering adaptation of established methods (Section 4) and is not claimed as a contribution. The contributions are:

1. **An effort-residual anomaly detector for humanoids** that compares measured generalised forces against the inverse dynamics of the controller's own belief model, resolves the unactuated floating-base part of the residual through the contact Jacobian into actuator space, and combines it with a motor-deficit and a stance-slip signal. It is insensitive to intentional motion and sensitive to model error, and every quantity it uses is available on hardware (Section 5.7).
2. **A telemetry compressor** that renders this residual, the robot's full state, its progress toward a goal, the decomposed MPC cost and contact events into a short labelled report tailored to an LLM (Section 6).
3. **An asynchronous orchestrator for a dynamically unstable robot**, with four invocation sources (anomaly, operator directive, self-armed listener, heartbeat), bounded multi-turn failure episodes, and a reply that is applied between two control ticks (Section 7).
4. **A typed scratchpad that exposes both task-level parameters and the controller's internal model** at fine granularity: individual gait-policy knobs and task targets on one side, and the mass, centre of mass, friction and actuator strength of named bodies, surfaces and motors on the other, together with self-authored listeners and a persistent plan (Section 8).
5. **Language-directed, residual-driven belief identification**: the language model selects the parameter, and an unbounded bracketing search on the same residual identifies its value while the robot stands in a stabilising gait (Section 9).

To our knowledge, no prior work exposes both the low-level controller's task parameters and its internal physical model to a language model at this granularity, and no prior work evaluates LLM orchestration of a model-based controller on humanoid locomotion.

---

## 3. Notation

| Symbol | Meaning |
|---|---|
| $q \in \mathbb{R}^{36}$, $\dot q \in \mathbb{R}^{35}$ | Generalised position (floating base as position + quaternion, 29 joints) and velocity |
| $S$ | Selection matrix of the 29 actuated coordinates; subscripts $b$ and $a$ denote the 6 floating-base and 29 actuated rows |
| $M(q)$, $h(q,\dot q)$ | Joint-space inertia matrix and bias force (gravity, Coriolis, centrifugal) |
| $\hat M$, $\hat h$, $\hat m$ | The same quantities evaluated on the controller's belief model |
| $r_i$, $J_c$ | Position of sole point $i$ (four per foot, eight in total) and the stacked contact Jacobian ($24 \times 35$) |
| $f \in \mathbb{R}^{24}$ | Contact forces at the eight sole points |
| $\tau \in \mathbb{R}^{29}$, $u \in \mathbb{R}^{29}$ | Joint torque, and actuator command |
| $c$, $\dot c$ | Centre of mass (CoM) position and velocity |
| $\Theta = (\phi, \theta, \psi)$, $\omega$ | Torso roll, pitch, yaw, and angular velocity |
| $z_0$, $\omega_0 = \sqrt{g / z_0}$ | CoM height above the sole plane and linear-inverted-pendulum natural frequency |
| $\xi = c + \dot c / \omega_0$ | Horizontal divergent component of motion (DCM) |
| $p$ | Zero-moment-point (ZMP) reference |
| $T$, $T_{ss}$, $T_{ds}$ | Step duration, single-support and double-support durations |
| $\Delta t$ | MPC discretisation step (50 ms); control period 2 ms |

---

## 4. Low-Level Model-Based Locomotion Controller

### 4.1 Structure and rates

The low-level controller follows the MPC + whole-body control split introduced for the MIT Cheetah 3 and Mini Cheetah by Di Carlo et al. [4] and Kim et al. [43]: a centroidal convex MPC plans contact forces over a short horizon at a low rate, and a whole-body controller realises those forces with joint torques at a high rate. We adapt it to a humanoid by (i) replacing point feet with four contact points per sole and constraining the centre of pressure to the physical sole, following Sleiman et al. [5]; (ii) placing a DCM-based gait planner [2], [3] in front of the MPC that supplies the footstep plan, the contact schedule, the CoM reference and the swing trajectories, so that a short force horizon suffices, as argued by Galliker et al. [44]; and (iii) closing the stepping loop with a capture-point foothold correction [45]. The plant is the 29-degree-of-freedom Unitree G1 [46] simulated in MuJoCo [47] with a 1 ms physics step.

| Layer | Rate | Input | Output |
|---|---|---|---|
| Gait planner | 500 Hz | measured CoM, feet, angular momentum; gait policy | contact schedule; CoM, DCM and yaw references; swing-foot position, velocity and acceleration; horizon preview of contacts and footholds |
| Centroidal convex MPC | 50 Hz | centroidal state; references; preview | contact forces for the eight sole points |
| Whole-body QP | 500 Hz | full state; belief dynamics; MPC forces; task-space targets | joint accelerations, contact forces and 29 joint torques |

The controller is a pure function from measured state $(q, \dot q)$ to torques. It does not step the simulator and does not read the plant's physical parameters; it evaluates kinematics and dynamics only on its belief model (Section 4.2). The same object can therefore drive hardware through a state estimator without modification.

Both QPs are solved with OSQP [48]. The sparsity pattern of each problem is fixed at start-up, so each control tick only updates numerical values and never re-instantiates the solver. This is also what makes live parameter edits from the orchestrator safe (Section 7.5). Both use warm starting, a tolerance of $10^{-4}$ and at most 200 iterations. A failed or non-finite solve returns the previous solution; if the MPC's forces are still non-finite, it falls back to an even split of body weight over the eight sole points. Failures are counted and reported.

### 4.2 Plant model and belief model

The controller holds its own rigid-body model of the robot, separate from the simulated plant. All masses, inertias, centres of mass, friction coefficients and actuator gear ratios the controller uses are read from this *belief model*. Faults are injected only into the plant model (for example, an added payload or a weakened knee motor), and the belief model is unaffected until the orchestrator changes it. Every control tick, the belief is rebuilt from the nominal robot description plus the list of accepted belief edits, so belief edits are absolute values rather than increments and cannot drift.

### 4.3 Gait planner

The gait planner decides which feet are on the ground, where the next feet will land, and where the CoM should be. Its geometry is classical and is not exposed to the language model; the language model only sets the high-level policy parameters listed in Section 4.8.

**Gait cycle and modes.** A gait cycle of period $T_{cyc}$ contains two steps of duration $T = T_{cyc}/2$. Each leg swings for $(1-\delta)T_{cyc}$, where $\delta$ is the duty factor, leaving $(\delta - \tfrac12)T_{cyc}$ of double support on either side of each swing. Four modes are available: *stand* (both feet planted, stationary reference), *walk*, *tread* (stepping in place), and *limp* (a conservative preset with a long cycle, a high duty factor, low swing clearance and a capped speed).

**Footstep chain.** Footholds form a chain anchored at the current support foot and previewed five steps ahead. Each link advances the chain's centre line by $v T$ along the yaw of that step, where $v$ is the commanded speed, and places the foot half the stance width to its side. Because the chain is re-anchored to the actual support foot at every step but each link is exactly one commanded stride, the plan's velocity equals the commanded velocity by construction, while its footholds are always reachable from where the robot actually stands. Each step may yaw toward the commanded heading by at most $\min(\dot\psi_{max} T,\ 0.12\ \text{rad})$, where $\dot\psi_{max}$ is the turn-rate parameter. Below 0.02 m/s a heading change becomes a turn on the spot: the centre of the footprint is latched and each foot is placed at the lateral offset rotated to the new yaw, with a small, capped velocity reference that returns the CoM to the latched centre.

**DCM reference.** The CoM reference is derived from the footstep chain with Englsberger's backward DCM recursion [2], [3]. The chain is assumed to come to rest on its last previewed step; the DCM at the end of each step is then computed backward,
$$
\xi^{eos}_{k-1} = p_k + \left(\xi^{eos}_k - p_k\right) e^{-\omega_0 T},
$$
and within the current step $k$,
$$
\xi_{ref}(t) = p(t) + \left(\xi^{eos}_k - p(t)\right) e^{\omega_0 (t - T)}.
$$
The influence of the terminal assumption on the current step decays by $e^{-\omega_0 T} \approx 0.27$ per step, so five previewed steps are sufficient. Following the linear-inverted-pendulum argument of Kajita et al. [1], the ZMP reference $p$ is not placed at the foot centre but inset 12 mm toward the midline, which leaves the outer part of the sole free for disturbance rejection. During double support the ZMP slides from one inset point to the next. The first step of a walk begins with an anticipatory transfer of the ZMP onto the future swing foot, so that the first lift-off does not throw the CoM toward the swing side. The CoM reference is recovered from the DCM reference by the stable first-order filter
$$
\dot c_{ref} = \omega_0 \left(\xi_{ref} - c_{ref}\right),
$$
integrated on the plan's own state, so that measurement noise does not enter the reference. The tracking error handed to the controller is saturated at 6 cm by pulling the reference toward the robot, never past the plan.

**Capture-point foothold correction.** Measured state enters the plan only through a bounded correction of the next foothold, following capture-point stepping [45]. During swing, the measured DCM is propagated to touchdown under the current support ZMP $p_s$ over the remaining step time $T_{rem}$,
$$
\hat\xi_{td} = p_s + \left(\xi - p_s\right) e^{\omega_0 T_{rem}},
$$
and the foothold that would leave the planned DCM offset beyond the next foot is computed. The error is measured against that *offset*, not against the planned end-of-step DCM itself, because the latter moves with the correction (the chain is rebuilt from the corrected foothold) and a law written on it settles at only a fraction $k/(1+k)$ of the error. The applied correction is a fraction $k$ (the capture gain, 0.7 by default; 1.0 is dead-beat) of the full placement, is clamped to 0.20 m from the nominal foothold, is projected back into a stride box and a minimum lateral foot separation of 0.16 m, and, after mid-swing, is limited to what the swing foot can still reach at 0.45 m/s in the remaining swing time. The foothold is frozen once a configurable fraction of the swing has elapsed. During the first two steps only the forward component is applied.

**Swing trajectories.** The swing foot follows an analytic arc: the horizontal motion is a smoothstep $3s^2 - 2s^3$ in the swing phase $s \in [0,1]$, so that the foot leaves and lands with zero horizontal speed, and the vertical motion is a half sine of the commanded clearance. Analytic velocity and acceleration are passed to the WBC as feed-forward. The foot's yaw is interpolated from lift-off to the planned foothold yaw.

**Touchdown and timing.** The clock leaves single support when the swing-foot centre is within 8 mm of the ground and, if the foot actually left the ground, within 30 mm of its planned foothold and with at least one sole point in contact. After 75 % of the swing such a contact may plant the foot early; if the foot is still high when the step time elapses, single support is extended by up to 0.2 s.

**Speed-dependent cadence.** If the operator or the language model sets only a speed, the cycle period, duty factor and stance width are filled in from a schedule measured on this plant: period from 0.80 s to 0.64 s, duty factor from 0.84 to 0.74 and stance width from 0.22 m to 0.24 m as the speed goes from 0 to 0.5 m/s, clipped to the safe bands of Table 2. A high duty factor at low speed matters: on this plant a duty factor of 0.65 at 0.2 m/s falls within about 3 s. Any parameter set explicitly overrides the schedule. Changes to period, duty factor, width and swing height take effect only at step boundaries and are slewed (0.04 s, 0.02, 15 mm and 10 mm per step) so that an edit never jumps the swing phase of a foot that is in the air.

**Heading, speed ramp and turn settling.** The commanded heading is a goal that the footstep chain approaches at the turn rate. Forward speed is ramped at a configurable acceleration and is cut toward a 0.10 m/s crawl as the heading error grows from 0.35 rad to 0.85 rad. Following the gait-transition logic of the open-source whole-body humanoid MPC project [49], the gait stays on the crawl after a large turn until the heading error has been below 0.10 rad for two steps. The torso yaw reference is interpolated across each step, and both hip-yaw joints are held so that each foot stays square to it.

**Go-to-goal.** A world-frame goal point can be handed to the planner. The planner then re-aims the heading at the goal every tick (frozen within 0.30 m of the goal, where the bearing swings with every lateral sway), and commands the approach speed
$$
v = \min\Big(\max\big(k_g (d - d_{tol}),\ v_{min}\big),\ v_{cruise}\Big),
$$
with $k_g = 0.5\ \text{s}^{-1}$, $v_{min} = 0.10$ m/s, $d$ the distance from the pelvis to the goal, $d_{tol}$ the arrival tolerance and $v_{cruise}$ the commanded speed. A stop is requested at 80 % of the tolerance; the robot finishes the step in progress and stands. If the settled stand drifts more than 5 cm past the tolerance, walking resumes. Closing the position loop in the planner, rather than through repeated language-model calls, is what allows the orchestrator to issue a goal and sleep.

**Stopping.** A stand request finishes the step in the air, ramps the speed to zero, and enters stand once both feet are down, the speed along the heading is below 0.12 m/s and the pelvis is within 6 cm of the foot midpoint. The gates are loosened after four extra steps, and after eight step boundaries the robot stands anyway (waiting up to twelve for both soles to be down). This bound matters under model error, which can otherwise hold the body outside the gates indefinitely.

**Alternative stepping law.** A second stack keeps all of the above and replaces only the foothold correction with the angular-momentum linear-inverted-pendulum (ALIP) stepping law of Xiong and Ames [50], with angular momentum about the contact as the second state, as motivated by Gong and Grizzle [51]. The orchestrator interface is identical for both stacks.

### 4.4 Centroidal convex MPC

The MPC follows the convex single-rigid-body formulation of Di Carlo et al. [4], which has been carried to bipeds with finite feet by Li and Nguyen [52].

**State and dynamics.** The state is
$$
x = \left[\Theta^\top,\ c^\top,\ \omega^\top,\ \dot c^\top,\ -g\right]^\top \in \mathbb{R}^{13},
$$
where the constant last entry carries gravity so that the dynamics remain linear. Under the small-roll-and-pitch assumption, with $R_z(\psi)$ the yaw rotation and $\hat I$ the world-frame centroidal inertia of the belief model,
$$
\dot\Theta \approx R_z(\psi)^\top \omega,\qquad
\ddot c = \frac{1}{\hat m}\sum_{i=1}^{8} f_i + g,\qquad
\dot\omega \approx \hat I^{-1} \sum_{i=1}^{8} (r_i - c) \times f_i .
$$
These are discretised by forward Euler with $\Delta t = 50$ ms into $x_{k+1} = A\,x_k + B_k u_k$ over a horizon of six steps (0.30 s), where $u_k$ stacks the 24 force components. $A$ depends only on the current yaw. $B_k$ depends on the contact positions and the CoM at step $k$ and is rebuilt per step.

**Contact positions over the horizon.** The gait planner supplies, for every horizon step, which feet are planted and where each foot centre will be (its planned foothold if scheduled down, its point on the swing arc otherwise) together with the foot's planned yaw. The four sole points are placed around each planned centre using the sole offsets measured on the robot at that instant, rotated by the change in foot yaw. The CoM entering $B_k$ is not the reference: it is a forward rollout of the translational dynamics under the force plan from the previous solve. This linearises the bilinear moment term about the last primal solution, in the spirit of real-time iteration [53], while keeping a single convex QP per solve.

**Constraints.** For every sole point $i$ and horizon step $k$:

- a linearised friction pyramid, $|f_{x}| \le \mu f_{z}$ and $|f_{y}| \le \mu f_{z}$;
- normal-force bounds $s_{i,k} f_{min} \le f_{z} \le s_{i,k} f_{max}$, with $s_{i,k} \in \{0,1\}$ the planned contact state, $f_{min} = 5$ N and $f_{max} = 400$ N. The positive minimum keeps every planted point loaded, which is what makes a two-foot stand stiff; an airborne point is forced to zero.

For every foot $j$ and horizon step $k$, the centre of pressure is constrained to the physical sole, following Sleiman et al. [5]. Writing $o_i = (o_i^{\parallel}, o_i^{\perp})$ for the offset of sole point $i$ from the foot centre in the foot's own yaw frame, $a = 85$ mm for the sole half-length and $b = 30 - 2 = 28$ mm for the half-width less a numerical margin,
$$
\sum_{i \in j} \big(\pm o_i^{\parallel} - a\big) f_{z,i} \le 0, \qquad
\sum_{i \in j} \big(\pm o_i^{\perp} - b\big) f_{z,i} \le 0 ,
$$
which is $|\text{CoP}^{\parallel}_j| \le a$ and $|\text{CoP}^{\perp}_j| \le b$ written linearly in the normal forces. The safety margin for disturbance rejection is placed on the ZMP *reference* (Section 4.3), not on this box, so the full sole remains available to the MPC.

**Cost.** The MPC minimises
$$
J = \sum_{k=1}^{N} \left\| x_k - x^{ref}_k \right\|^2_{Q_k} + \epsilon \sum_{k=0}^{N-1} \|u_k\|^2 ,
$$
with diagonal $Q$ (roll and pitch 500, yaw 300; CoM horizontal 120, height 1000; angular velocity 10; linear velocity 60) and $\epsilon = 10^{-6}$. The force term only selects an even force distribution from the redundant eight-point null space and must stay several orders of magnitude below the state weights. The height weight at step $k$ is scaled by the fraction of sole points planted at that step, so that single support does not spend force on a height that one foot cannot hold. The reference translates: the CoM reference at step $k$ is $c_{ref} + \dot c_{ref}\, k\Delta t$ and the yaw reference is $\psi + \dot\psi_{ref}\, k\Delta t$. A fixed position target combined with a non-zero velocity target is contradictory, and with it the position term brakes every step.

The problem is condensed into a single QP in the 144 force variables, with 240 friction and normal-force rows and 48 centre-of-pressure rows. Only the first force block is applied. The weighted residual of each of the twelve state channels, summed over the horizon, is exported to the telemetry as the *state cost* (Section 6.2).

### 4.5 Whole-body QP

The WBC follows the weighted-task formulation used in MPC + whole-body-control stacks [43] and in torque-controlled humanoids [6], [54]. Unlike Herzog et al. it uses one weighted QP rather than a strict hierarchy [55], [6].

**Decision variables and constraints.** The decision vector is $z = [\ddot q^\top, f^\top]^\top \in \mathbb{R}^{59}$. Only physics is imposed as a hard constraint:
$$
\hat M_b(q)\,\ddot q + \hat h_b = J_{c,b}^\top f \quad\text{(unactuated floating base)},
$$
$$
|f_x|, |f_y| \le \mu f_z,\qquad s_i f_{min} \le f_z \le s_i f_{max}\quad\text{(friction and normal bounds)},
$$
$$
-\tau_{max} \le \hat M_a \ddot q + \hat h_a - J_{c,a}^\top f \le \tau_{max}\quad\text{(torque limits on the implied joint torque)}.
$$

**Tasks.** Every task is a weighted quadratic residual:
$$
\min_{z}\ w_c \left\| J_c \ddot q + \dot J_c \dot q + k_{c}\, J_c \dot q \right\|^2
+ \sum_{t} w_t \left\| J_t \ddot q + \dot J_t \dot q - a_t^\star \right\|^2
+ w_f \left\| f - f^{MPC} \right\|^2
+ \lambda \|z\|^2 .
$$
The first term holds planted sole points still, with pure velocity damping $k_c$; it is a heavily weighted cost ($w_c = 2000$) rather than an equality constraint. This is a deliberate departure: as a heel lands with residual velocity, a hard "hold this point" row can make the program infeasible at the instant the robot most needs an answer, whereas with the contact task in the cost, $f = 0$ with a free-falling $\ddot q$ always satisfies the remaining constraints, and the solution degrades smoothly. The task set $t$ comprises:

| Task | Target | Weight | Notes |
|---|---|---|---|
| CoM / base position | $a^\star = K_p (c_{ref} - c) - K_d (\dot c - \dot c_{ref})$, saturated at 20 m/s² | 10 | applied to the floating-base acceleration; the horizontal stiffness is halved while walking; the vertical stiffness is scaled by the planted fraction |
| Torso orientation | PD on the orientation error to the yaw reference, saturated at 40 rad/s² | 40 | both base tasks are scaled by the planted fraction, floored at 0.4 |
| Joint posture | PD toward the nominal stance, saturated at 200 rad/s² | 0.02 planted legs (damping only), 5 airborne legs, 5 waist and arms | a planted leg is already pinned by its contacts |
| MPC force tracking | $f^{MPC}$ | $5 \times 10^{-3}$ | lets the WBC deviate from the plan when kinematics require |
| Swing-foot position | arc acceleration + PD on position and velocity, minus $\dot J \dot q$, saturated at 120 m/s² | 500 | the actuator of the footstep plan; must be in the same decade as the contact task or the QP will not lift the foot |
| Swing-foot orientation | yaw tracked throughout; roll and pitch levelled after 70 % of swing, saturated at 80 rad/s² | 40 | before 70 %, ankle pitch and roll are held at the stance pose |
| Hip-yaw hold | foot square to the torso yaw reference | 30 | while walking, both legs |
| Stance-knee guard | light bias to a height-dependent knee angle; stiff hold if flexion drops below 15° | 12 / 200 | the knee table comes from a foot-fixed inverse kinematics solve over CoM heights; it prevents the height task from driving the knee through its stop |
| Sole levelling (stand) | flatten a sole that landed toe-first | 40 | only when exactly one foot is partially planted |

The joint torques are recovered from the solution as $\tau = \hat M_a \ddot q + \hat h_a - J_{c,a}^\top f$ and clipped to the (belief-scaled) limits.

### 4.6 Belief-aware actuation

The controller plans joint torques. The actuator command is obtained by dividing each joint torque by the believed actuator strength relative to its calibration, $\rho_i = \hat g_i / g_i^0$, where $\hat g_i$ is the believed gear ratio (equivalently, torque constant) and $g_i^0$ its nominal value:
$$
u_i = \operatorname{clip}\!\left(\frac{\tau_i}{\rho_i},\ -u_{max,i},\ u_{max,i}\right),
\qquad
\tau_{max,i} = \rho_i\, u_{max,i}.
$$
The torque-limit rows of the WBC use $\tau_{max,i}$, so a motor believed to be weak is both commanded harder and planned with less authority. An actuator believed dead ($\hat g_i = 0$) receives a zero command. The plant's true gear is never read.

### 4.7 Summary of borrowed elements and modifications

| Element | Source | Modification in LOKA's plant |
|---|---|---|
| Convex single-rigid-body MPC over contact forces | Di Carlo et al. [4]; biped extension by Li and Nguyen [52] | Four points per sole; contact schedule and swing-posed contact positions from the gait preview, rotated with planned foot yaw; linearisation about the previous force plan's CoM rollout [53]; height weight scaled by planted fraction; translating position reference; per-channel cost exported |
| Centre of pressure inside the sole | Sleiman et al. [5] | Linear per-foot CoP box in the foot's yaw frame on the full sole; the safety inset lives in the ZMP reference instead |
| Short horizon justified by a gait reference | Galliker et al. [44] | Adopted as a design argument only (0.3 s horizon); the solver remains the convex QP |
| MPC + WBC split | Kim et al. [43] | Single weighted QP; contact as a weighted cost; torque limits on the implied torque; belief-scaled torque limits; support-scaled base tasks |
| Weighted whole-body control on a humanoid | Herzog et al. [6]; Wensing and Orin [54] | Weighted rather than hierarchical [55]; swing-foot Cartesian and orientation tasks with analytic feed-forward; knee guard; hip-yaw hold |
| DCM walking pattern | Englsberger et al. [2], [3] | Chain anchored at the support foot, five-step preview with terminal rest; double-support ZMP transfer; opening anticipatory transfer; first-order CoM filter on plan state; saturated tracking error |
| ZMP inset inside the sole | Kajita et al. [1] (linear-inverted-pendulum argument) | 12 mm inset; Kajita's preview controller is not used |
| Capture-point stepping | Pratt et al. [45] | Partial gain; error measured as the offset past the target foothold; clamps, stride box, foot separation, late-swing reach budget, freeze fraction |
| Gait transitions | Open-source whole-body humanoid MPC [49] | Turn-settle dwell and speed-dependent cadence (logic re-implemented; schedule measured on this plant) |
| ALIP stepping (alternative stack) | Xiong and Ames [50]; Gong and Grizzle [51] | Nominal chain step plus ALIP correction on the predicted pre-impact error |
| Not adopted | Step-timing adaptation [56]; footstep location as an MPC decision variable [57] | Step timing was implemented and disabled after it degraded straight walking and push recovery on this plant; footstep optimisation inside the MPC was considered and not built |

### 4.8 Interface exposed to the orchestrator

The low-level controller exposes three groups of quantities. All edits are applied between two control ticks and are clamped to the listed bands. None of them changes the structure of either QP.

**Table 1. Task targets.**

| Target | Meaning | Bound |
|---|---|---|
| CoM height | CoM height above the sole plane | [0.50 m, the nominal stance height of about 0.70 m]; while walking, at least 0.60 m and slewed at 0.05 m/s |
| Torso yaw offset | Torso yaw relative to the feet while standing; ignored while walking | ±0.5 rad |
| Forward lean | CoM offset forward of the support centre (in the travel frame while walking) | 55 % of the current support margin in that direction, and at most 6 cm |
| Lateral lean | CoM offset to the left | same |

**Table 2. Gait-policy parameters.** The default values are those of the 0.25 m/s walk.

| Parameter | Band | Default | Notes |
|---|---|---|---|
| Mode | stand, walk, tread, limp | stand | stand cancels any goal; walk with no speed defaults to 0.25 m/s |
| Speed | [0, 0.30] m/s | 0.25 | the cruise speed when a goal is active |
| Heading | [−π, π] rad | 0 | world-frame goal; setting it alone cancels an active goal |
| Cycle period | [0.55, 1.10] s | 0.72 | follows the speed schedule unless set |
| Duty factor | [0.78, 0.85] | 0.79 | follows the speed schedule unless set |
| Maximum step length | [0.08, 0.40] m | 0.30 | |
| Swing clearance | [0.02, 0.08] m | 0.045 | |
| Stance width | [0.18, 0.28] m | 0.23 | follows the speed schedule unless set |
| Capture gain | [0.30, 1.00] | 0.70 | fraction of full capture-point placement |
| Speed ramp | [0.15, 1.60] m/s² | 0.50 | |
| Foothold freeze fraction | [0.20, 0.90] | 0.70 | fraction of swing after which the foothold no longer moves |
| Turn rate | [0.10, 0.80] rad/s | 0.40 | subject to the 0.12 rad per-step cap |
| Goal x, goal y | [−20, 20] m | – | setting either activates go-to-goal |
| Goal active | {0, 1} | 0 | |
| Goal tolerance | [0.05, 0.50] m | 0.15 | arrival radius |

The bands were measured on this plant with closed-loop 6 s straight walks at 0.25 m/s, varying one parameter at a time: outside them the robot falls within a few seconds (for example, a duty factor of 0.74, a period of 0.45 s, a clearance of 0.10 m, a width of 0.32 m, or a capture gain of 0).

**Table 3. Editable belief state.** Edits are written into the controller's private model, by the name of the body, surface or actuator in the robot description.

| Object | Attribute | Consequence in the controller |
|---|---|---|
| Body | Mass | The body's rotational inertia is scaled by the same ratio and the composite inertias are recomputed. The total mass used by the MPC and the gait planner is updated. Once the belief has settled (not during a search probe), the nominal stance geometry (CoM offset over the feet and the knee-height table) is re-measured. |
| Body | Centre of mass (3-vector in the body frame) | Composite CoM and inertia are recomputed; affects the centroidal state, the WBC dynamics and, once settled, the nominal stance. |
| Surface (floor or sole) | Sliding friction | The friction coefficients of both QPs are capped at $\hat\mu / \sqrt{2}$, which keeps the square friction pyramid inside the believed friction cone. A belief at or above nominal restores the configured values. |
| Actuator | Gear ratio (torque constant) | Command scaling and torque limits as in Section 4.6. Zero disables the actuator. |

---

## 5. Anomaly Detector

### 5.1 Rationale

A fault that the controller's model does not know about (an unexpected payload, a weakened motor, a slippery floor) first appears as a mismatch between the forces the robot actually exerts and the forces its model says the observed motion requires. It becomes a tracking error, a tilt or a fall only later, after the feedback loops have been saturated. Kinematic tracking error is also a poor detector for a robot whose commanded motion changes: an intentional crouch, lean or walk produces tracking transients that a correct model would fully explain. LOKA therefore detects anomalies in the space of generalised forces, using the residual between measured forces and the inverse dynamics of the belief model. This is closely related to model-based residuals for collision and contact detection on manipulators [58], [59] and humanoids [60], but is used here for a different purpose: to detect *where the controller's own model is wrong* and to drive the identification of the responsible parameter (Section 9).

The detector tracks three signals: the effort bias, the motor deficit and the stance slip.

### 5.2 Effort bias

At every control tick, after the plant has stepped, the *measured generalised force* is
$$
\tilde\tau = S^\top \tau_{m} + J_c^\top f_{m} + \tau_{ext},
$$
where $\tau_m$ is the torque the actuators actually applied, $f_m$ the measured contact forces and $\tau_{ext}$ any externally applied generalised force known to the runtime. The belief model's inverse dynamics at the measured position, velocity and acceleration is computed with the recursive Newton–Euler algorithm [61],
$$
\hat\tau = \hat M(q)\,\ddot q + \hat h(q, \dot q).
$$
Their difference $\Delta = \tilde\tau - \hat\tau \in \mathbb{R}^{35}$ is zero when the belief explains the motion. It has a floating-base part $\Delta_b \in \mathbb{R}^6$ and an actuated part $\Delta_a \in \mathbb{R}^{29}$.

A torso payload or a centre-of-mass error acts mainly on the floating base, which has no actuator; a residual expressed only on joints would therefore miss most of it. LOKA resolves the base part through the contacts: it finds the minimum-norm set of contact forces that would account for the base residual,
$$
f_\Delta = \arg\min_{f} \left\| J_{c,b}^\top f - \Delta_b \right\|_2 = \big(J_{c,b}^\top\big)^{+} \Delta_b ,
$$
and maps those forces onto the actuated joints through the actuated block of the contact Jacobian. The *per-actuator effort deviation* is
$$
e = J_{c,a}^\top f_\Delta + \Delta_a \in \mathbb{R}^{29}.
$$
Each component has a direct physical reading. A persistent positive value means the plant is heavier along that joint than the belief (mass or CoM too low); a persistent negative value means the belief overshoots the plant; a symmetric positive pair at the knees indicates an additional mass on the trunk; a left–right difference of more than about 2 Nm between symmetric actuators indicates an asymmetric payload or a lateral CoM offset; a roll residual (waist roll, hip roll, ankle roll) points to the lateral CoM coordinate and a pitch residual (hip pitch, waist pitch, knee, ankle pitch) to the forward one.

The effort deviation is averaged over a window, $\bar e = \frac{1}{N}\sum_{n} e_n$, and summarised by its $\ell_1$ norm, the *effort bias*
$$
\beta = \|\bar e\|_1 = \sum_{i=1}^{29} |\bar e_i| \quad [\text{Nm}].
$$

### 5.3 Motor deficit

Effort bias catches errors in the rigid-body model but cannot distinguish a weak motor from a heavy limb. The motor deficit isolates actuator faults by comparing, per actuator, the torque the controller expected the motor to deliver with what it delivered:
$$
d_i = \rho_i\, u_i - \tau_{m,i},
$$
where $u_i$ is the command and $\rho_i$ the believed strength ratio (Section 4.6), so that $\rho_i u_i$ is the planned joint torque whenever the command is not saturated. A healthy motor gives $d_i = 0$. The windowed $\ell_1$ norm $\|\bar d\|_1$ is the *motor deficit*. Where the command is large enough to divide by ($|u_i| > 1$), the ratio $d_i / u_i$ estimates the error of the believed gear relative to the true one and is the signal used by the gear search (Section 9).

### 5.4 Stance slip

The stance slip is the speed, in the ground plane, of the fastest sole point that is currently planted:
$$
\sigma = \max_{i \in \mathcal{P}} \left\| \big(J_{c,i}\, \dot q\big)_{xy} \right\|,
$$
where $\mathcal{P}$ is the set of sole points within 5 mm of the estimated ground. It is averaged over the same window as the other signals. A healthy walk stays below about 0.11 m/s; a low-friction floor does not. Persistent slip with a quiet motor deficit is attributed to friction rather than to a motor fault.

### 5.5 Windowing, triggering and immediate reflexes

**Windows.** While standing, all three signals are averaged over the last 100 control ticks (0.2 s). While walking, they are averaged over two full gait cycles (about 1.5 s at the default cadence), which removes the periodic forces of the gait from the mean. Nothing is published, and nothing can trigger, until the window is full, so a start-up transient is not mistaken for a fault. The window and all counters are reset when the robot switches between standing and walking.

**Trigger.** A tick is *out of bounds* if
$$
\beta > \beta_{th} \quad\lor\quad \|\bar d\|_1 > d_{th} \quad\lor\quad \bar\sigma > \sigma_{th},
$$
with $\beta_{th} = 5$ Nm (standing and walking), $d_{th} = 0.75$ Nm and $\sigma_{th} = 0.15$ m/s in the default configuration. These sit just above the noise floor of the healthy plant; for reference, a knee motor at 75 % strength produces a motor deficit of about 4 Nm while standing, and a 10 kg torso payload produced an effort bias of 7.4 Nm in the run of Section 6.4. A failure episode opens after 250 consecutive out-of-bounds ticks (0.5 s), and closes only after the signals have been back in bounds for more than 0.5 s and no belief search is running.

**Immediate reflexes.** Two actions are taken by the runtime itself, without waiting for the language model:

1. *Stand on fault.* If the robot is walking when a fault opens, the runtime commands the gait planner to stand before the orchestrator is invoked. The planner finishes the step in the air and stops (Section 4.3). The orchestrator is told that this happened.
2. *Walk lock-out.* While any of the three signals is out of bounds, the gait planner rejects any request to walk, whether it comes from the orchestrator directly or from a go-to-goal re-engagement. The orchestrator can only resume walking once its fix has brought the signals back in bounds.

### 5.6 Auxiliary friction estimate

Friction does not appear in the effort residual. It is identified from the contact forces themselves: a planted point that slides sits on the boundary of its friction cone, so the ratio of tangential to normal force there reads the friction coefficient. The runtime keeps a 2 s window of this ratio over sole contacts that are planted, sliding faster than 0.05 m/s and loaded with more than 20 N, and freezes a copy of the window at the moment a fault trips. A friction search (Section 9) reads the 75th percentile of the frozen window, provided it contains at least 50 samples; the percentile rather than the mean is used because partly stuck contacts read below the cone.

### 5.7 Computation in simulation and on hardware

All three signals use only quantities that a torque-controlled humanoid can measure; the plant's inertial parameters are never read, even in simulation.

| Quantity | In simulation | On a real robot |
|---|---|---|
| Joint position and velocity; floating-base pose and twist | Ground-truth plant state | Joint encoders, plus a floating-base estimator fusing leg kinematics and the inertial measurement unit [62], [63] |
| Generalised acceleration | The plant's acceleration at the step just taken | Numerical differentiation with low-pass filtering; alternatively, a generalised-momentum observer formulation removes the need for acceleration altogether [58], [59] |
| Applied actuator torque | The simulator's actuator generalised force (the stand-in for current-derived torque) | Motor current times torque constant times gear ratio, as reported by the motor drivers |
| Contact forces | The simulator's constraint generalised force (contact solver output) | Foot force/torque or pressure sensors. Where these are unavailable and contact forces are inferred from the belief model's own base dynamics, the floating-base residual vanishes by construction and the effort deviation reduces to the joint-space residual, which still responds to mass error but loses the independent check through the base |
| Planted set | Belief kinematics at the measured state: sole point within 5 mm of the estimated floor | Contact estimation from foot sensors or probabilistic fusion of kinematics and dynamics [10], [64] |
| Slip velocity | Contact Jacobian of the belief model times measured velocity | Same, with the estimated base velocity; this is the slip signal used for slippery-ground locomotion on quadrupeds [11], [65] |
| Commanded torque | Output of the WBC after belief scaling | Same |
| Friction ratio | Contact solver forces at sliding sole contacts | Three-axis foot force sensing at sliding contacts |

The detector requires no learned component and no training data. Its thresholds are the only tuned quantities; they sit just above the noise floor of the healthy plant.

---

## 6. Telemetry Compressor

### 6.1 Purpose

The controller produces about 500 states per second, each with hundreds of numbers. The compressor turns the most recent two seconds of this stream into a short, labelled text report that a language model can read reliably, in the spirit of REFLECT's hierarchical summaries of robot experience [38] but online, for proprioception, and with the anomaly signals of Section 5 at its centre. Three choices guide it:

1. **Explain each signal in place.** Every section carries a one-line definition of its numbers and their sign conventions, so that the model does not have to remember them from the system prompt.
2. **Rank by relevance.** Per-actuator quantities are sorted by magnitude, so the joints carrying the fault come first.
3. **Report direction explicitly.** Speeds that the model would otherwise have to sign itself (motion along the heading, motion toward the goal) are computed and labelled.

### 6.2 Sections

The report consists of the following sections. Sections without content are omitted.

**Full state (snapshot at the invocation).** Simulation time; pelvis position in the world frame; torso roll, pitch and yaw in degrees; orientation quaternion; linear and angular velocity; unsigned planar speed. Then the gait command (mode, speed, heading, whether the clock is stepping), the safe band of every gait parameter with its default (Table 2), and the heading quantities: the torso yaw reference, the heading goal, the raw heading error, a *gated* heading error that stays zero until the time a turn needs at the configured turn rate has elapsed, and the cross-track drift (a leaky integral of lateral velocity in the heading frame).

**Progress (while walking or when a goal is active).** Speed along the heading (positive in the commanded direction). When a goal is active: the goal position, the distance to it, the *closing speed* (velocity projected onto the direction of the goal) and a tag that reads *toward goal*, *walking backwards* or *moving away from goal*.

**Effort deviation.** The effort bias $\beta$, the current mode (standing or walking) and its threshold, and the current out-of-bounds streak against the streak length. Then all 29 per-actuator values of $\bar e$ by decreasing magnitude, with their signs. Then the motor deficit with its threshold and up to four actuators whose individual deficit exceeds 0.2 Nm. Then the stance slip with its threshold.

**Prop contact (only if the robot touched an object in the window).** For each object: its kind, the peak normal force, the fraction of the window spent in contact, whether contact is ongoing, and the robot links involved. The section states explicitly that this force enters the effort residual as an external load and is not a model error, so that a collision with furniture is not misdiagnosed as a payload.

**State cost.** The MPC's weighted residual per state channel (roll, pitch, yaw, three CoM coordinates, three angular and three linear velocities), averaged over the detector's window. This tells the model which part of the centroidal state the controller is failing to track.

**Camera frame (VLM mode only).** The pose of the head camera at the instant the image was captured (Section 7.3).

### 6.3 Context sections

The compressor's report is the telemetry part of every user turn. The orchestrator turn wraps it with the following context, always in this order: the locked mission directives; the self-assigned plan; the active listeners; the live MPC parameters and live gait parameters; the current model belief, as a list of every edited parameter with its nominal and believed value and the time it was set. Failure-episode turns additionally begin with the episode header, the turn budget, any note that the runtime stopped a walk, and the *attempted fixes* of the episode, including the results of finished belief searches.

### 6.4 Example

The following is a truncated user turn from a logged evaluation run in which a 10 kg mass was added to the torso of the plant while the robot walked toward a goal at (3, 0) m. The runtime stood the robot, opened a failure episode, and invoked the orchestrator. Ellipses mark omitted lines, and some long lines are shortened. The run was logged before the self-assigned plan, motor deficit and stance slip were added to the report; those lines are shown in their current format with their values elided. The listener line is also shown in its current form.

```text
--- FAILURE EPISODE (round 1, t=4.59s) ---
Initial instability detected. Queue Belief_Searches for the model mismatch.
Orchestrator turn 1 of 6 this episode. The count resets when the bias returns to normal.
The runtime already set gait.mode=stand and gait.speed=0 before this call. [...]
Primary objective: Hold a robust standing balance on the Unitree G1, [...]

## ATTEMPTED FIXES
None. This is the first intervention for this failure episode.

## LOCKED MISSION DIRECTIVES
No operator mission directives. The standing charter applies: hold balance, and wait for a goal.

## SELF-ASSIGNED PLAN
None. When a directive needs more than one move [...], write one under Plan [...]

## ACTIVE LISTENERS
None armed. A periodic check still runs every 15 s.

## CURRENT MPC TUNABLES
  mpc.friction_mu: 0.5
  mpc.weight_orientation_rp: 500
  [...]
## CURRENT DCM TUNABLES
  gait.capture_gain: 0.7
  gait.duty_factor: 0.81
  gait.goal_x: 3
  gait.mode: 0 (stand)
  gait.speed: 0
  [...]

## CURRENT MODEL BELIEF
No active mutations. Internal model matches nominal MJCF.

## NEW TELEMETRY
FULL STATE (snapshot at this invoke)
  t=4.59s
  pelvis xyz [m] (x is pitch, y is roll, z is height): +0.587 -0.032 +0.789
  attitude [deg] roll=y pitch=x yaw=heading: -2.0 -1.4 +0.4
  linear vel xyz [m/s]: +0.236 -0.163 +0.023
  angular vel xyz [rad/s]: -0.598 -0.425 -0.290
  planar speed: 0.287 m/s  (unsigned; use along heading and closing speed for direction)
  gait cmd: mode=0.0 speed=0.0 heading=0.0 walking=False
  gait safe band (out of range is clipped): gait.speed [0, 0.3] default 0.25; [...]
  yaw_ref=0.000 heading_goal=0.0 heading_err_raw=0.0 heading_err_gated=0.0 cross_track=-0.0035

EFFORT DEVIATION [Nm]
  Measured force minus the belief model's inverse dynamics, resolved into actuators
  through the foot contacts. [...] A roll residual is a com y search. A pitch residual
  is a com x search. Height is a com z search.
  |bias|=7.38  mode=walking  threshold=5.0
  consecutive out of bounds: 250 / 250
  right_knee: +1.74
  left_knee: +1.74
  left_ankle_pitch: -1.38
  right_ankle_pitch: -1.27
  waist_pitch: -0.49
  right_hip_roll: -0.38
  [... 23 further actuators, all below 0.15 Nm ...]
  |motor|=[...]  threshold=0.8  commanded torque minus applied torque (current). [...]
  stance slip=[...] m/s  threshold=0.15  fastest planted sole, ground plane. [...]

STATE COST
  roll: 1.022   pitch: 0.3274   yaw: 0.2312
  x: 0.2421     y: 0.0243       z: 0.1558
  wx: 0.4271    wy: 0.2222      wz: 0.0958
  vx: 0.03925   vy: 0.7745      vz: 0.04539
```

The report points to the fault without naming it: the two knee residuals are positive, equal and the largest, and the remaining residuals are sagittal and nearly symmetric, which indicates added mass on the trunk rather than a lateral offset. (With the current report, a quiet motor deficit and low slip would additionally rule out a motor or friction fault.) The orchestrator replied by queueing a torso mass search followed by a forward CoM search and arming a wake-up for when the residual returned in bounds (Section 8.5). The search set the believed torso mass from 9.6 kg to 19.2 kg (the true value was 19.6 kg) and the effort bias fell from 7.4 Nm to 0.8 Nm.

---

## 7. High-Level Orchestrator

### 7.1 Model and invocation

The orchestrator is a general-purpose LLM, or a VLM when the camera is enabled, accessed through a hosted chat-completion API. It receives a system message and a user message (plus, within the memory window, the earlier exchanges of the same conversation) and must reply with YAML only. The reply is parsed with a safe YAML loader after stripping any Markdown code fences. The model identifier is a configuration parameter; no fine-tuning or in-context examples beyond the schema example in the system prompt are used.

### 7.2 When the orchestrator is invoked

The orchestrator is never polled at the control rate. It is invoked by exactly four kinds of events, checked every control tick in the following order:

1. **Operator directive.** A natural-language instruction from the operator (for example, "go to the chair"). It is stored as a *locked mission directive*: it is shown on every later turn, and nothing the orchestrator writes can edit, complete or remove it. If the sentence is a navigation request containing a coordinate pair, the parsed world point is shown beside it.
2. **Self-assigned listener.** A condition that the orchestrator itself armed on an earlier turn (Table 4), together with the message it wants to receive when the condition becomes true. Listeners are one-shot: a listener is removed when it fires. Several listeners that fire together are delivered in one turn.
3. **Anomaly.** A failure episode opened or continued by the anomaly detector (Section 5.5).
4. **Heartbeat.** A periodic check every 15 s, issued only when no failure episode is open and no other invocation has happened in that interval. It guards against a goal that was abandoned silently, for example because the robot was pushed away and no listener covered that case.

**Table 4. Listener conditions.**

| Condition | Parameters | Fires when |
|---|---|---|
| Near a point | world x, y; radius (default 0.4 m) | the pelvis is within the radius of the point |
| Heading error below | tolerance (default 0.15 rad) | the heading error is below the tolerance |
| Speed below | speed (default 0.05 m/s) | the planar speed is below the value |
| Pelvis below / above | height | the pelvis height crosses the value |
| After a delay | seconds | the delay has elapsed since the listener was armed |
| Residual in bounds | none | the effort bias is at or below the mode threshold and no belief search is running or queued |

**Gating.** An invocation is dispatched only if (i) no orchestrator call is in flight; (ii) at least 2 s of simulated time have passed since the last reply was applied (the cooldown, which lets the plant respond to an edit before it is judged); and (iii) no belief search is running or queued, since a search changes the belief deliberately and the orchestrator has delegated that interval. Operator directives are held until the orchestrator is idle and out of cooldown, and listeners are only evaluated then, so a listener whose condition is still true fires on the first such tick.

**Failure episodes and conversations.** Anomaly invocations belong to a *failure episode*, a multi-turn conversation that lasts from the opening of a fault until the detector reports the signals back in bounds for more than 0.5 s. An episode may use at most six orchestrator turns; after the last, the orchestrator is not called for that episode again, and the plant continues on the last applied configuration. Directive, listener and heartbeat invocations belong to a separate, long-lived *operator session*. Each conversation keeps its earlier exchanges in the context only while it has completed fewer than three turns; after that, each turn is composed from the current state alone, and the turn says so. What persists regardless of the memory window is the state the orchestrator wrote: the plan, the listeners, the belief, the gait and task targets, and the attempted fixes of the current episode while it is still within the memory window. This is why the self-assigned plan exists (Section 8.2).

### 7.3 What the orchestrator sees

**System message.** The same for every call within a run:

1. *Charter.* The orchestrator's role and its five duties (set task targets toward the directive; correct the belief by queueing searches; decide when to stand and walk; arm listeners; maintain its own plan), the semantics of goal-directed walking and turning in place, the meaning of each telemetry section and of each anomaly signal, how residual patterns map to search parameters (Section 5.2), the episode and memory rules, the reply schema with an annotated example, and a short list of problem-solving heuristics (for example, start a walk at 0.15 m/s; route around obstacles with waypoints that keep each heading change below 45°; search the mass of a body before its CoM coordinates).
2. *Robot model reference.* Generated from the robot description: the body tree with parent links and offsets; every joint with its type, its index in the position and velocity vectors and its range; every actuator with its joint, gear and command range; the list of mutable attributes (actuator gear, surface friction, body mass and CoM); and the kinematic part of the robot description file itself (MJCF). Together these make up most of the system message, which was about 62,000 characters long in the logged VLM runs.
3. *Camera description (VLM mode only).* The head camera's mount, intrinsics (512 × 512 pixels, 70° vertical field of view, pitched 30° down), the pixel-to-floor projection formulas from pinhole geometry, reference floor distances per image row, the size of the floor checkerboard, a catalogue of the objects that may appear (from HumanoidBench [66]), and navigation guidance for using the image.

**User message.** The header for the invocation type, then the context and telemetry of Sections 6.2 and 6.3. In VLM mode it also carries one RGB image rendered from the head camera at the instant of dispatch, together with the camera's world pose. Only the current turn carries an image; earlier turns are kept as text.

**What the orchestrator does not see.** It is never told which fault was injected, when, or by how much. It does not see the plant's physical parameters, only its own belief. It does not see the footstep plan, the contact schedule or the QP internals beyond the per-channel MPC cost.

### 7.4 Reasoning before acting

Every reply must begin with a semantic-state section containing a one-line *hypothesis* (the single search or target change the reply makes) and a free-text *analysis* explaining what the telemetry shows and why the change follows. A reply without it is rejected as a whole. This makes the model state its diagnosis before it edits anything, following chain-of-thought [34] and reason-then-act prompting [35]; it also gives a human reviewer, and the model's own later turns (through the attempted-fixes section), a record of why each change was made.

### 7.5 Asynchronous, non-blocking execution

When an invocation is dispatched, the control thread composes the message list, copies it, and starts a background worker thread. It then returns immediately and continues to step the plant. The worker renders the camera image if needed (from a copy of the plant state taken at dispatch, in a separate rendering process, so that rendering never runs on the control thread), calls the API, parses the reply and places it on a result queue created for this dispatch only; a reply that arrives after it has been abandoned can therefore never be applied to a later turn.

At the start of every control tick the control thread checks the queue without blocking. When a reply is present, it is applied in full *between two ticks*: the tick that follows sees either none or all of the edits, never a partial state. Applying edits does not re-create either QP, because the problem structures are fixed (Section 4.1); parameter edits only change the numbers that the next solve reads, and the few constants that are baked into constraint matrices (friction coefficients, force bounds) are rewritten in place. Gait-timing edits are deferred to the next step boundary and slewed (Section 4.3). During the whole round trip, the robot keeps executing the last applied configuration. If the API call fails or the reply cannot be parsed, nothing is applied, and the cooldown starts as for a normal reply.

For evaluation, the reply can be applied in one of three ways: as soon as it arrives in wall-clock time (the deployment behaviour); after a fixed amount of *simulated* time, with the simulation blocking until the reply arrives if it is late, which makes runs reproducible regardless of API speed; or with zero simulated latency, which isolates the quality of the orchestrator's decisions from its latency. A reply that does not arrive within 120 s of wall-clock time is discarded.

---

## 8. Scratchpad: Belief State, Task State and Plan

### 8.1 Schema

The orchestrator's reply is a YAML document with the sections in Table 5. Every section except the semantic state is optional, and omitting a section leaves the corresponding state unchanged; this keeps replies short and makes every edit explicit. Section and field names that the model writes in the wrong case are normalised before parsing.

**Table 5. Scratchpad sections.**

| Section | Content | Effect |
|---|---|---|
| Semantic state (required) | Hypothesis; analysis | Logged; shown back in the episode's attempted fixes |
| Task targets | Any subset of Tables 1 and 2 | Clamped and applied to the controller |
| Belief searches | Ordered list of (kind, name, optional axis); kinds: mass, CoM, friction, gear | Queued for the belief-search executor (Section 9) |
| Listeners | List of (condition, parameters, message) from Table 4 | Replaces the armed set; an empty list clears it |
| Plan | Goal, rationale, steps, current step, notes | Creates or edits the self-assigned plan |
| Model mutations | List of (object type, name, attribute, value) | Written directly into the belief (Table 3) |

### 8.2 Capabilities

**Edit task targets and the gait policy.** The orchestrator can change the CoM height, lean and standing yaw, switch between stand, walk, tread and limp, set speed, heading and cadence, swing clearance, stance width, capture gain and turn rate, and hand the planner a world-frame goal (Section 4.3). A goal turns walking into a closed-loop behaviour of the plant: once set, the robot walks there, slows, stops inside the tolerance and walks back if pushed away, without further orchestrator calls. Turning on the spot is a walk with zero speed and a new heading. The gait parameters are also the orchestrator's means of adapting the walk to the environment and to an identified fault: the prompt suggests raising the swing clearance and slowing down to step over a low obstacle, and walking more slowly with longer double support after a friction search has lowered the believed friction. Footholds remain on the floor plane, so stepping onto a raised surface is outside the interface.

**Start belief searches.** The orchestrator names *which* parameter it believes is wrong (for example, the mass of the torso, the lateral CoM of the left shoulder, the floor friction or the gear of the right knee), in priority order. It does not supply the value; the executor of Section 9 identifies it. The system prompt instructs the model not to write a model mutation for a parameter it has queued, and to search a body's mass before its CoM coordinates.

**Edit the belief directly.** The model-mutation section writes an absolute value (a mass in kilograms, a CoM vector, a friction coefficient, a gear ratio) into the belief. It is intended for edits that cannot be identified by search, the main one being declaring an actuator dead (gear zero), which disables it (Section 4.6). Names are resolved against the robot description, tolerating case errors; unknown names are rejected.

**Arm listeners.** The orchestrator decides when it wants to be called next and with what message (Table 4). Because a call takes seconds and the robot keeps moving, the prompt asks the model to arm listeners *early* (for example, a proximity listener short of a waypoint rather than on it). The residual-in-bounds listener lets the model queue a search and ask to be woken once it has worked, at which point it resumes the interrupted walk.

**Maintain a self-assigned plan.** The plan is the orchestrator's own record of how it intends to satisfy the locked directives: a goal, a rationale, up to 12 steps, the index of the current step and free-text notes. It is shown on every turn with each step marked done, current or next, together with a revision number and the times it was written and last edited. A reply may edit any subset of fields (for example, only advance the current step), replace the steps (which resets the current step to the first unless one is given), or clear the plan. Because the conversation history is cut after three turns, the plan is the only place where the reasoning behind an earlier decision survives, for example why a waypoint was placed 0.9 m to the side of the direct line. It is the slow loop's equivalent of a reasoning scratchpad [36], made persistent and editable.

### 8.3 What the orchestrator cannot do

The interface rejects or ignores, independent of the prompt:

- **Controller costs, gains and constraints.** No parameter of either program can be written by the orchestrator: not the MPC's state weights, force weight, horizon, time step or force bounds, and not the WBC's task weights, stiffnesses and dampings, contact weight, force-tracking weight or acceleration limits. A section that tries to set them is ignored. The one controller quantity that follows the orchestrator's actions is the friction coefficient of both programs, and only indirectly, through the believed floor friction (Table 3). These parameters define the certified plant; early experiments in which a language model was allowed to rewrite them wholesale led to falls. Hey Robot! [30] and Language to Rewards [31] give the language model this authority; LOKA deliberately does not.
- **Footholds, contact schedules and trajectories.** These are computed by the gait planner and the MPC. The orchestrator can only change the policy parameters that shape them. This contrasts with interfaces in which a language model writes the contact pattern directly [29].
- **Code.** There is no execution path for generated code, in contrast with code-as-policy [25] and reward-code generation [31], [39].
- **Out-of-band values.** Every task target and gait parameter is clamped to its band (Tables 1 and 2); unknown keys and non-numeric values are dropped with a warning, without aborting the rest of the reply.
- **Mission directives.** The operator's directives are locked; directive or objective sections in a reply are ignored.
- **Walking while a fault is open.** Requests to walk are refused while any anomaly signal is out of bounds (Section 5.5).
- **Interrupting a search.** No orchestrator call is made while a belief search is running or queued (Section 7.2).

### 8.4 Persistent state

Across turns, the following state persists and is shown back to the orchestrator: the locked directives; the plan with its revision history; the armed listeners; the gait policy and task targets; the belief; and, within a failure episode, the attempted fixes together with the results of finished searches (for example, that the torso-mass search finished at 19.2 kg with the residual settled). Mutations accumulate as absolute values per parameter, so the latest value for a given body, surface or actuator replaces the earlier one.

### 8.5 Example reply

The reply to the turn shown in Section 6.4:

```yaml
Semantic_State:
  Hypothesis: "Search torso_link mass first, then its forward COM axis if the pitch residual remains."
  Analysis: "The bias is above the threshold and persistent; the runtime has already put
    the gait in stand, so I am not commanding a gait change. The bilateral knee residuals
    and near-symmetric sagittal pattern point to a shared model mismatch rather than a
    lateral imbalance. Search torso mass first while standing, then COM x for the
    pitch-related residual."
Belief_Searches:
  - kind: mass
    name: torso_link
  - kind: com
    name: torso_link
    axis: x
Listeners:
  - kind: bias_under_threshold
    message: "Check the standing balance after the belief search finishes."
```

---

## 9. Residual-Driven Belief Search

### 9.1 Rationale

Language models are good at qualitative diagnosis from structured evidence ("the knees carry an equal positive residual and the motors are healthy, so the trunk is heavier than believed") and poor at numerical identification under feedback. Identifying inertial parameters is a classical problem [8], [9], but full identification needs persistent excitation that a balancing humanoid cannot safely provide. LOKA splits the problem: the orchestrator chooses one scalar parameter, and a one-dimensional search drives the signed component of the effort residual that this parameter controls to zero, on the robot itself.

### 9.2 Signed error per parameter

Each search kind reads a signed scalar error from the windowed effort deviation $\bar e$ (Section 5.2) or the motor signals (Section 5.3). The sign is chosen so that a positive error means the parameter should increase.

| Kind | Signed error | Reasoning |
|---|---|---|
| Body mass | Mean of the two knee residuals | The knees carry the body's weight; a positive residual means the body is heavier than believed |
| CoM, lateral axis | Waist-roll residual | A lateral CoM offset loads the roll joints |
| CoM, forward axis | Negative mean of the hip-pitch and waist-pitch residuals | A forward or upward CoM shift makes the pitch residual positive, so the parameter is too large |
| CoM, vertical axis | Same as forward, with a much tighter tolerance | The vertical CoM coordinate influences the pitch residual far more weakly |
| Actuator gear | Negative gear-ratio error of the named actuator (Section 5.3) | Positive gear-ratio error means the believed motor is stronger than the real one |
| Friction | Not searched; estimated in one shot (Section 5.6) | Friction does not enter the effort residual |

### 9.3 Unbounded bracketing search

The value is found with an unbounded search in the sense of Bentley and Yao [67]: an expanding phase finds a bracket, then bisection closes it. Starting from the current belief $\theta_0$ (the nominal value, or the latest edit), each probe $\theta_n$ is written into the belief, the plant is given time to settle, and the signed error $\varepsilon_n$ is read.

*Expansion.* While no bracket exists, the parameter moves in the direction of the error's sign with a step that doubles each time. Mass and gear are scaled multiplicatively,
$$
\theta_{n+1} = \theta_n \cdot 2^{\operatorname{sign}(\varepsilon_n)} .
$$
For the CoM, which can be zero or negative, the offset from the nominal coordinate $\theta^0$ doubles, starting from 4 cm:
$$
\theta_{n+1} = \theta^0 + \operatorname{sign}(\varepsilon_n)\, \max\!\big(2\,|\theta_n - \theta^0|,\ 0.04\ \text{m}\big).
$$

*Bisection.* The largest probe with positive error and the smallest probe with negative error form the bracket $[\theta_{lo}, \theta_{hi}]$. Once both exist, the next probe is its midpoint.

*Termination.* The search stops, and writes the final value into the belief, when:

1. the error is within tolerance (0.4 Nm for mass and for the lateral and forward CoM; 0.05 Nm for the vertical CoM; 5 % gear-ratio error for gear);
2. the bracket is narrower than the value tolerance (1 % of the starting mass, at least 0.05 kg; 5 mm for the CoM; 5 % for gear);
3. the next probe would leave the parameter's range (mass 0.05 to 80 kg; CoM within 0.4 m of nominal; gear from zero to the larger of 5 and four times its starting value); or
4. fourteen probes have been made, in which case the probe with the smallest error is kept.

If, during expansion, a probe changed the parameter but the error changed by less than a small tolerance without a sign change, the parameter does not influence the residual; the search stops and *restores* the belief to its value before the search. This protects the belief from the language model naming the wrong parameter.

### 9.4 Probe protocol

**Measurement posture.** Searches are run while standing. If the robot is still finishing a fault stand, the first probe waits until it has stopped. If a search is started while the robot is walking, the gait is switched for the duration of the search to a high-stability *search gait* (stepping in place at zero speed with a wide stance of 0.28 m and a duty factor of 0.85), and restored afterwards.

**Settling.** Between probes, the executor waits for the detector's averaging window to be completely refilled under the new belief (0.2 s standing; two gait cycles walking), plus 0.2 s for the feedback loops and contacts to settle. Each probe therefore costs between 0.4 s and about 1.7 s, and a search of up to fourteen probes completes within seconds to tens of seconds.

**Gear searches.** While a gear search probes, the command scaling of Section 4.6 keeps using the gear value from before the search, so that a probe changes only the residual and not the torque the robot actually receives. The accepted value takes effect on the tick after the search ends.

**Sequencing and reporting.** Searches are executed one at a time, in the order the orchestrator listed them. The orchestrator is not called while the queue is non-empty. Each finished search appends a result line (parameter, final value, reason for termination) to the attempted fixes of the current episode. A search is abandoned and the queue cleared if the robot falls.

### 9.5 Effect of an accepted value

An accepted value enters the belief like any other edit (Table 3). The total mass used by the MPC and the gait planner is refreshed every tick from the belief; once the search queue is empty, the nominal stance geometry is re-measured from the new belief, so that, for example, a heavier torso produces a correspondingly adjusted standing posture. In the example of Section 6.4, the first doubling of the torso mass (9.6 kg to 19.2 kg) already brought the knee residual within tolerance; the subsequent forward-CoM search found its residual within tolerance at the starting value and left the CoM unchanged; and the effort bias fell from 7.4 Nm to 0.8 Nm, after which the orchestrator's residual-in-bounds listener fired.

---

## 10. Integrated Control Flow

Each 2 ms control tick executes the following steps. Only step 4 is on the critical path of balance; the others are bookkeeping that take a negligible fraction of the period, and no step waits on the language model.

1. **Events.** Apply scheduled plant events (fault injections and pushes in simulation).
2. **Drain the orchestrator.** If a reply is waiting, apply its scratchpad (Section 8) and enqueue its belief searches; start the cooldown.
3. **Rebuild the belief.** Reset the belief to nominal and re-apply all accepted edits; refresh the total mass and the friction caps.
4. **Control.** Gait planner, MPC (every tenth tick) and WBC compute joint torques on the belief; torques are converted to actuator commands with the believed actuator strength, dead actuators are zeroed, and the plant advances by two 1 ms physics steps.
5. **Detect.** Compute the effort deviation, motor deficit and stance slip; update the windows, the out-of-bounds streak and the walk lock-out; sample the friction ratio.
6. **Search.** Advance the belief-search executor (Section 9): wait, read the signed error, propose the next probe, or finish.
7. **Record.** Append the state to the 2 s telemetry buffer.
8. **Invoke.** If ungated: dispatch a pending operator directive; otherwise dispatch any fired listeners. If the fault streak is complete: stand if walking, open or continue the failure episode, and dispatch a failure turn within the episode's budget. Otherwise, if the signals have recovered and no search is running: close the episode, and dispatch a heartbeat if one is due.

---

## 11. Experimental Setup

The experiments ask whether LOKA makes a model-based humanoid controller robust to changes in the robot and its environment that its model does not know about, and how that robustness compares with a learned locomotion policy, without any training. Section 11.1 describes the experiments run so far; Section 11.2 describes the planned extension to a public benchmark and to vision-language-action baselines.

### 11.1 Current evaluation

**Platform and common protocol.** All experiments use the Unitree G1 with 29 actuated joints [46] simulated in MuJoCo [47], with the controller of Section 4 in its default configuration and the scene of Section 4.2: a flat floor with a sliding friction coefficient of 1.0. Each condition is run for five trials. Trial $k$ uses the same random seed for every baseline, so that baselines are compared from identical initial states. Each episode starts from the standing keyframe; a uniform perturbation of at most 0.005 (in radians and radians per second) is then added to every joint position and joint velocity, leaving the floating-base pose untouched. Episodes run offscreen, without a real-time throttle, and the state is logged at 100 Hz. A *fall* is declared when the pelvis drops below 0.45 m, and ends the episode. Every perturbation is applied to the simulated plant only; the controller's belief model is never told (Section 4.2).

For LOKA, the orchestrator uses a commercial language model through its API with the text-only interface (the camera of Section 7.3 is disabled), the default session limits of Section 7 (six turns per failure episode, three turns of memory), and stand-on-fault enabled. The simulation keeps stepping while an orchestrator call is in flight, and the reply is applied at the control tick at which it arrives, so the robot experiences the call's real latency in simulated time at the speed at which the simulation runs; a call that does not return within 120 s is abandoned.

**Experiment 1: walking to a goal under a perturbation.** The robot starts standing at the origin and is asked, at $t = 0$, to walk to the world point (3, 0) m at 0.15 m/s and to stop within 0.2 m of it. LOKA receives this request as a natural-language operator directive ("Walk to the world coordinate (3, 0) at 0.15 m/s and stop within 0.2 m."), which it must translate into targets itself. Baselines without a language interface receive the same goal, speed and tolerance written directly into their goal-following input. At $t = 5$ s, while the robot is walking, one of the six perturbations in Table 6 is applied. An episode succeeds when the pelvis is within 0.2 m of the goal; it fails on a fall, or on a timeout at 60 s of simulated time.

**Table 6. Perturbations in Experiment 1.** All are applied at $t = 5$ s.

| Condition | Change to the plant | Unknown to the controller as |
|---|---|---|
| Nominal | none | – |
| Torso payload | +10 kg on the torso | body mass |
| Shoulder payload | +7 kg on the left shoulder | body mass and an off-centre CoM |
| Weak knee | right-knee actuator strength reduced to 75 % | actuator gear |
| Low friction | floor friction coefficient reduced from 1.0 to 0.4 | surface friction |
| Lateral shove | 120 N for 50 ms at the pelvis, to the robot's left (6 N·s) | an external impulse; no model change |

The shove is a control condition: it changes no model parameter, so a correct response from LOKA is to recover and continue without identifying anything.

**Experiment 2: push resistance while standing with an asymmetric payload.** The robot stands. At $t = 2$ s a 10 kg mass is added to its left shoulder. After a further 10 s, which gives LOKA time to detect and correct the model error, a 50 ms lateral push is applied at the pelvis, and the robot is observed for 10 s more. The measured quantity is the largest push force the robot survives. It is found with the same unbounded search as in Section 9: starting from a bracket of 0 to 200 N, the upper bound is doubled (up to 2000 N) until a push topples the robot, and the bracket is then bisected to a resolution of 0.2 N. Every probe is a fresh episode from the same seed, and the search assumes that a force that topples the robot topples it at every larger force. If the payload alone topples the robot before the push, the score is zero.

**Baselines.**

1. *Low-level controller.* The MPC and WBC of Section 4 with the same goal-following law, but without the anomaly detector, the orchestrator or the belief search. It isolates what LOKA adds: any difference between it and LOKA is due to detection, reasoning and belief correction.
2. *LOKA.* The same controller with the full system of Sections 5 to 9.
3. *Reinforcement-learning policy.* The open-source velocity-tracking policy for the G1 released with Unitree's reinforcement-learning framework [42], trained by its authors with domain randomisation, run on the same MuJoCo scene and subjected to the same plant perturbations. Because the policy accepts only a body-frame velocity command, the goal is converted into one by a proportional heading-and-speed law with the same arrival tolerance; commands below 0.30 m/s are raised to 0.30 m/s, because the released policy does not walk below about 0.25 m/s. This baseline represents the robustness that learned locomotion obtains from training, against which LOKA's training-free robustness is compared.

**Metrics.** For Experiment 1 we record, per condition and baseline:

- *Successes* and *falls*: the number of the five trials that reached the goal and that fell. The success rate is reported with a Wilson score interval [68], which remains meaningful at five trials and at rates of 0 or 1.
- *Time to goal*: simulated time from the request to arrival, over successful trials only; a failure is not scored as the timeout.
- *Wall-clock time to goal*: real time from the start of the episode to arrival, over successful trials only. For LOKA it includes the time spent waiting for the language model.
- *Path efficiency*: the straight-line distance from the starting pelvis position to the goal, divided by the length of the pelvis path actually walked, over successful trials only.
- *Auxiliary quantities*: the final distance to the goal, the time of the first fall, the number of orchestrator calls, and the wall-clock duration of the episode.

For Experiment 2 the metric is the largest survived push force. Continuous metrics are reported as the mean, the sample standard deviation and a 95 % Student-$t$ interval over trials.

### 11.2 Planned evaluation

**Benchmark navigation tasks.** The current experiments use one goal on a flat, empty floor. The next step is to evaluate on HumanoidBench [66], a MuJoCo benchmark of whole-body humanoid tasks, so that results are comparable with other methods. The first target is a goal-reaching navigation task in which the direct line to the goal is blocked, such as the benchmark's maze task. It exercises the parts of LOKA that the current experiments do not: the camera image of Section 7.3, waypoint planning in the self-assigned plan, and proximity listeners. As the scope widens, tasks with uneven ground, such as stairs and hurdles, would follow. These require footholds off the floor plane, which the current gait planner does not produce (Section 8.2), so they also depend on an extension of the low-level controller. HumanoidBench is built around the Unitree H1 by default, whereas the controller here is tuned for the G1. The evaluation will therefore either use a G1 version of the benchmark scenes, or port the task geometry and success criteria into the G1 scene.

**Vision-language-action baselines.** To compare LOKA with methods that obtain generality from large-scale training, the planned evaluation adds a humanoid vision-language-action (VLA) model that can be run in the same MuJoCo environment. Candidates include GR00T N1 [23], an open humanoid foundation model with a slow vision-language module and a fast learned action module, and LeVERB [69], which conditions a learned whole-body humanoid controller on vision and language. RT-2 [20] is a natural reference, but its weights are not public, so it can only be discussed rather than run. Most other VLAs, such as OpenVLA [21] and π0 [22], target arm manipulation and do not control humanoid balance. Each VLA will be given the same instruction and observations as LOKA, plus whatever robot-specific fine-tuning it requires, and the amount of data and training this involves will be reported as part of the comparison. The reinforcement-learning policy and the low-level controller of Section 11.1 will be retained.

**Planned metrics.**

- *Success rate*, with Wilson intervals, as defined by the benchmark task.
- *Time to completion*, in both simulated and wall-clock time.
- *Number of falls*.
- *Number of orchestrator invocations*, broken down by trigger (anomaly, listener, directive, heartbeat). This measures how much language-model reasoning a task needs.
- *Orchestrator cost*: per-call latency, and tokens and API cost per episode.
- *Training cost*: the robot-specific data and compute each method requires before deployment. This is zero for LOKA and the low-level controller, and is the quantity on which LOKA's claim rests.
- *Path efficiency* on navigation tasks, measured to the boundary of the goal region (see Section 12).

---

## 12. Notes for Drafting the Paper

These points are for the authors and should not appear in the paper.

1. **The current paper draft describes an earlier design.** The methodology section in `paper/sections/03_method.tex` defines an effort deviation integral as the squared difference between actual and nominal control inputs, a mission error-tracking section, a plateau/intervention-cap gate that accepts residuals, and semantic tags such as "saturated-seized". None of these exist in the implementation. The detector is the inverse-dynamics residual of Section 5; the plateau gate and the error-tracking section have been removed; belief values come from the search of Section 9. The MPC cost in the draft also omits the constraints and the condensed form.
2. **Bibliography corrections.** SayTap is by Tang et al., not Ji et al. Sleiman et al. appeared in IEEE Robotics and Automation Letters, not Transactions on Robotics. Galliker et al.'s Humanoids 2022 title is "Planar bipedal locomotion with nonlinear model predictive control." Several entries in `paper/sections/references.tex` are placeholders ("Anonymous et al.") and need real author lists before submission.
3. **Hardware path.** All detector inputs exist on hardware (Section 5.7), but no estimator is in the loop yet: the simulation uses ground-truth state. The paper should say so plainly.
4. **Numbers to re-check before submission.** Thresholds and bands above are the October 2026 defaults. If they change, Sections 4.8, 5.5 and 9.3 need updating.
5. **Comparability caveats in Experiment 1.** The reinforcement-learning baseline walks at no less than 0.30 m/s while the other two walk at 0.15 m/s, so its time to goal is not comparable with theirs; either rerun the model-based baselines at 0.30 m/s or report time to goal only between the two model-based baselines. Path efficiency divides by the distance to the goal centre, but arrival is declared at 0.2 m from it, so a policy that stops at the edge of the goal region can score above 1 (the reinforcement-learning baseline does). Measuring the straight-line distance to the goal boundary fixes this.
6. **Simulated latency depends on compute speed.** With the latency mode used so far, an orchestrator call costs as much simulated time as the simulation advances during the call's wall-clock duration, which depends on the machine and its load. For reproducibility, the paper should either report the simulation's real-time factor or rerun with the mode that charges a fixed simulated latency per call.
7. **Language model name.** The configuration names the orchestrator model; the paper should name the exact model and version, and the decoding settings.

---

## References

[1] S. Kajita et al., "Biped walking pattern generation by using preview control of zero-moment point," in *Proc. IEEE Int. Conf. Robot. Autom. (ICRA)*, 2003, pp. 1620–1626.

[2] J. Englsberger, C. Ott, M. A. Roa, A. Albu-Schäffer, and G. Hirzinger, "Bipedal walking control based on capture point dynamics," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, 2011, pp. 4420–4427.

[3] J. Englsberger, C. Ott, and A. Albu-Schäffer, "Three-dimensional bipedal walking control based on divergent component of motion," *IEEE Trans. Robot.*, vol. 31, no. 2, pp. 355–368, 2015.

[4] J. Di Carlo, P. M. Wensing, B. Katz, G. Bledt, and S. Kim, "Dynamic locomotion in the MIT Cheetah 3 through convex model-predictive control," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, 2018, pp. 1–9.

[5] J.-P. Sleiman, F. Farshidian, M. V. Minniti, and M. Hutter, "A unified MPC framework for whole-body dynamic locomotion and manipulation," *IEEE Robot. Autom. Lett.*, vol. 6, no. 3, pp. 4688–4695, 2021.

[6] A. Herzog, N. Rotella, S. Mason, F. Grimminger, S. Schaal, and L. Righetti, "Momentum control with hierarchical inverse dynamics on a torque-controlled humanoid," *Auton. Robots*, vol. 40, no. 3, pp. 473–491, 2016.

[7] P. M. Wensing, M. Posa, Y. Hu, A. Escande, N. Mansard, and A. Del Prete, "Optimization-based control for dynamic legged robots," *IEEE Trans. Robot.*, vol. 40, pp. 43–63, 2024.

[8] C. G. Atkeson, C. H. An, and J. M. Hollerbach, "Estimation of inertial parameters of manipulator loads and links," *Int. J. Robot. Res.*, vol. 5, no. 3, pp. 101–119, 1986.

[9] P. M. Wensing, S. Kim, and J.-J. E. Slotine, "Linear matrix inequalities for physically consistent inertial parameter identification: A statistical perspective on the mass distribution," *IEEE Robot. Autom. Lett.*, vol. 3, no. 1, pp. 60–67, 2018.

[10] G. Bledt, P. M. Wensing, S. Ingersoll, and S. Kim, "Contact model fusion for event-based locomotion in unstructured terrains," in *Proc. IEEE Int. Conf. Robot. Autom. (ICRA)*, 2018, pp. 4399–4406.

[11] F. Jenelten, J. Hwangbo, F. Tresoldi, C. D. Bellicoso, and M. Hutter, "Dynamic locomotion on slippery ground," *IEEE Robot. Autom. Lett.*, vol. 4, no. 4, pp. 4170–4176, 2019.

[12] J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel, "Domain randomization for transferring deep neural networks from simulation to the real world," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, 2017, pp. 23–30.

[13] X. B. Peng, M. Andrychowicz, W. Zaremba, and P. Abbeel, "Sim-to-real transfer of robotic control with dynamics randomization," in *Proc. IEEE Int. Conf. Robot. Autom. (ICRA)*, 2018, pp. 3803–3810.

[14] J. Hwangbo et al., "Learning agile and dynamic motor skills for legged robots," *Sci. Robot.*, vol. 4, no. 26, Art. no. eaau5872, 2019.

[15] J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, and M. Hutter, "Learning quadrupedal locomotion over challenging terrain," *Sci. Robot.*, vol. 5, no. 47, Art. no. eabc5986, 2020.

[16] A. Kumar, Z. Fu, D. Pathak, and J. Malik, "RMA: Rapid motor adaptation for legged robots," in *Proc. Robot.: Sci. Syst. (RSS)*, 2021.

[17] I. Radosavovic, T. Xiao, B. Zhang, T. Darrell, J. Malik, and K. Sreenath, "Real-world humanoid locomotion with reinforcement learning," *Sci. Robot.*, vol. 9, no. 89, Art. no. eadi9579, 2024.

[18] N. Rudin, D. Hoeller, P. Reist, and M. Hutter, "Learning to walk in minutes using massively parallel deep reinforcement learning," in *Proc. Conf. Robot Learn. (CoRL)*, 2022, pp. 91–100.

[19] W. Zhao, J. P. Queralta, and T. Westerlund, "Sim-to-real transfer in deep reinforcement learning for robotics: A survey," in *Proc. IEEE Symp. Ser. Comput. Intell. (SSCI)*, 2020, pp. 737–744.

[20] A. Brohan et al., "RT-2: Vision-language-action models transfer web knowledge to robotic control," in *Proc. Conf. Robot Learn. (CoRL)*, 2023.

[21] M. J. Kim et al., "OpenVLA: An open-source vision-language-action model," in *Proc. Conf. Robot Learn. (CoRL)*, 2024.

[22] K. Black et al., "π0: A vision-language-action flow model for general robot control," arXiv:2410.24164, 2024.

[23] NVIDIA et al., "GR00T N1: An open foundation model for generalist humanoid robots," arXiv:2503.14734, 2025.

[24] M. Ahn et al., "Do as I can, not as I say: Grounding language in robotic affordances," in *Proc. Conf. Robot Learn. (CoRL)*, 2022.

[25] J. Liang et al., "Code as policies: Language model programs for embodied control," in *Proc. IEEE Int. Conf. Robot. Autom. (ICRA)*, 2023, pp. 9493–9500.

[26] W. Huang et al., "Inner monologue: Embodied reasoning through planning with language models," in *Proc. Conf. Robot Learn. (CoRL)*, 2022.

[27] D. Shah, B. Osiński, B. Ichter, and S. Levine, "LM-Nav: Robotic navigation with large pre-trained models of language, vision, and action," in *Proc. Conf. Robot Learn. (CoRL)*, 2022.

[28] H. Yuan et al., "Being-0: A humanoid robotic agent with vision-language models and modular skills," arXiv:2503.12533, 2025.

[29] Y. Tang, W. Yu, J. Tan, H. Zen, A. Faust, and T. Harada, "SayTap: Language to quadrupedal locomotion," in *Proc. Conf. Robot Learn. (CoRL)*, 2023.

[30] D. Martinez-Baselga, O. de Groot, L. Knoedler, J. Alonso-Mora, L. Riazuelo, and L. Montano, "Hey robot! Personalizing robot navigation through model predictive control with a large language model," in *Proc. IEEE Int. Conf. Robot. Autom. (ICRA)*, 2025.

[31] W. Yu et al., "Language to rewards for robotic skill synthesis," in *Proc. Conf. Robot Learn. (CoRL)*, 2023.

[32] R. Bommasani et al., "On the opportunities and risks of foundation models," arXiv:2108.07258, 2021.

[33] OpenAI, "GPT-4 technical report," arXiv:2303.08774, 2023.

[34] J. Wei et al., "Chain-of-thought prompting elicits reasoning in large language models," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2022.

[35] S. Yao et al., "ReAct: Synergizing reasoning and acting in language models," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2023.

[36] M. Nye et al., "Show your work: Scratchpads for intermediate computation with language models," arXiv:2112.00114, 2021.

[37] R. Sinha, A. Elhafsi, C. Agia, M. Foutter, E. Schmerling, and M. Pavone, "Real-time anomaly detection and reactive planning with large language models," in *Proc. Robot.: Sci. Syst. (RSS)*, 2024.

[38] Z. Liu, A. Bahety, and S. Song, "REFLECT: Summarizing robot experiences for failure explanation and correction," in *Proc. Conf. Robot Learn. (CoRL)*, 2023.

[39] Y. J. Ma et al., "Eureka: Human-level reward design via coding large language models," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2024.

[40] D. Kahneman, *Thinking, Fast and Slow*. New York, NY, USA: Farrar, Straus and Giroux, 2011.

[41] A. d'Avila Garcez and L. C. Lamb, "Neurosymbolic AI: The 3rd wave," *Artif. Intell. Rev.*, vol. 56, no. 11, pp. 12387–12406, 2023.

[42] Unitree Robotics, "unitree_rl_lab: Reinforcement learning environments for Unitree robots," GitHub repository. [Online]. Available: https://github.com/unitreerobotics/unitree_rl_lab (accessed Oct. 2026).

[43] D. Kim, J. Di Carlo, B. Katz, G. Bledt, and S. Kim, "Highly dynamic quadruped locomotion via whole-body impulse control and model predictive control," arXiv:1909.06586, 2019.

[44] M. Y. Galliker, N. Csomay-Shanklin, R. Grandia, A. J. Taylor, F. Farshidian, M. Hutter, and A. D. Ames, "Planar bipedal locomotion with nonlinear model predictive control: Online gait generation using whole-body dynamics," in *Proc. IEEE-RAS Int. Conf. Humanoid Robots (Humanoids)*, 2022.

[45] J. Pratt, J. Carff, S. Drakunov, and A. Goswami, "Capture point: A step toward humanoid push recovery," in *Proc. IEEE-RAS Int. Conf. Humanoid Robots (Humanoids)*, 2006, pp. 200–207.

[46] Unitree Robotics, "Unitree G1 humanoid agent," 2024. [Online]. Available: https://www.unitree.com/g1

[47] E. Todorov, T. Erez, and Y. Tassa, "MuJoCo: A physics engine for model-based control," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, 2012, pp. 5026–5033.

[48] B. Stellato, G. Banjac, P. Goulart, A. Bemporad, and S. Boyd, "OSQP: An operator splitting solver for quadratic programs," *Math. Program. Comput.*, vol. 12, no. 4, pp. 637–672, 2020.

[49] "wb_humanoid_mpc: Whole-body model predictive control for humanoid robots," GitHub repository. [Online]. Available: https://github.com/manumerous/wb_humanoid_mpc (accessed Oct. 2026).

[50] X. Xiong and A. Ames, "3-D underactuated bipedal walking via H-LIP based gait synthesis and stepping stabilization," *IEEE Trans. Robot.*, vol. 38, no. 4, pp. 2405–2425, 2022.

[51] Y. Gong and J. W. Grizzle, "Zero dynamics, pendulum models, and angular momentum in feedback control of bipedal locomotion," *J. Dyn. Syst., Meas., Control*, vol. 144, no. 12, 2022.

[52] J. Li and Q. Nguyen, "Force-and-moment-based model predictive control for achieving highly dynamic locomotion on bipedal robots," in *Proc. IEEE Conf. Decis. Control (CDC)*, 2021.

[53] M. Diehl, H. G. Bock, and J. P. Schlöder, "A real-time iteration scheme for nonlinear optimization in optimal feedback control," *SIAM J. Control Optim.*, vol. 43, no. 5, pp. 1714–1736, 2005.

[54] P. M. Wensing and D. E. Orin, "Generation of dynamic humanoid behaviors through task-space control with conic optimization," in *Proc. IEEE Int. Conf. Robot. Autom. (ICRA)*, 2013, pp. 3103–3109.

[55] A. Escande, N. Mansard, and P.-B. Wieber, "Hierarchical quadratic programming: Fast online humanoid-robot motion generation," *Int. J. Robot. Res.*, vol. 33, no. 7, pp. 1006–1028, 2014.

[56] M. Khadiv, A. Herzog, S. A. A. Moosavian, and L. Righetti, "Walking control based on step timing adaptation," *IEEE Trans. Robot.*, vol. 36, no. 3, pp. 629–643, 2020.

[57] Y. Ding, C. Khazoom, M. Chignoli, and S. Kim, "Orientation-aware model predictive control with footstep adaptation for dynamic humanoid walking," in *Proc. IEEE-RAS Int. Conf. Humanoid Robots (Humanoids)*, 2022.

[58] A. De Luca and R. Mattone, "Sensorless robot collision detection and hybrid force/motion control," in *Proc. IEEE Int. Conf. Robot. Autom. (ICRA)*, 2005, pp. 999–1004.

[59] S. Haddadin, A. De Luca, and A. Albu-Schäffer, "Robot collisions: A survey on detection, isolation, and identification," *IEEE Trans. Robot.*, vol. 33, no. 6, pp. 1292–1312, 2017.

[60] F. Flacco, A. Paolillo, and A. Kheddar, "Residual-based contacts estimation for humanoid robots," in *Proc. IEEE-RAS Int. Conf. Humanoid Robots (Humanoids)*, 2016, pp. 409–415.

[61] R. Featherstone, *Rigid Body Dynamics Algorithms*. New York, NY, USA: Springer, 2008.

[62] M. Bloesch et al., "State estimation for legged robots: Consistent fusion of leg kinematics and IMU," in *Proc. Robot.: Sci. Syst. (RSS)*, 2012.

[63] R. Hartley, M. Ghaffari, R. M. Eustice, and J. W. Grizzle, "Contact-aided invariant extended Kalman filtering for robot state estimation," *Int. J. Robot. Res.*, vol. 39, no. 4, pp. 402–430, 2020.

[64] M. Camurri et al., "Probabilistic contact estimation and impact detection for state estimation of quadruped robots," *IEEE Robot. Autom. Lett.*, vol. 2, no. 2, pp. 1023–1030, 2017.

[65] M. Focchi, V. Barasuol, M. Frigerio, D. G. Caldwell, and C. Semini, "Slip detection and recovery for quadruped robots," in *Robotics Research* (Springer Proc. Adv. Robot.). Cham, Switzerland: Springer, 2018.

[66] C. Sferrazza, D.-M. Huang, X. Lin, Y. Lee, and P. Abbeel, "HumanoidBench: Simulated humanoid benchmark for whole-body locomotion and manipulation," in *Proc. Robot.: Sci. Syst. (RSS)*, 2024.

[67] J. L. Bentley and A. C.-C. Yao, "An almost optimal algorithm for unbounded searching," *Inf. Process. Lett.*, vol. 5, no. 3, pp. 82–87, 1976.

[68] E. B. Wilson, "Probable inference, the law of succession, and statistical inference," *J. Amer. Statist. Assoc.*, vol. 22, no. 158, pp. 209–212, 1927.

[69] H. Xue et al., "LeVERB: Humanoid whole-body control with latent vision-language instruction," arXiv:2506.13751, 2025.
