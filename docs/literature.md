# Literature behind this stack

What the repository actually runs, and which part of each paper is in the code.
Papers that explain the open turning and push problem are listed last. They are
not implemented.

The fast loop is `loka/control/`: a gait plan, a convex force MPC at 50 Hz, and
a whole-body torque QP at 500 Hz. The slow loop is `loka/agent/`: compressed
telemetry, an LLM, and a typed YAML scratchpad. The language model never sets
a foothold and never waits inside the control thread.

## Convex force MPC

| Paper | In the code |
|---|---|
| J. Di Carlo, P. M. Wensing, B. Katz, G. Bledt, S. Kim, *Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control*, IROS 2018 | This is `ConvexMPC` in `loka/control/mpc.py`. State \(x = [\theta, p, \omega, v, g] \in \mathbb{R}^{13}\), one condensed convex QP, friction pyramids, horizon \(6 \times 50\,\mathrm{ms}\). Decision variables are contact forces. Foot positions are inputs. |
| J.-P. Sleiman, F. Farshidian, M. V. Minniti, M. Hutter, *A Unified MPC Framework for Whole-Body Dynamic Locomotion and Manipulation*, IEEE TRO 2021, [arXiv:2103.00946](https://arxiv.org/abs/2103.00946) | The CoP-in-sole inequalities, so each foot is a surface rather than four independent points. The 12 mm ZMP inset lives on the gait reference, not in this box. We did not take their nonlinear centroidal program or the full joint kinematics inside the horizon. Contact pose along the horizon is the gait Bézier. |
| M. Y. Galliker et al., *Bipedal Locomotion with Nonlinear Model Predictive Control*, Humanoids 2022, [arXiv:2203.07429](https://arxiv.org/abs/2203.07429) | The lesson only: a 0.2–0.3 s force horizon works when a gait reference already carries the step. The solver here is still Di Carlo's QP. Not taken: ocs2, hybrid zero dynamics, or torque nonlinear MPC. That would replace the whole-body QP. |
| B. Stellato, G. Banjac, P. Goulart, A. Bemporad, S. Boyd, *OSQP: an operator splitting solver for quadratic programs*, Mathematical Programming Computation, 2020 | `loka/control/qp.py`. The sparsity pattern is fixed at setup so both QPs refactor once and only update numbers. |

## Whole-body QP

| Paper | In the code |
|---|---|
| D. Kim, J. Di Carlo, B. Katz, G. Bledt, S. Kim, *Highly Dynamic Quadruped Locomotion via Whole-Body Impulse Control and Model Predictive Control*, arXiv:1909.06586, 2019 | The split itself. The MPC publishes a force plan; the faster QP (`loka/control/wbc.py`) realises it with torques. Tasks are contact, base position, base orientation, posture, and swing. |
| A. Herzog, N. Rotella, S. Mason, F. Grimminger, S. Schaal, L. Righetti, *Momentum control with hierarchical inverse dynamics on a torque-controlled humanoid*, Autonomous Robots, 2016 | Contact, posture, and swing as costs on a torque-controlled humanoid at the WBC rate. Ours is one weighted QP, not their strict hierarchy. |
| `loka/control/wbc.py` (local choice, not a paper) | Contact is a cost, not a hard constraint. A hard "hold this point" row goes infeasible when a heel lands with leftover velocity. \(f = 0\) and a free-falling \(\ddot q\) remain feasible, so a bad contact change degrades instead of returning a garbage iterate. |

## ALIP stepping

| Paper | In the code |
|---|---|
| X. Xiong, A. D. Ames, *3-D Underactuated Bipedal Walking via H-LIP Based Gait Synthesis and Stepping Stabilization*, IEEE T-RO 2022, [arXiv:2101.09588](https://arxiv.org/abs/2101.09588) | `loka/control/alip.py` and `AlipStepFootholdPolicy` on `alip_footstep`. The state is \((x, L)\) about the stance contact, not CoM velocity. The G1 plant keeps the DCM chain and the convex MPC; the stepping law corrects the next foothold from the pre-impact error. |
| Y. Gong, J. Grizzle, *Zero Dynamics, Pendulum Models, and Angular Momentum in Bipedal Walking*, JDSMC 2022, [arXiv:2105.08170](https://arxiv.org/abs/2105.08170) | Why \(L\) about the contact is the right second state when the swing leg carries angular momentum. Not a separate controller. |

`legacy_dcm` does not call `alip.py`. Both stacks use the same convex MPC and whole-body QP.

## Gait

| Paper | In the code |
|---|---|
| J. Englsberger, C. Ott, M. A. Roa, A. Albu-Schäffer, G. Hirzinger, *Bipedal walking control based on capture point dynamics*, IROS 2011, and *Three-dimensional bipedal walking control based on divergent component of motion*, IROS 2013 | `loka/control/gait.py`. Backward DCM recursion from a terminal rest, ZMP moving across double support, CoM recovered from the DCM by \(\dot c = \omega(\xi - c)\). |
| J. Pratt, J. Carff, S. Drakunov, A. Goswami, *Capture Point: A Step toward Humanoid Push Recovery*, Humanoids 2006 | The next foothold takes a fraction of the full DCM-offset placement (`gait.capture_gain`, default 0.7, 1.0 is deadbeat) and is clamped. The error is the offset past the target foothold, not the end-of-step DCM itself: that DCM moves with the correction, so the old law settled at `k/(1+k)` of the error. Feedback may rescue a step and may not redesign the gait. |
| S. Kajita, F. Kanehiro, K. Kaneko, K. Fujiwara, K. Harada, K. Yokoi, H. Hirukawa, *Biped walking pattern generation by using preview control of zero-moment point*, ICRA 2003 | The LIP idea behind the 12 mm ZMP inset (`ZMP_INSET`): the reference ZMP sits inside the sole so the centre of pressure has spare travel. We do not run Kajita's preview controller. The CoM reference is the DCM, not the cart-table preview. |
| S. Kajita et al., *A realtime pattern generator for biped walking*, ICRA 2003 | Not the turn we run. They rotate the planning frame at a foot place and add a separate foot-orientation pattern. HRP-2L changed direction by up to about 0.34 rad per step. Our chain yaws by at most 0.08 rad per step and only while it is also stepping forward. See the turning section. |
| A. Herdt, H. Diedam, P.-B. Wieber, D. Dimitrov, K. Mombaur, M. Diehl, *Online walking motion generation with automatic footstep placement*, International Journal of Robotics Research, 2010 | Not used. Their linear MPC decides footstep positions. Here the chain is geometric (`speed × T_step` along the step yaw, crossed by `stance_width`) and the only online footstep change is the clamped DCM nudge. |
| Swing arc in `gait.py` | Local, not a cited generator. Horizontal motion is a smoothstep so the foot leaves and lands with zero horizontal speed. Vertical motion is a half sine. The WBC tracks that arc with acceleration feedforward; the gains only clean up the residual. Galliker is the reason the arc exists at all: a short force horizon cannot invent the swing. |

`gait_schedule_for_speed` is measured on this plant, not taken from a paper. Duty stays near 0.80 at a crawl because 0.65 pitches the robot over in about 3 s. The LLM may pin period, duty, and width; otherwise the schedule fills them from speed.

## Turning, as it exists today

The heading path is a local design on top of the DCM chain, not a paper method.
`gait.heading` is a goal. Each step yaws by at most `min(turn_rate × T_step, 0.12 rad)`.
Forward speed falls to 0.10 m/s while the heading error is large, and
`TURN_SETTLE` keeps that crawl until the foot yaw has been within 0.10 rad for
two steps and the measured speed is back near 0.10 m/s. That dwell is the
idea in `ProceduralMpcMotionManager` from
[wb_humanoid_mpc](https://github.com/manumerous/wb_humanoid_mpc) (BSD-3-Clause;
the gait-switch code itself is not copied — it latches a `static` config and
reads the command in the slow-down check). Measured speed never enters the
stride. Pelvis `yaw_ref` smoothsteps across the whole step, including single
support. The MPC tracks that yaw and yaw rate. The WBC tracks the same yaw and
sets hip yaw so a planted foot is not twisted as the pelvis turns. A +0.3 rad
heading at 0.20 m/s holds, and so do a 90° turn at 0.20 m/s and turn-in-place.
Notes and the bench are in `docs/walking.md`.

## Simulation and the robot

| Source | In the code |
|---|---|
| E. Todorov, T. Erez, Y. Tassa, *MuJoCo: A physics engine for model-based control*, IROS 2012 | The plant in `loka/sim.py`. The controller calls MuJoCo only for kinematics on its private belief model (`loka/control/robot.py`): Jacobians, bias forces, subtree centre of mass. `compute_torque(qpos, qvel)` does not step the simulator. |
| Unitree G1, 29-DoF MJCF in `models/g1/` | The embodiment. Sole half-size used by the CoP box is read off the contact sites: about 85 mm fore-aft and 30 mm laterally. Hip yaw range is ±2.76 rad; the turn limit in the gait is not that joint. |
| T. Howell, N. Gileadi, S. Tunyasuvunakool, K. Zakka, T. Erez, Y. Tassa, *Predictive Sampling: Real-time Behaviour Synthesis with MuJoCo*, arXiv:2212.00541, 2022 | The earlier walker on the `poc` branch, and the comparison point for turning. Predictive sampling can try a foot lift and a foot yaw because those are actions. This convex QP cannot: foot yaw is an input. We do not run MJPC on the G1 plant. |

## Orchestrator

The slow loop is specified in `docs/orchestrator.md` and `paper/sections/02_related_work.tex`.
These are the works the implementation actually follows. The paper's bibliography
lists others we looked at and did not build.

| Paper | In the code |
|---|---|
| W. Yu et al., *Language to Rewards for Robotic Skill Synthesis*, CoRL 2023 | The contrast. They emit reward code for MuJoCo MPC. We emit a typed YAML scratchpad (`Controller_Targets`, `Task_Targets`, `Error_Tracking`, `Model_Mutations`, `Listeners`) and reject code in the balance loop. |
| Z. Liu, A. Bahety, S. Song, *REFLECT: Summarizing Robot Experiences for Failure Explanation and Correction*, CoRL 2023 | The compressor idea: turn a stream into a short text the model can diagnose. Ours is online proprioception (tracked state, balance, motor load, full pose), not a post-mortem manipulation log. |
| Hey Robot! (LLM-personalized navigation MPC), ICRA 2025, as cited in `paper/sections/02_related_work.tex` | The authority to rewrite cost weights from language. Ours is narrower: an allowlist of friction and impedance paths. Locked QP costs stay put. |
| *Real-Time Anomaly Detection and Reactive Planning with LLMs* (AESOP), RSS 2024, as cited in the paper | The dual rate. Anything that has to happen in under 300 ms is classical. The LLM runs on a background thread. |
| G. Ji et al., *SayTap: Language to Quadrupedal Locomotion*, CoRL 2023 | The contrast on contacts. SayTap lets language set a foot-contact pattern. We expose `gait.*` and keep footholds and contact flags in `GaitScheduler`. |

Not built, and named so they are not mistaken for the stack: Eureka and Text2Reward (offline reward code), VLA policies, domain-randomization skill discovery, and an RL residual on joint torque. A per-step residual on the gait layer (foothold offset, swing-clock scale, optional yaw offset) is scoped in `docs/walking.md` and not trained.

## Turning and push recovery that this architecture can still absorb

These fit the current split. None of them is what the gait does for a large heading today.

| Paper | What it would change |
|---|---|
| Kajita et al., *A realtime pattern generator for biped walking*, ICRA 2003 | Turn-in-place as a footstep list. Forward speed stays zero. Each swing foot replants rotated. On HRP-2L the direction change was up to about 0.34 rad per step, with the body facing the step at mid-stance. Foot yaw is planned. The preview controller does not invent it. This stays inside `GaitScheduler`. The MPC and WBC already consume per-foot yaw and `yaw_ref`. |
| Englsberger DCM and Pratt capture point (already cited) | Push recovery by replanting. The correction is the offset law above. Turn-in-place holds with it. A larger disturbance still needs step timing, which was tried and ruled out. |
| R. J. Griffin, G. Wiedebach, S. Bertrand, A. Leonessa, J. Pratt, *Walking stabilization using step timing and location adjustment on the humanoid robot, Atlas*, IROS 2017, [arXiv:1703.00477](https://arxiv.org/abs/1703.00477) | The split the timing QP was built from: shortening the step handles errors along travel, moving the foot handles errors across it. The QP is in `gait.py` and switched off; see the Khadiv row. |
| Y. Ding, C. Khazoom, M. Chignoli, S. Kim, *Orientation-Aware Model Predictive Control with Footstep Adaptation for Dynamic Humanoid Walking*, arXiv:2205.15443, 2022 | The convex upgrade if the external foothold nudge is not enough. An augmented single rigid body keeps the QP, and the footstep location becomes a decision variable beside the forces, so a yaw disturbance can be answered by a step. A task-space QP still tracks the swing. Foot orientation stays a reference. This would replace "footholds are not decision variables" for the force layer only. It does not replace the WBC, and it still cannot discover foot yaw the way MuJoCo MPC can. |
| Z. Xie, Y. Wang, X. Luo, P. Arpenti, F. Ruggiero, B. Siciliano, *Three-dimensional variable center of mass height biped walking using a new model and nonlinear model predictive control* | The zero frictional moment point: yaw moment that would spin a planted foot is infeasible. Their nonlinear MPC refuses a 90° turn in one step and spreads it over several steps. We do not have a ZFMP constraint. The same fact is why pelvis yaw during single support, against one planted sole, rolls the G1. Their solver is a nonlinear program; we would take the constraint, not the solver. |
| M. Khadiv, A. Herzog, S. A. A. Moosavian, L. Righetti, *Walking control based on step timing adaptation*, IEEE TRO 2020 | Step duration beside the foothold, one 5-variable QP (`_step_adjust`, flag `STEP_TIMING`). With the duration fixed it is the offset foothold law. Left off: at every weight tried, a straight 0.20 m/s walk went backwards and pushes that used to be survived became falls. |
| Y. Chen and Q. Nguyen, *Adapting Gait Frequency for Posture-Regulating Humanoid Push-Recovery via Hierarchical Model Predictive Control*, ICRA 2025 | Cited in the paper as hierarchical push recovery. Not in this plant. Same moral as Khadiv: timing is a recovery input, not only foot position. |
