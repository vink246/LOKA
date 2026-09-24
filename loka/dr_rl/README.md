# Domain-Randomized PPO Walker Baseline

Frozen **proximal policy optimization** (PPO) on the **same native MuJoCo plant** as LOKA (`models/walker/task.xml`), trained with **in-distribution domain randomization** (DR). The policy is a 10 ms torque MLP. It does not use MJPC, residual RL, or the LLM orchestrator.

The point of this baseline is the comparison in the LOKA paper: a DR policy that is robust to expected mass/friction/push variation, and **fails on unmodeled structural faults** (dead actuator `gear=0`, ice \(\mu=0.2\), 25% backpack) that LOKA mutates online.

## Architecture

```
training (many rollouts)                          suite eval (same harness as LOKA)
────────────────────────                          ────────────────────────────────
Gymnasium env  ──reset──► sample DR               PlantFaults injects suite test
       │                   mass / mu / gear /     at t = perturbation_time_s
       │                   damping / shove
       ▼                                          qpos, qvel, last action, gait phase
MuJoCo mj_step (0.0025 s)                                │
ctrl[6] ∈ [-1, 1]                                        ▼
       ▲                                          frozen PPO (deterministic)
       │                                                 │
PPO clip + GAE  ←  VecNormalize obs                      ▼
                                                       data.ctrl → mj_step
```

Physics is `mujoco.mj_step` on MuJoCo 3.2, not MJX/Brax/Isaac. Gymnasium is only the `reset`/`step` API Stable-Baselines3 needs. Evaluation reuses `loka.walker_suite.faults.PlantFaults` and `run_walker_suite`.

Action: the six Walker motors, same order as MJPC (`right_hip` … `left_ankle`). Policy dt is `frame_skip=4` × `0.0025 s` = **10 ms**, matching MJPC `agent_timestep`. Observation is proprioception plus an open-loop walk clock: `qpos` without `rootx`, full `qvel`, last action, `sin/cos` of gait phase (**25 dims**). The actor never sees privileged plant parameters (friction, mass, gear).

This is **pure DR PPO**, not residual RL on MPC. Residual-on-MPC is LOKA’s fast loop, not this baseline.

## Why this recipe (recent locomotion practice)

The *algorithm* is the 2022–2025 locomotion default: **clipped PPO + GAE + observation normalization + dynamics randomization**. The *systems* that currently set the bar run that recipe at huge batch sizes:

| Work | Year | What we take | What we do not copy |
|---|---|---|---|
| [Tobin et al., Domain Randomization](https://arxiv.org/abs/1703.06907) | IROS 2017 | Randomize sim parameters instead of identifying them | Vision randomization (this walker is proprioceptive) |
| [Peng et al., Sim-to-Real](https://arxiv.org/abs/1804.10332) | RSS 2018 | Dynamics DR (mass, friction, damping) for transfer | Hardware deployment |
| [Rudin et al., Learning to Walk in Minutes](https://arxiv.org/abs/2109.11978) | CoRL 2022 | PPO + DR + obs RMS; the RSL-RL stack still used in Isaac Lab | GPU vectorization (Isaac / PhysX) |
| [Siekmann / Li et al., Cassie](https://hybrid-robotics.berkeley.edu/publications/ICRA2021_RL-Cassie-Walking.pdf) | ICRA 2021 | Biped DR ranges (mass, friction, damping, pushes) | Command-conditioned residual on a gait library |
| [Margolis & Agrawal, Walk These Ways](https://arxiv.org/abs/2212.03238) | CoRL 2022 | Mid-episode pushes + motor strength randomization | Multiplicity-of-behavior gait commands |
| [Ma et al., DrEureka](https://arxiv.org/abs/2406.01967) | RSS 2024 | LLM-chosen DR as a *related* baseline, not LOKA | Using an LLM to design the training distro |
| [Zakka et al., MuJoCo Playground](https://arxiv.org/abs/2502.08844) | 2025 | Same DR fields on MuJoCo (`geom_friction`, mass, actuator gain) | MJX/JAX batched sim (different engine than MJPC) |
| [Raffin et al., Stable-Baselines3](https://jmlr.org/papers/v22/20-1364.html) | JMLR 2021 | Clipped PPO implementation used here | — |

SOTA *throughput* in 2025 is RSL-RL on Isaac Lab or Playground/MJX (thousands of parallel envs). That would **not** share MJPC’s CPU MuJoCo 3.2 plant. This folder stays on that plant so LOKA vs DR vs fixed MPC is an apples-to-apples controller comparison, not a simulator comparison.

We also **do not** train with failure-aware tricks such as random joint masking ([Lee et al., 2024](https://arxiv.org/abs/2403.00398)) or locked-joint DR. Those enumerate the test faults offline. The named `dr_rl` baseline is standard in-distribution DR; a failure-curriculum policy would be a separate ablation (`DR-oracle`).

## PPO, specifically

Stable-Baselines3 **`PPO` + `MlpPolicy`**, i.e. Schulman et al. 2017 **clipped surrogate** PPO (not TRPO, not RecurrentPPO/LSTM, not RMA teacher–student).

Hyperparameters follow [RL Baselines3 Zoo `Walker2d-v4`](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml), which is the closest public PPO tuning for a 6-DoF planar walker:

| Setting | Value | Role |
|---|---|---|
| Policy | diagonal Gaussian MLP, `pi=vf=[256, 256]`, ReLU | Independent actor and critic |
| `log_std_init` | −0.5 | Initial Gaussian std ≈ 0.61 (stops the policy from collapsing to a stand) |
| Horizon `n_steps` | 512 | Rollout length per env before an update |
| `batch_size` / `n_epochs` | 64 / 20 | Minibatch SGD on the clipped objective |
| `clip_range` | 0.1 | PPO-clip ε (tighter than the textbook 0.2) |
| `learning_rate` | 5.05×10⁻⁵ | Zoo-tuned Adam step size |
| `gamma` / `gae_lambda` | 0.99 / 0.95 | Discount + GAE-λ advantage |
| `ent_coef` / `vf_coef` | 5×10⁻³ / 0.87 | Entropy bonus (higher than Zoo so std does not die) and value loss weight |
| `max_grad_norm` | 1.0 | Gradient clip |
| `VecNormalize` | obs + reward, clip obs ±10 | Running mean/std, standard in RSL-RL / Playground |
| Parallel envs | 4× `DummyVecEnv` by default | Use `--vec subproc --n-envs 8` if you want more throughput |
| Device | CPU is enough for this MLP | `--device cuda` if available |
| Nominal warmup | 250k steps | DR off until a gait exists, then in-distribution DR |

Reward (training only) starts from **Gymnasium Walker2d-v5** ([source](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/walker2d_v5.py)), which is what SB3 Zoo / CleanRL train (`r = v_x + 1_{\text{alive}} - 10^{-3}\|a\|^2`, ~2000–4500 at 1e6 steps). DMC `walker walk` is a different scale (reward in [0, 1], speed ≥ 1 m/s, no cap). Our extras sit on the Gym base:

\[
r = v_x - 3\max(v_x-1,0)^2 + 1_{\text{alive}} + r_{\text{stride}} + r_{\text{clearance}} + r_{\text{gait}} - c_{\text{pose}} - c_{\text{sym}} - c_{\text{lead}} - c_{\text{rom}} - c_{\text{flight}} - c_{\text{slip}} - c_{\text{height}}^{\text{deadband}} - c_{\text{pitch}}^{\text{deadband}} - 10^{-3}\|a\|^2 - \cdots
\]

- Height/pitch free inside ±0.10 m / ±0.25 rad.
- Isaac `feet_slide` 0.25; Isaac-style air-time on a supported landing; Unitree G1 `foot_clearance` 0.5.
- Unitree G1 `feet_gait` 0.5: +1 per foot matching the 0.8 s contact clock (offset 0.5, stance if phase < 0.55), times `clip(v_x/1, 0, 1)`. Phase `sin/cos` is in the observation. Standing at t=0 is double-support (a match) but collects nothing until it moves.
- Sine-walk pose **2.0** (not speed-gated). Hip is `0.35 + 0.40\cos(2\pi\phi)` (flexed at heel-strike); knee/ankle stay near stance on the ground and flex mid-swing. At 0.20 gated by `vx`, the 500k compass ignored the sine (~0.15/step to track vs ~2.5 from `vx`+alive+gait).
- L/R tracking symmetry 0.5: `|err_R − err_L|`. Cycle-mean lead 1.0: `(mean q_{\mathrm{hip},L} - mean q_{\mathrm{hip},R})^2` over the last 0.8 s. Hip ROM 1.0 after one period: each hip must travel `2 \times hip_amp`. These are **not** same-time `|q_L − q_R|` (that would be a hop/stand). Residual-on-MPC is out of scope for this POC.
- Flight 0.25 if both feet off.
- Training **ends only on a real fall** (height < 0.45 m or |pitch| > 1.2 rad, same as the suite). Unhealthy height/pitch drop the alive bonus but do not cut the episode — terminating at 0.85 m was a 0.75 s lunge every time.

A 1 m/s scissor that tracks the sine is ~**2000–2800**. The 500k compass scores **negative** under these weights. Retrain from scratch (obs is still 25-D).

## Randomizations implemented

Configured in `config.yaml` under `domain_randomization`. Sampled **per episode at reset**, except pushes which fire at a random time so the policy sees a sudden change (the suite injects at t = 2 s).

| Channel | Training range | Suite test (OOD) |
|---|---|---|
| Floor sliding friction | \(\mu \sim U(0.45, 1.2)\) | `ice`: \(\mu = 0.2\) |
| Body mass (each link except world) | × \(U(0.85, 1.15)\) | `backpack`: torso +25% |
| Actuator gear (each of 6 motors) | × \(U(0.70, 1.10)\), **never 0** | `dead_right_hip`: gear = 0 |
| Joint damping | × \(U(0.75, 1.25)\) | — |
| Torso shove | 50% of episodes, `force_frac ~ U(0, 0.35)`, 0.1–0.3 s, onset 0.5–4 s | `shove_back`: 0.5 bodyweight |
| Reset qpos/qvel noise | \(U(\pm 0.005)\) | — |
| Observation noise (train only) | qpos σ=0.01, qvel σ=0.05 | eval is deterministic, no noise |

Not randomized (and not in the training env): obstacles (`box`), language commands, `gear=0`, \(\mu \le 0.2\).

If ice or backpack look too easy after training, the ranges leaked — tighten them, do not retune the eval tests.

## Layout

| File | Role |
|---|---|
| `config.yaml` | Gait harness: DR, sine/clock reward, PPO, `results/dr_rl/` |
| `config_gym.yaml` | Gym-style harness: height/pitch/speed only, `results/dr_rl_gym/` |
| `env.py` | `LokaWalkerDREnv` (Gymnasium, native MuJoCo) |
| `randomize.py` | Reset-time dynamics DR + training pushes |
| `observation.py` | Proprioceptive obs (23-D, or 25-D with gait phase) |
| `train_ppo.py` | Gait-harness SB3 entry point |
| `train_gym_ppo.py` | Gym-style entry point (does not overwrite `results/dr_rl/`) |
| `policy.py` | Load frozen `.zip` + obs RMS |
| `runtime.py` | Suite adapter (`DrRlRuntime`, one `mj_step` per call) |

Checkpoints default to `results/dr_rl/ppo_walker.zip` and `results/dr_rl/vecnormalize.pkl`. Eval prefers `results/dr_rl/best/` if present.

## Train

```bash
# from the LOKA repo root, env with mujoco 3.2 + gymnasium + stable-baselines3
pip install gymnasium stable-baselines3
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA wheel

python -m loka.dr_rl.train_ppo
python -m loka.dr_rl.train_ppo --seed 7
python -m loka.dr_rl.train_ppo --vec subproc --n-envs 8 --timesteps 2000000
python -m loka.dr_rl.train_ppo --no-dr   # PPO without DR (ablation)
```

A useful smoke run is `--timesteps 2048 --n-envs 1`. A policy that actually walks needs on the order of **1–3e6** steps. This overwrites `results/dr_rl/ppo_walker.zip`.

## Gym-style harness (no gait clock)

Experimental ablation: same plant, DR, PPO, height/pitch/overspeed, **no** sine / contact clock / lead / ROM / slip / stride / clearance. Observation is **23-D** (no `sin/cos` phase). Weights go to `results/dr_rl_gym/` only. Unlike the gait harness, this one **ends the episode when unhealthy** (Gym Walker2d-v5: z outside `[0.8, 2.0]` or `|pitch| > 1.0`) so a lunge cannot farm `vx` down to the suite fall at 0.45 m.

```bash
python -m loka.dr_rl.train_gym_ppo
python -m loka.dr_rl.train_gym_ppo --seed 0 --timesteps 1000000
python -m loka.dr_rl.play --config loka/dr_rl/config_gym.yaml \
    --checkpoint results/dr_rl_gym/best
```

Periodic zips: `results/dr_rl_gym/checkpoints/ppo_walker_gym_<steps>_steps.zip`. Suite eval: `LOKA_DR_RL_CONFIG=loka/dr_rl/config_gym.yaml LOKA_DR_RL_CHECKPOINT=results/dr_rl_gym/ppo_walker_gym.zip`.

## Watch the nominal gait

Clean plant, no ice/backpack/dead hip — this is what the trained policy looks like when nothing is wrong:

```bash
python -m loka.dr_rl.play
python -m loka.dr_rl.play --checkpoint results/dr_rl/best
python -m loka.dr_rl.play --checkpoint results/dr_rl/best/best_model.zip
python -m loka.dr_rl.play --checkpoint results/dr_rl/checkpoints/ppo_walker_500000_steps.zip
# Gym-style (no gait clock); do not mix with results/dr_rl zips
python -m loka.dr_rl.play --config loka/dr_rl/config_gym.yaml \\
    --checkpoint results/dr_rl_gym/checkpoints/ppo_walker_gym_350000_steps.zip
python -m loka.run_walker_suite --baselines dr_rl --tests nominal
```

`--checkpoint` can be a zip or a directory. Periodic training snapshots are `results/dr_rl/checkpoints/ppo_walker_<steps>_steps.zip` (with a matching `*_vecnormalize_*_steps.pkl` beside them). Eval-best is `results/dr_rl/best/`. The latest full run is `results/dr_rl/ppo_walker.zip`.

Video lands at `results/dr_rl/play/<stamp>/nominal__dr_rl/episode.mp4` (play) or under `results/walker/<stamp>/nominal__dr_rl/` (suite).

## Evaluate on the Walker suite

```bash
python -m loka.run_walker_suite --baselines dr_rl
python -m loka.run_walker_suite --baselines loka,fixed_mpc,dr_rl --tests ice,backpack,shove_back,dead_right_hip
```

Override the checkpoint with `--checkpoint` on play. For the suite, set `dr_rl_checkpoint` and `dr_rl_config` in `loka/config/walker_suite.yaml` (or pass `--dr-rl-checkpoint` / `--dr-rl-config`). When those are unset, `LOKA_DR_RL_CHECKPOINT` and `LOKA_DR_RL_CONFIG` still apply. Put `vecnormalize.pkl` next to the zip.

Expected pattern for this **in-distribution** policy: partial on shove / ice / backpack, **fail** on dead hip and the box. That is the comparison LOKA is designed to win.
