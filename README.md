# LOKA

**LLM-Orchestrated Kinematic Adaptation** — an immune system for dynamic locomotion on [MuJoCo MPC (MJPC)](https://github.com/google-deepmind/mujoco_mpc).

Foundation models reason about the world in language; bipeds are dynamically unstable and fall easily. Bridging that gap is hard: VLAs struggle to compose novel skills online, Domain Randomization cannot isolate discrete unmodeled faults at runtime, and code-as-policy LLM loops are too slow (and brittle) for balancing underactuated robots. LOKA resolves this latency–stability paradox with a hybrid neuro-symbolic architecture: keep a mathematically transparent MPC in the fast loop, and let a language model act as a slow, online system-identification engine that rewrites the controller’s objective landscape and physical self-awareness by mutating typed parameters mid-flight.

When motors seize, friction collapses, mass shifts, or an operator says *“crouch and walk at 1 m/s”*, LOKA compresses proprioceptive anomalies into semantic tags, diagnoses in the background, and applies surgical edits—cost weights, planner numerics, task goals, mission criteria, even virtual amputations (zero gear on a dead actuator)—so the robot can rediscover a gait before collapse.

### Key Contributions

- **Effort-driven telemetry compression** — monitors torque deficits, energy gradients, and mission error bands; delta-masks nominal behavior into dense semantic tags that fire *before* tracking error dooms the state.
- **Asynchronous dual-rate execution** — high-frequency MJPC never waits on the LLM; inference runs on a background thread with thread-safe queues while the plant stays under predictive control.
- **Dynamic kinematic malleability** — a YAML scratchpad turns the LLM into a live system ID engine with authority over costs, planner settings, task targets, `Error_Tracking`, and MJCF-aligned model belief (gear / friction / mass).

## Layout

Suggested sibling layout (paths below assume this):

```text
Research/
├── LOKA/           # this repo
└── mujoco_mpc/     # google-deepmind/mujoco_mpc (clone + build)
```

You can clone `mujoco_mpc` elsewhere; you only need its Python package installed into the `loka` conda env.

## Setup

### 1. Conda env

```bash
cd LOKA
conda env create -f environment.yml
conda activate loka
```

Update later with `conda env update -f environment.yml --prune`.

### 2. Build and install MuJoCo MPC

Follow upstream steps in the [mujoco_mpc README](https://github.com/google-deepmind/mujoco_mpc#installation), then install the Python API **into the `loka` env**:

```bash
# from a sibling of LOKA, or any path you prefer
git clone https://github.com/google-deepmind/mujoco_mpc.git
cd mujoco_mpc

# system deps (Ubuntu example — see upstream README for your OS)
# sudo apt-get install cmake libgl1-mesa-dev libxinerama-dev libxcursor-dev \
#   libxrandr-dev libxi-dev ninja-build zlib1g-dev clang

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build . --config Release
cd ../python
conda activate loka
python setup.py install
python mujoco_mpc/agent_test.py   # smoke-check the binding
```

Pin **MuJoCo 3.2.x** in the env (already set in `environment.yml`) so it matches a typical MJPC build. If the agent server fails to start, rebuild MJPC against the same MuJoCo version you install with pip.

### 3. API key

```bash
cd LOKA
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

## Run

```bash
conda activate loka
cd LOKA
python main.py
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `--task-id Walker` | MJPC task id (default: Walker; uses `models/walker/`) |
| `--xml-path PATH` | Override task XML |
| `--objective "..."` | Mission text for the orchestrator |
| `--record [PATH]` | Record from a tracking camera to video (default: `loka_recording.mp4`) |
| `--record-camera NAME` | Camera for `--record` (default: `side_follow`, moves beside the robot) |

Operator requests: type in the same terminal while the sim runs (optional `loka:` prefix), e.g. `walk crouched at 1 m/s`.

## Tests

Unit tests (no live LLM; `test_error_spec` does not need the agent server):

```bash
conda activate loka
cd LOKA
python -m unittest discover -s tests -v
```

MJPC weight API smoke test (needs `mujoco_mpc` installed):

```bash
python tests/test_weights.py
```

## Walker perturbation suite

The suite walks the biped toward a distance goal under each configured plant perturbation, once per baseline. MuJoCo's integrator is deterministic, so trials would otherwise be copies of each other. Each trial draws a seeded initial state: `Uniform(-init_noise, init_noise)` on `qpos` and `qvel` after the nominal reset (`init_noise` defaults to `0.005`, the same half-width as DR-RL reset noise). Trial `k` uses `seed + k` for every baseline, so LOKA, fixed MJPC, and DR-RL start from the same pose on that trial and a different pose on the next one. Set `init_noise: 0` to freeze the spawn pose.

`num_trials` defaults to `1`. The YAML key `numtrials` is accepted as an alias.

```bash
conda activate loka
cd LOKA
python -m loka.run_walker_suite \
  --baselines loka,fixed_mpc,dr_rl \
  --num-trials 5
```

A shorter slice:

```bash
python -m loka.run_walker_suite \
  --tests nominal,ice \
  --baselines fixed_mpc \
  --num-trials 2 \
  --timeout 12 \
  --no-record
```

Other flags: `--tests`, `--baselines`, `--goal-distance`, `--perturbation-time`, `--timeout`, `--speed-goal`, `--seed`, `--init-noise`, `--dr-rl-checkpoint`, `--dr-rl-config`, `--output-dir`, `--trial`, `--run-dir`, `--no-record`. Config defaults live in `loka/config/walker_suite.yaml`.

On a PACE-ICE login node, one Slurm job per baseline trial (every perturbation in that trial), CPU only, at most 10 jobs in flight:

```bash
bash scripts/pace_ice_walker_suite.sh --num-trials 5 --cpus 32 \
    --conda-prefix /home/hice1/vkulkarni46/scratch/envs/loka
```

`--conda-prefix` is the env directory to activate on the login node and on every job. Pass `--partition`, `--account`, `--qos`, or `--constraint` when the default partition is not the CPU node you want. `sinfo -o "%P %c %f"` lists partitions, CPU counts, and features. A later job merges the partials and writes the plots into `results/walker/pace_<timestamp>/`.

`command_latency` holds the motor command back by `delay_steps` control updates (10 ms each; `8` is 80 ms, the same count as TWIST `action_buf_len`). The planner and the policy still emit the latest command. Only the plant input is late, and it holds that delayed command across the physics steps until the next update. The shipped test turns the lag on at `perturbation_time_s: 0`. `action_buf_len` is accepted as an alias for `delay_steps`.

The `dr_rl` baseline reads `dr_rl_checkpoint` and `dr_rl_config` from that file. Either may be null. A checkpoint path is a zip or a directory containing one; the config path is the DR-RL training YAML, and it has to match that policy. When a field is null, `LOKA_DR_RL_CHECKPOINT` / `LOKA_DR_RL_CONFIG` still apply, then `loka/dr_rl/config.yaml`.

```yaml
dr_rl_checkpoint: results/dr_rl/checkpoints/ppo_walker_500000_steps.zip
dr_rl_config: loka/dr_rl/config.yaml
```

Each run writes `results/walker/<timestamp>/`. Every trial is its own folder, including the video when recording is on:

```text
results/walker/<timestamp>/
├── config.resolved.yaml
├── summary.json
├── summary.csv
├── statistics.json
├── statistics.csv
├── plots/
│   ├── ice_metrics.png
│   └── ice_rms_over_time.png
└── ice__loka/
    ├── trial_00/
    │   ├── episode.mp4
    │   ├── timeseries.csv
    │   ├── timeseries.npz
    │   └── metadata.json
    └── trial_01/
```

After the trials of one test and baseline finish, the runner prints that condition's averages. When the whole grid is done it writes `statistics.json` / `statistics.csv` and one pair of charts per perturbation:

- **Bar chart** (`<test>_metrics.png`): success rate, task completion time, time to recovery, RMS error, mean ground-reaction-force magnitude, and time not fallen. Baseline is on the x axis. Error bars are the sample standard deviation across trials. The RMS panel groups height, pitch, speed, and the combined residual.
- **Line chart** (`<test>_rms_over_time.png`): 0.5 s rolling RMS of height, pitch, speed, and the combined residual, then ground-reaction-force magnitude and its horizontal part when the log has them. One line per baseline (mean across trials, band is the trial standard deviation).
- **Non-foot contact** (`nonfoot_floor_contacts.png`): one chart for the whole suite. Each perturbation is a group, and each baseline is a bar. The height is the fraction of log samples where a knee, thigh, or the torso was touching the floor. Those contacts are flags written at log time (`nonfoot_floor`, `floor_<geom>`), and `nonfoot_geoms` on the episode lists which of them touched.

Definitions, all taken from the 100 Hz log:

| Metric | Definition |
|--------|------------|
| Success rate | Fraction of trials that reach the distance goal. A fall does not end the episode. |
| Task completion time | Mean `t_end` among successful trials. Timeouts are left out of this average. |
| Height RMS | RMS of torso height minus the Height Goal (1.2 m). |
| Pitch RMS | RMS of torso pitch minus upright (0 rad). |
| Speed RMS | RMS of forward speed minus the speed goal (default 1 m/s). |
| Combined RMS | `sqrt(mean(e_height² + e_pitch² + e_speed²))`. |
| Time to recovery | Longest interval from a fail sample until the walker is nominal again. Nominal means not fallen and inside the tracking-error band, held for 0.5 s. Standing up without that hold does not end the interval, and a brief dip under the deadband does not either. The first second after spawn is ignored. If the episode ends before the hold, the open interval runs to the end of the log. A trial that never fails after the first second scores 0. |
| Mean \|GRF\| | Mean magnitude of the force the floor applies to the walker. Feet, knees, thighs, and the torso all count. A trudging gait spends more of the step in hard contact, so this sits higher than a gait that lets the body unload. |
| Time not fallen | Seconds the torso stays upright, integrated from the `fallen` flag. Gaps use the state at the start of the gap, including the span from the last sample to `t_end`. |
| Non-foot floor contact | Fraction of samples where any geom other than the feet is touching the floor. Knees and the torso are logged by name at that sample. |

## Project map

| Path | Role |
|------|------|
| `main.py` | Sim loop, viewer, failure/operator dispatch |
| `loka/` | Orchestrator, telemetry compression, sessions, model mutations |
| `loka/dr_rl/` | Domain-randomized PPO walker baseline (Gymnasium + SB3, native MuJoCo) |
| `loka/walker_suite/` | Perturbation suite: seeded trials, logs, videos, statistics, plots |
| `loka/config/walker_suite.yaml` | Default tests, baselines, `num_trials`, seed, and recording |
| `models/walker/` | Default Walker MJCF / task XML |
| `system_prompt.txt` | LLM policy for YAML scratchpad edits |
| `environment.yml` | Conda env for this project |
| `paper/` | Optional LaTeX writeup (not required to run LOKA) |
