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

## Project map

| Path | Role |
|------|------|
| `main.py` | Sim loop, viewer, failure/operator dispatch |
| `loka/` | Orchestrator, telemetry compression, sessions, model mutations |
| `models/walker/` | Default Walker MJCF / task XML |
| `system_prompt.txt` | LLM policy for YAML scratchpad edits |
| `environment.yml` | Conda env for this project |
| `paper/` | Optional LaTeX writeup (not required to run LOKA) |
