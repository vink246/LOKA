# LOKA

**LLM-Orchestrated Kinematic Adaptation** — an immune system for dynamic locomotion on [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) with the Unitree G1.

Foundation models reason about the world in language; bipeds are dynamically unstable and fall easily. Bridging that gap is hard: VLAs struggle to compose novel skills online, Domain Randomization cannot isolate discrete unmodeled faults at runtime, and code-as-policy LLM loops are too slow (and brittle) for balancing underactuated robots. LOKA resolves this latency–stability paradox with a hybrid neuro-symbolic architecture: keep a mathematically transparent controller in the fast loop, and let a language model act as a slow, online system-identification engine that rewrites the controller’s objective landscape and physical self-awareness by mutating typed parameters mid-flight.

When motors seize, friction collapses, mass shifts, or an operator says *“crouch and walk at 1 m/s”*, LOKA compresses proprioceptive anomalies into semantic tags, diagnoses in the background, and applies surgical edits—cost weights, planner numerics, task goals, mission criteria, even virtual amputations (zero gear on a dead actuator)—so the robot can rediscover a gait before collapse.

### Key Contributions

- **Effort-driven telemetry compression** — monitors torque deficits, energy gradients, and mission error bands; delta-masks nominal behavior into dense semantic tags that fire *before* tracking error dooms the state.
- **Asynchronous dual-rate execution** — the high-frequency control loop never waits on the LLM; inference runs on a background thread with thread-safe queues while the plant stays under predictive control.
- **Dynamic kinematic malleability** — a YAML scratchpad turns the LLM into a live system ID engine with authority over costs, planner settings, task targets, `Error_Tracking`, and MJCF-aligned model belief (gear / friction / mass).

## Layout

```text
LOKA/
├── unitree_mujoco/   # git submodule: Unitree MuJoCo sim + robot MJCF
├── sim/              # G1 launcher, DDS interface, demos
├── loka/             # Orchestrator, telemetry compression, sessions, model mutations
├── models/           # Task / MJCF assets
├── paper/            # Optional LaTeX writeup (not required to run)
├── environment.yml
└── .gitmodules
```

Suggested sibling layout for DDS deps (any path works):

```text
Research/
├── LOKA/
├── cyclonedds/              # eclipse-cyclonedds 0.10.x (build → install/)
└── unitree_sdk2_python/     # pip install -e . into the loka env
```

## Setup

### 1. Clone with submodule

```bash
git clone --recurse-submodules <LOKA-url>
cd LOKA
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

### 2. Conda env

```bash
conda env create -f environment.yml
conda activate loka
```

Update later with `conda env update -f environment.yml --prune`.

### 3. cyclonedds + unitree_sdk2_python

The Python simulator talks DDS via [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python). Install into the `loka` env (clone anywhere; `~` is only an example):

```bash
# Build cyclonedds 0.10.x (required if pip cannot find it)
cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir -p build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install

# Install unitree_sdk2_python
cd ~
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
export CYCLONEDDS_HOME=~/cyclonedds/install
conda activate loka
pip install -e .
```

If `pip install -e .` fails with `Could not locate cyclonedds`, ensure `CYCLONEDDS_HOME` points at the `install` directory above.

### 4. Enable multicast on loopback (sim DDS)

On many hosts (including WSL), `lo` starts without multicast and the sim/controller never see each other:

```bash
sudo ip link set lo multicast on
```

You should no longer see `selected interface "lo" is not multicast-capable` when launching. Re-run after reboot if needed.

### 5. API key (orchestrator)

```bash
cd LOKA
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

## Run

Two terminals. Simulation DDS uses **domain id 1** and interface **`lo`** (real robot uses domain `0` and the robot NIC).

### Terminal 1 — simulator

```bash
conda activate loka
cd LOKA
python sim/run_g1_sim.py
```

You should see the MuJoCo viewer with the G1 (29 DoF). Elastic band is enabled for humanoid hang/lift:

| Key | Action |
|-----|--------|
| `9` | Attach / release virtual band |
| `8` | Lift |
| `7` | Lower |

### Terminal 2 — control / smoke test

```bash
conda activate loka
cd LOKA
python sim/demos/g1_torque_smoke.py          # 1 Nm on all motors
python sim/demos/g1_torque_smoke.py --zero   # zero torque hold
```

Or use the low-level interface directly:

```bash
python sim/g1_interface.py
```

`LOKA_G1_Interface` publishes `rt/lowcmd` and subscribes `rt/lowstate` with **unitree_hg** messages. Torque-only commands set `kp=kd=0`. CRC is applied on every write. Live IMU in the smoke test is reported as quaternion / gyro / accelerometer (the sim bridge fills those fields).

## Demos

| Demo | Command | Notes |
|------|---------|-------|
| G1 sim | `python sim/run_g1_sim.py` | G1 29 DoF, elastic band, domain 1 / `lo` |
| G1 torque smoke | `python sim/demos/g1_torque_smoke.py` | Subscribe lowstate, publish constant torque |
| Upstream Go2 sim | `cd unitree_mujoco/simulate_python && python unitree_mujoco.py` | Default Go2 `config.py` |
| Upstream Go2 SDK test | `cd unitree_mujoco/simulate_python && python test/test_unitree_sdk2.py` | Go2 / `unitree_go`; start Go2 sim first |
| Upstream Go2 stand | `python unitree_mujoco/example/python/stand_go2.py` | No NIC → sim; with NIC → real robot |

Joystick test (optional; needs a gamepad and `USE_JOYSTICK=1` in upstream config):

```bash
cd unitree_mujoco/simulate_python
python test/gamepad_test.py
```

## Sim to real

| Mode | Domain | Interface |
|------|--------|-----------|
| Simulation | `1` | `lo` |
| Real G1 | `0` | robot NIC, e.g. `enp3s0` |

```bash
python sim/demos/g1_torque_smoke.py enp3s0
python sim/g1_interface.py enp3s0
python unitree_mujoco/example/python/stand_go2.py enp3s0   # Go2 example
```

Omit the NIC → simulation; pass the NIC → hardware.

## Project map

| Path | Role |
|------|------|
| `sim/run_g1_sim.py` | Launch G1 Python sim (unitree_mujoco) |
| `sim/g1_interface.py` | `LOKA_G1_Interface` (unitree_hg DDS) |
| `sim/demos/` | Smoke / demo controllers |
| `unitree_mujoco/` | Submodule: MuJoCo sim, bridges, robot MJCF |
| `loka/` | Orchestrator, telemetry compression, sessions, model mutations |
| `models/` | Task / MJCF assets |
| `system_prompt.txt` | LLM policy for YAML scratchpad edits |
| `environment.yml` | Conda env for this project |
| `paper/` | Optional LaTeX writeup (not required to run LOKA) |
