"""Phase A standing LOKA unit tests (no API key required)."""

from __future__ import annotations

import numpy as np
import pytest

from loka.control.locomotion import LocomotionController
from loka.sim import Simulation
from loka.agent.apply import apply_stand_scratchpad
from loka.agent.compress import synthesize_stand_telemetry
from loka.agent.error_defaults import default_stand_error_spec
from loka.agent.faults import FaultSpec, apply_plant_fault, clear_plant_faults
from loka.agent.runtime import LokaConfig, LokaRuntime


def test_lean_and_height_are_clamped():
    controller = LocomotionController()
    applied = controller.set_task_targets(
        {"lean_x": 1.0, "lean_y": -1.0, "height": 0.1, "yaw": 5.0}
    )
    assert abs(applied["lean_x"]) <= 0.06
    assert abs(applied["lean_y"]) <= 0.06
    assert applied["height"] >= 0.50
    assert abs(applied["yaw"]) <= 0.8


def test_apply_scratchpad_task_and_weights():
    controller = LocomotionController()
    loka_state = {
        "mutations": [],
        "nominal_gears": controller.robot.model.actuator_gear[:, 0].copy(),
        "error_spec": default_stand_error_spec(),
    }
    scratchpad = {
        "Semantic_State": {
            "Hypothesis": "operator crouch",
            "Analysis": "test",
        },
        "Controller_Targets": {
            "wbc.kp_base_position": 35.0,
            "mpc.friction_mu": 0.4,
        },
        "Task_Targets": {
            "height": 0.62,
            "lean_x": 0.02,
            "lean_y": -0.01,
            "yaw": 0.1,
        },
    }
    summary = apply_stand_scratchpad(controller, scratchpad, loka_state)
    assert summary["ok"]
    assert controller.config.wbc.kp_base_position == pytest.approx(35.0)
    assert controller.config.mpc.friction_mu == pytest.approx(0.4)
    snap = controller.task_snapshot()
    assert snap["height"] == pytest.approx(0.62)
    assert snap["lean_x"] == pytest.approx(0.02)
    assert snap["lean_y"] == pytest.approx(-0.01)


def test_apply_rejects_ambiguous_bare_weight_name():
    controller = LocomotionController()
    loka_state = {"mutations": [], "nominal_gears": controller.robot.model.actuator_gear[:, 0].copy()}
    scratchpad = {
        "Semantic_State": {"Hypothesis": "x", "Analysis": "y"},
        "Controller_Targets": {"weight_force": 1e-4},
    }
    summary = apply_stand_scratchpad(controller, scratchpad, loka_state)
    # Should not crash; bare name must not silently write both layers.
    assert controller.config.mpc.weight_force == pytest.approx(1e-6)
    assert controller.config.wbc.weight_force == pytest.approx(5e-3)
    assert summary["controller_targets"] == {}
    assert "weight_force" in summary["rejected_controller_targets"]


def test_apply_locks_robot_tuned_mpc_weights():
    controller = LocomotionController()
    before_z = controller.config.mpc.weight_position[2]
    before_contact = controller.config.wbc.weight_contact
    loka_state = {"mutations": [], "nominal_gears": controller.robot.model.actuator_gear[:, 0].copy()}
    scratchpad = {
        "Semantic_State": {"Hypothesis": "retune everything", "Analysis": "bad"},
        "Controller_Targets": {
            "mpc.weight_position_z": 6000.0,
            "wbc.weight_contact": 9000.0,
            "wbc.kp_base_position": 40.0,
            "stand.max_linear_acc": 12.0,
        },
    }
    summary = apply_stand_scratchpad(controller, scratchpad, loka_state)
    assert controller.config.mpc.weight_position[2] == pytest.approx(before_z)
    assert controller.config.wbc.weight_contact == pytest.approx(before_contact)
    assert controller.config.wbc.kp_base_position == pytest.approx(40.0)
    assert "mpc.weight_position_z" in summary["rejected_controller_targets"]
    assert "wbc.weight_contact" in summary["rejected_controller_targets"]
    assert "stand.max_linear_acc" in summary["rejected_controller_targets"]
    assert "wbc.kp_base_position" in summary["controller_targets"]


def test_compressor_emits_balance_section():
    sim = Simulation()
    runtime = LokaRuntime(sim, LokaConfig(enable_llm=False))
    for _ in range(50):
        runtime.step()
    text = synthesize_stand_telemetry(
        runtime.nominal_baseline,
        list(runtime.nominal_baseline),
        control_dt=sim.config.control_dt,
        error_spec=runtime.loka_state["error_spec"],
    )
    assert "BALANCE" in text
    assert "MOTOR LOAD" in text
    assert "DIRECTIVE" in text


def test_mass_fault_injection_changes_plant_not_belief():
    sim = Simulation()
    runtime = LokaRuntime(sim, LokaConfig(enable_llm=False))
    belief_mass = float(runtime.controller.robot.model.body_mass.sum())
    backup = apply_plant_fault(
        sim.model, FaultSpec("mass", 0.0, {"delta_kg": 5.0, "body": "torso_link"})
    )
    assert backup is not None
    assert float(sim.model.body_mass.sum()) == pytest.approx(belief_mass + 5.0, abs=0.2)
    # Controller belief model untouched.
    assert float(runtime.controller.robot.model.body_mass.sum()) == pytest.approx(
        belief_mass, abs=0.05
    )
    clear_plant_faults(sim.model, [backup])


def test_runtime_survives_scripted_lean_without_llm():
    sim = Simulation()
    runtime = LokaRuntime(sim, LokaConfig(enable_llm=False))
    for _ in range(200):
        runtime.step()
    runtime.controller.set_task_targets({"lean_x": 0.02, "height": 0.65})
    for _ in range(800):
        runtime.step()
        if sim.fell:
            break
    assert not sim.fell
    assert runtime.controller.command.height == pytest.approx(0.65)


def test_evaluate_lean_scenario():
    from loka.evaluate_loka import scenario_lean_operator_ablation

    result = scenario_lean_operator_ablation()
    assert not result.fell
    assert result.final_height == pytest.approx(0.65)
    assert result.final_lean_x == pytest.approx(0.025)


def test_evaluate_scripted_multi_turn_recovery():
    from loka.evaluate_loka import scenario_mass_fault_scripted_recovery

    result = scenario_mass_fault_scripted_recovery()
    assert not result.fell
    assert result.final_height == pytest.approx(0.64)
    assert "turns=2" in result.notes


def test_interactive_fault_commands_parse():
    from loka.agent.interactive import parse_fault_command

    assert parse_fault_command("help") == "help"
    assert parse_fault_command("clear") == "clear"
    mass = parse_fault_command("mass 7")
    assert mass.kind == "mass" and mass.params["delta_kg"] == 7.0
    assert mass.params["body"] == "torso_link"
    left = parse_fault_command("mass left 3")
    assert left.params["body"] == "left_shoulder_roll_link"
    assert left.params["delta_kg"] == 3.0
    right = parse_fault_command("mass 4 right")
    assert right.params["body"] == "right_shoulder_roll_link"
    assert parse_fault_command("ice").params["mu"] == 0.2
    push = parse_fault_command("push 8 left")
    assert push.kind == "push"
    assert push.params["impulse"] == 8.0
    assert push.params["direction"] == (0.0, 1.0, 0.0)
    assert parse_fault_command("please crouch") is None


def test_inject_and_clear_faults_update_viz():
    sim = Simulation()
    runtime = LokaRuntime(sim, LokaConfig(enable_llm=False))
    runtime.inject_fault_now(
        FaultSpec("mass", 0.0, {"delta_kg": 5.0, "body": "torso_link"})
    )
    assert runtime.viz.mass_delta_kg == pytest.approx(5.0)
    runtime.inject_fault_now(
        FaultSpec(
            "mass", 0.0, {"delta_kg": 3.0, "body": "left_shoulder_roll_link"}
        )
    )
    assert runtime.viz.mass_delta_kg == pytest.approx(8.0)
    assert len(runtime.viz.mass_loads) == 2
    runtime.inject_fault_now(FaultSpec("friction", 0.0, {"mu": 0.2}))
    assert runtime.viz.friction_mu == pytest.approx(0.2)
    runtime.clear_interactive_faults()
    assert runtime.viz.mass_delta_kg == 0.0
    assert runtime.viz.friction_mu is None


def test_mission_baseline_rebases_after_task_change():
    sim = Simulation()
    runtime = LokaRuntime(sim, LokaConfig(enable_llm=False))
    while not runtime._baseline_full:
        runtime.step()
    standing_z = float(np.mean([f["qpos"][2] for f in runtime.nominal_baseline]))
    assert standing_z > 0.7

    runtime.controller.set_task_targets({"height": 0.55})
    runtime._invalidate_mission_baseline(
        float(sim.data.time),
        reason="test crouch",
        close_failure_episode=False,
    )
    assert not runtime._baseline_full
    assert runtime._baseline_stale

    # Advance through settle + re-baseline window.
    t_end = float(sim.data.time) + 8.0
    while sim.data.time < t_end:
        runtime.step()
        if not runtime._baseline_stale and runtime._baseline_full:
            break
    assert runtime._baseline_full
    assert not runtime._baseline_stale
    crouched_z = float(np.mean([f["qpos"][2] for f in runtime.nominal_baseline]))
    assert crouched_z < standing_z - 0.05


def test_early_anomaly_fires_on_shoulder_mass_not_quiet_stand():
    from loka.agent.anomaly import assess_stand_anomaly

    quiet = {
        "com_error": np.zeros(3),
        "rpy": np.zeros(3),
        "support_margin": np.array([0.08, 0.08, 0.1, 0.1]),
        "desired_forces": np.zeros((8, 3)),
        "contact_forces": np.zeros((8, 3)),
        "contact_mask": np.ones(8, dtype=bool),
        "contact_speed_max": 0.0,
        "joint_util": np.zeros(29),
    }
    assert not assess_stand_anomaly(quiet).triggered

    # Healthy deep crouch: large fore-aft CoM residual, planted, low tilt — not a fault.
    crouch = {
        **quiet,
        "com_error": np.array([-0.025, 0.0, 0.011]),
        "rpy": np.array([0.0, np.deg2rad(0.4), 0.0]),
    }
    crouch_a = assess_stand_anomaly(crouch)
    assert crouch_a.balance_healthy
    assert not crouch_a.triggered

    # Asymmetric shoulder mass: lateral CoM bias should fire.
    loaded = {
        **quiet,
        "com_error": np.array([0.01, 0.03, 0.005]),
        "rpy": np.array([0.02, 0.0, 0.0]),
    }
    assessment = assess_stand_anomaly(loaded)
    assert assessment.triggered
    assert any("COM_BIAS" in c for c in assessment.clues)

    # Push-like distress: elevated tilt even with moderate CoM.
    tipped = {
        **quiet,
        "com_error": np.array([0.015, 0.0, 0.0]),
        "rpy": np.array([0.0, np.deg2rad(2.5), 0.0]),
    }
    assert assess_stand_anomaly(tipped).triggered

    # Nominal walk telemetry must not look like a plant fault.
    walk = {
        **quiet,
        "walking": True,
        "com_error": np.array([-0.042, 0.012, 0.012]),
        "rpy": np.array([np.deg2rad(0.8), np.deg2rad(0.5), 0.0]),
        "contact_speed_max": 0.13,
        "desired_forces": np.ones((8, 3)) * 40.0,
        "contact_forces": np.ones((8, 3)) * 10.0,  # ‖Δf‖ large but below walk bar
        "contact_mask": np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool),
        "wbc_failures": 1,
    }
    walk_a = assess_stand_anomaly(walk)
    assert not walk_a.triggered

    # Cumulative-style failures used to permanently arm QP_FALLBACK; per-tick
    # delta of 1 while standing still counts.
    qp = {**quiet, "wbc_failures": 1}
    assert assess_stand_anomaly(qp).triggered


def test_plateau_improvement_and_stop_reasons():
    from loka.agent.plateau import (
        EpisodeMetrics,
        is_improved,
        should_stop_episode,
    )

    mild = EpisodeMetrics(
        anomaly_score=0.27,
        com_xy_m=0.02,
        com_lat_m=0.015,
        com_error=(0.0, 0.015, -0.02),
        force_mismatch_n=60.0,
        tilt_rad=0.02,
        mpc_cost=5.0,
    )
    worse = EpisodeMetrics(
        anomaly_score=0.45,
        com_xy_m=0.04,
        com_lat_m=0.03,
        com_error=(0.0, 0.03, -0.04),
        force_mismatch_n=70.0,
        tilt_rad=0.03,
        mpc_cost=12.0,
    )
    better = EpisodeMetrics(
        anomaly_score=0.10,
        com_xy_m=0.008,
        com_lat_m=0.005,
        com_error=(0.0, 0.005, -0.01),
        force_mismatch_n=35.0,
        tilt_rad=0.005,
        mpc_cost=2.0,
    )
    assert not is_improved(worse, mild)
    assert is_improved(better, mild)
    assert should_stop_episode(
        n_interventions=5,
        stall_count=0,
        max_interventions=5,
        plateau_min_interventions=2,
        plateau_stall_rounds=2,
    ) == "max_interventions"
    assert should_stop_episode(
        n_interventions=2,
        stall_count=2,
        max_interventions=5,
        plateau_min_interventions=2,
        plateau_stall_rounds=2,
    ) == "plateau"
    assert (
        should_stop_episode(
            n_interventions=1,
            stall_count=2,
            max_interventions=5,
            plateau_min_interventions=2,
            plateau_stall_rounds=2,
        )
        is None
    )


def test_accept_residual_updates_mass_belief_and_lean():
    from loka.agent.session import FailureEpisode
    from loka.agent.plateau import EpisodeMetrics

    sim = Simulation()
    runtime = LokaRuntime(sim, LokaConfig(enable_llm=False))
    while not runtime._baseline_full:
        runtime.step()

    body = "right_shoulder_roll_link"
    import mujoco

    body_id = mujoco.mj_name2id(
        runtime.controller.robot.model, mujoco.mjtObj.mjOBJ_BODY, body
    )
    before = float(runtime.controller.robot.model.body_mass[body_id])

    metrics = EpisodeMetrics(
        anomaly_score=0.40,
        com_xy_m=0.035,
        com_lat_m=0.030,
        com_error=(-0.01, -0.030, -0.04),  # RIGHT bias (com_y < 0)
        force_mismatch_n=68.0,
        tilt_rad=0.03,
        mpc_cost=10.0,
    )
    runtime.failure_episode = FailureEpisode(
        started_at=float(sim.data.time),
        nominal_baseline=list(runtime.nominal_baseline),
    )
    # Pretend two stalled interventions already happened.
    runtime.failure_episode.interventions = [{}, {}]
    runtime.failure_episode.stall_count = 2

    runtime._accept_residual(
        metrics,
        reason="plateau",
        now=float(sim.data.time),
        live_pelvis_z=float(sim.data.qpos[2]),
    )

    after = float(runtime.controller.robot.model.body_mass[body_id])
    assert after > before + 0.5
    snap = runtime.controller.task_snapshot()
    assert snap["lean_y"] < 0.0  # lean toward right (heavy) side
    assert runtime.failure_episode is None
    assert runtime._residual_floor is not None
    assert runtime._baseline_stale


def test_gate_failure_dispatch_detects_plateau():
    from loka.agent.session import FailureEpisode
    from loka.agent.plateau import EpisodeMetrics

    sim = Simulation()
    runtime = LokaRuntime(
        sim,
        LokaConfig(
            enable_llm=False,
            max_interventions_per_episode=5,
            plateau_min_interventions=2,
            plateau_stall_rounds=2,
        ),
    )
    episode = FailureEpisode(started_at=0.0)
    episode.interventions = [{}, {}]  # two prior fixes
    runtime.failure_episode = episode

    mild = EpisodeMetrics(
        anomaly_score=0.25,
        com_xy_m=0.02,
        com_lat_m=0.015,
        com_error=(0.0, 0.015, -0.02),
        force_mismatch_n=55.0,
        tilt_rad=0.02,
        mpc_cost=4.0,
    )
    worse = EpisodeMetrics(
        anomaly_score=0.35,
        com_xy_m=0.03,
        com_lat_m=0.025,
        com_error=(0.0, 0.025, -0.03),
        force_mismatch_n=65.0,
        tilt_rad=0.025,
        mpc_cost=8.0,
    )
    worse2 = EpisodeMetrics(
        anomaly_score=0.40,
        com_xy_m=0.035,
        com_lat_m=0.028,
        com_error=(0.0, 0.028, -0.035),
        force_mismatch_n=70.0,
        tilt_rad=0.03,
        mpc_cost=11.0,
    )
    assert runtime._gate_failure_dispatch(mild) is None
    assert runtime._gate_failure_dispatch(worse) is None  # stall=1
    assert runtime._gate_failure_dispatch(worse2) == "plateau"


def test_session_log_rewrites_each_session(tmp_path):
    from loka.agent.session_log import StandSessionLog

    path = tmp_path / "loka" / "session.log"
    log1 = StandSessionLog(path)
    log1.dispatch(sim_time=1.0, reason="operator", user_content="FIRST")
    assert "FIRST" in path.read_text(encoding="utf-8")

    log2 = StandSessionLog(path)
    text = path.read_text(encoding="utf-8")
    assert "FIRST" not in text
    assert "rewritten each run" in text
    log2.dispatch(sim_time=2.0, reason="operator", user_content="SECOND")
    text = path.read_text(encoding="utf-8")
    assert "SECOND" in text
    assert "FIRST" not in text
    assert path.with_name("session.jsonl").exists()


def test_com_and_mass_mutations_reset_from_nominal():
    import mujoco

    from loka.agent.model_state import apply_loka_mutations

    controller = LocomotionController()
    model = controller.robot.model
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    assert body >= 0
    nominal_mass = float(model.body_mass[body])
    nominal_com = model.body_ipos[body].copy()
    loka_state = {"mutations": []}
    scratchpad = {
        "Semantic_State": {"Hypothesis": "offset load", "Analysis": "shift com and mass"},
        "Model_Mutations": [
            {
                "object_type": "body",
                "name": "torso_link",
                "attribute": "mass",
                "value": nominal_mass + 4.0,
            },
            {
                "object_type": "body",
                "name": "torso_link",
                "attribute": "com",
                "value": [-0.04, 0.03, 0.01],
            },
        ],
    }
    summary = apply_stand_scratchpad(controller, scratchpad, loka_state)
    assert summary["mutations"] == 2
    apply_loka_mutations(model, loka_state, loka_state["nominal_gears"])
    assert model.body_mass[body] == pytest.approx(nominal_mass + 4.0)
    np.testing.assert_allclose(model.body_ipos[body], [-0.04, 0.03, 0.01])
    model.body_mass[body] = 1.0
    model.body_ipos[body] = nominal_com
    apply_loka_mutations(model, loka_state, loka_state["nominal_gears"])
    assert model.body_mass[body] == pytest.approx(nominal_mass + 4.0)
    np.testing.assert_allclose(model.body_ipos[body], [-0.04, 0.03, 0.01])


def test_scratchpad_cannot_rewrite_a_locked_directive():
    from loka.agent.directives import MissionDirective

    controller = LocomotionController()
    loka_state = {
        "mutations": [],
        "nominal_gears": controller.robot.model.actuator_gear[:, 0].copy(),
        "error_spec": default_stand_error_spec(),
        "mission_directives": [
            MissionDirective("go to 10,10", 1.0, (10.0, 10.0))
        ],
    }
    apply_stand_scratchpad(
        controller,
        {
            "Semantic_State": {"Hypothesis": "done", "Analysis": "stop"},
            "Mission_Directives": ["stay here"],
            "Listeners": [
                {
                    "kind": "near_xy",
                    "x": 9.0,
                    "y": 9.0,
                    "radius": 0.5,
                    "message": "Close to the destination. Stop and stand.",
                }
            ],
        },
        loka_state,
        sim_time=2.0,
    )
    assert loka_state["mission_directives"][0].text == "go to 10,10"
    assert len(loka_state["listeners"]) == 1
    assert loka_state["listeners"][0].kind == "near_xy"


def test_near_listener_and_directive_registration():
    from loka.agent.directives import heartbeat_due, listener_fired, parse_goal_xy

    assert parse_goal_xy("go to coordinate 10, 10") == (10.0, 10.0)
    assert parse_goal_xy("please crouch") is None
    assert heartbeat_due(30.0, 0.0, 30.0)
    assert not heartbeat_due(29.0, 0.0, 30.0)

    sim = Simulation()
    runtime = LokaRuntime(sim, LokaConfig(enable_llm=False, heartbeat_s=30.0))
    runtime._baseline_full = True
    runtime.submit_operator("go to 10,10")
    runtime.step()
    assert runtime.loka_state["mission_directives"][0].goal_xy == (10.0, 10.0)
    assert "go to" not in runtime.loka_state["primary_objective"]
    listener = runtime.loka_state.get("listeners")
    assert listener == []
    from loka.agent.directives import parse_listeners

    runtime.loka_state["listeners"] = parse_listeners(
        [{
            "kind": "near_xy",
            "x": 9.0,
            "y": 9.0,
            "radius": 0.4,
            "message": "close to destination",
        }],
        armed_at=0.0,
    )
    far = {"qpos": np.array([1.0, 1.0, 0.7]), "time": 1.0}
    assert not listener_fired(runtime.loka_state["listeners"][0], far)
    near = {"qpos": np.array([8.8, 9.1, 0.7]), "time": 2.0}
    assert listener_fired(runtime.loka_state["listeners"][0], near)
    fired = runtime._consume_listeners(near)
    assert len(fired) == 1
    assert runtime.loka_state["listeners"] == []


def test_compressor_includes_full_state():
    sim = Simulation()
    runtime = LokaRuntime(sim, LokaConfig(enable_llm=False))
    runtime._baseline_full = True
    frame = runtime.step()
    text = synthesize_stand_telemetry(
        [frame],
        [frame],
        control_dt=sim.config.control_dt,
        error_spec=runtime.loka_state["error_spec"],
    )
    assert "FULL STATE" in text
    assert "pelvis xyz" in text
    assert "linear vel" in text
