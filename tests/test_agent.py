"""Phase A standing LOKA unit tests (no API key required)."""

from __future__ import annotations

import numpy as np
import pytest

from loka.control.locomotion import LocomotionController
from loka.sim import Simulation
from loka.agent.apply import apply_stand_scratchpad
from loka.agent.compress import synthesize_stand_telemetry
from loka.agent.faults import FaultSpec, apply_plant_fault, clear_plant_faults
from loka.agent.runtime import LokaConfig, LokaRuntime


def test_lowercase_scratchpad_sections_are_accepted():
    controller = LocomotionController()
    loka_state = {
        "mutations": [],
        "nominal_gears": controller.robot.model.actuator_gear[:, 0].copy(),
    }
    scratchpad = {
        "semantic_state": {
            "hypothesis": "torso_link com y",
            "analysis": "waist_roll dominates",
        },
        "belief_searches": [{"kind": "com", "name": "torso_link", "axis": "y"}],
    }
    summary = apply_stand_scratchpad(controller, scratchpad, loka_state)
    assert summary["ok"] is True
    assert scratchpad["Semantic_State"]["Hypothesis"] == "torso_link com y"
    assert scratchpad["Belief_Searches"][0]["axis"] == "y"


def test_lean_and_height_are_clamped():
    controller = LocomotionController()
    applied = controller.set_task_targets(
        {"lean_x": 1.0, "lean_y": -1.0, "height": 0.1, "yaw": 5.0}
    )
    assert abs(applied["lean_x"]) <= 0.06
    assert abs(applied["lean_y"]) <= 0.06
    assert applied["height"] >= 0.50
    assert abs(applied["yaw"]) <= 0.8


def test_apply_scratchpad_task_targets():
    controller = LocomotionController()
    loka_state = {
        "mutations": [],
        "nominal_gears": controller.robot.model.actuator_gear[:, 0].copy(),
    }
    scratchpad = {
        "Semantic_State": {
            "Hypothesis": "operator crouch",
            "Analysis": "test",
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
    snap = controller.task_snapshot()
    assert snap["height"] == pytest.approx(0.62)
    assert snap["lean_x"] == pytest.approx(0.02)
    assert snap["lean_y"] == pytest.approx(-0.01)


def test_apply_never_changes_controller_tunables():
    controller = LocomotionController()
    before = controller.tunables()
    loka_state = {"mutations": [], "nominal_gears": controller.robot.model.actuator_gear[:, 0].copy()}
    scratchpad = {
        "Semantic_State": {"Hypothesis": "retune everything", "Analysis": "bad"},
        "Controller_Targets": {
            "mpc.weight_position_z": 6000.0,
            "wbc.kp_base_position": 40.0,
            "mpc.friction_mu": 0.1,
        },
    }
    summary = apply_stand_scratchpad(controller, scratchpad, loka_state)
    assert summary["ok"]
    assert controller.tunables() == before


def test_compressor_emits_state_effort_and_cost():
    sim = Simulation()
    runtime = LokaRuntime(sim, LokaConfig(enable_llm=False))
    for _ in range(50):
        runtime.step()
    text = synthesize_stand_telemetry(
        runtime.recent_frames,
        control_dt=sim.config.control_dt,
    )
    assert "FULL STATE" in text
    assert "EFFORT DEVIATION" in text
    assert "STATE COST" in text


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
    weak = parse_fault_command("weak")
    assert weak.kind == "actuator_scale"
    assert weak.params["actuator"] == "right_knee"
    assert weak.params["scale"] == pytest.approx(0.75)
    scaled = parse_fault_command("scale left_knee 0.25")
    assert scaled.params["actuator"] == "left_knee"
    assert scaled.params["scale"] == pytest.approx(0.25)


def test_number_keys_add_shoulder_mass_and_scale_knee_torque():
    import mujoco

    from loka.agent.interactive import InteractivePlant, fault_for_keycode

    sim = Simulation()
    plant = InteractivePlant(sim)
    shoulder = mujoco.mj_name2id(
        sim.model, mujoco.mjtObj.mjOBJ_BODY, "left_shoulder_roll_link"
    )
    knee = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_knee")
    mass0 = float(sim.model.body_mass[shoulder])
    gear0 = float(sim.model.actuator_gear[knee, 0])
    belief_mass = float(sim.controller.robot.model.body_mass[shoulder])
    belief_gear = float(sim.controller.robot.model.actuator_gear[knee, 0])

    joint = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "right_knee_joint")
    dof = int(sim.model.jnt_dofadr[joint])
    data = mujoco.MjData(sim.model)
    mujoco.mj_resetDataKeyframe(sim.model, data, 0)
    data.ctrl[:] = 0.0
    data.ctrl[knee] = 40.0
    mujoco.mj_forward(sim.model, data)
    # Motor torque on the joint is gear * ctrl. actuator_force itself stays
    # equal to ctrl; the gear is what the plant delivers.
    full = float(data.qfrc_actuator[dof])

    assert fault_for_keycode(ord("1")) is None
    assert fault_for_keycode(ord("5")).params["body"] == "left_shoulder_roll_link"
    assert fault_for_keycode(ord("4")).params["mu"] == pytest.approx(0.2)
    assert fault_for_keycode(ord("8")) == "clear"
    # GLFW keypad codes: KP_5 = 325, KP_2 = 322.
    assert fault_for_keycode(325).params["body"] == "left_shoulder_roll_link"
    assert fault_for_keycode(322).kind == "actuator_scale"
    assert fault_for_keycode(322).params["scale"] == pytest.approx(0.75)

    plant.handle_key(ord("5"))
    assert sim.model.body_mass[shoulder] == pytest.approx(mass0 + 5.0)
    # Key repeat must not stack another 5 kg.
    plant.handle_key(ord("5"))
    assert sim.model.body_mass[shoulder] == pytest.approx(mass0 + 5.0)
    assert sim.controller.robot.model.body_mass[shoulder] == pytest.approx(belief_mass)

    plant.handle_key(ord("2"))
    assert sim.model.actuator_gear[knee, 0] == pytest.approx(0.75 * gear0)
    plant._keys.interval = 0.0
    plant.handle_key(ord("2"))
    assert sim.model.actuator_gear[knee, 0] == pytest.approx(0.75 * gear0)
    assert sim.controller.robot.model.actuator_gear[knee, 0] == pytest.approx(belief_gear)
    assert sim.model.actuator_ctrlrange[knee, 1] == pytest.approx(139.0)

    mujoco.mj_forward(sim.model, data)
    assert float(data.qfrc_actuator[dof]) == pytest.approx(0.75 * full, rel=1e-5)

    plant.handle_key(ord("8"))
    assert sim.model.body_mass[shoulder] == pytest.approx(mass0)
    assert sim.model.actuator_gear[knee, 0] == pytest.approx(gear0)


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
    log2.dispatch(
        sim_time=2.0,
        reason="operator",
        user_content="SECOND",
        system_prompt="SYSTEM PROMPT\n\nROBOT MODEL REFERENCE\nbody tree",
    )
    text = path.read_text(encoding="utf-8")
    assert "SECOND" in text
    assert "FIRST" not in text
    assert "SYSTEM PROMPT" not in text
    assert "ROBOT MODEL REFERENCE" not in text
    jsonl = path.with_name("session.jsonl").read_text(encoding="utf-8")
    assert "SYSTEM PROMPT" not in jsonl
    assert "ROBOT MODEL REFERENCE" not in jsonl


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
    from loka.agent.directives import Listener

    bias_listener = Listener(
        message="Bias is back in bounds. Walk to the goal.",
        kind="bias_under_threshold",
    )
    assert not listener_fired(bias_listener, {"bias_nm": 8.0, "bias_threshold_nm": 5.0})
    assert not listener_fired(
        bias_listener,
        {"bias_nm": 1.0, "bias_threshold_nm": 5.0, "belief_search_busy": True},
    )
    assert listener_fired(bias_listener, {"bias_nm": 1.0, "bias_threshold_nm": 5.0})
    fired = runtime._consume_listeners(near)
    assert len(fired) == 1
    assert runtime.loka_state["listeners"] == []


def test_compressor_includes_full_state():
    sim = Simulation()
    runtime = LokaRuntime(sim, LokaConfig(enable_llm=False))
    frame = runtime.step()
    text = synthesize_stand_telemetry(
        [frame],
        control_dt=sim.config.control_dt,
    )
    assert "FULL STATE" in text
    assert "pelvis xyz" in text
    assert "linear vel" in text


def test_full_state_flags_walking_away_from_the_goal():
    """The t=62s go-to snapshot: world speed looks slow, but it points backward."""
    from loka.agent.compress import format_full_state_section

    frame = {
        "time": 62.43,
        "qpos": np.array([2.456, 2.523, 0.792, 1, 0, 0, 0]),
        "qvel": np.array([-0.081, 0.015, 0.002, 0, 0, 0]),
        "rpy": np.zeros(3),
        "walking": True,
        "goal_active": True,
        "goal_x": 4.0,
        "goal_y": 4.0,
        "heading_goal": 0.759,
        "cmd_mode": 1.0,
        "cmd_speed": 0.14,
        "cmd_heading": 0.759,
        "heading_error_raw": 0.0,
        "heading_error": 0.0,
        "cross_track": 0.001,
        "yaw_ref": 0.759,
    }
    text = format_full_state_section([frame])
    assert "MOVING AWAY FROM GOAL" in text
    assert "along heading" in text
    assert "closing speed" in text


def test_max_memory_drops_prior_turns_and_cannot_exceed_max_turns():
    from loka.agent.session import FailureEpisode, OperatorSession

    with pytest.raises(AssertionError, match="max_memory"):
        LokaConfig(max_turns=2, max_memory=3)

    episode = FailureEpisode(0.0)
    episode.append_exchange("see the torso mass fix", "queued mass")
    episode.record_intervention(
        1.0,
        {
            "Semantic_State": {"Hypothesis": "torso mass", "Analysis": "heavy"},
            "Model_Mutations": [{
                "object_type": "body",
                "name": "torso_link",
                "attribute": "mass",
                "value": 14.4,
            }],
        },
    )
    assert episode.prior_turn_count == 1
    forgotten = episode.build_user_turn(
        "telemetry",
        {"primary_objective": "stand"},
        2.0,
        "cfg",
        max_turns=5,
        max_memory=1,
    )
    assert "torso mass" not in forgotten
    assert "torso_link" not in forgotten
    assert "see the torso mass fix" not in forgotten
    assert "Earlier orchestrator turns are omitted" in forgotten
    assert "remain in effect" in forgotten
    blank = episode.compose_api_messages("system", forgotten, max_memory=1)
    assert [item["role"] for item in blank] == ["system", "user"]

    remembered = episode.build_user_turn(
        "telemetry",
        {"primary_objective": "stand"},
        2.0,
        "cfg",
        max_turns=5,
        max_memory=2,
    )
    assert "torso mass" in remembered
    assert "torso_link" in remembered
    full = episode.compose_api_messages("system", remembered, max_memory=2)
    assert any(item["content"] == "see the torso mass fix" for item in full)

    session = OperatorSession()
    session.append_exchange("go to 3,3", "walking")
    request = session.build_request_turn(
        "go to 4,4", "telemetry", {}, 3.0, "cfg", max_memory=1
    )
    assert "go to 3,3" not in request
    assert "Earlier turns in this session are not included" in request
    operator_messages = session.compose_api_messages("system", request, max_memory=1)
    assert [item["role"] for item in operator_messages] == ["system", "user"]


def test_failure_episode_turn_cap_resets_when_bias_recovers():
    sim = Simulation()
    runtime = LokaRuntime(
        sim,
        LokaConfig(
            enable_llm=True,
            cooldown_s=0.0,
            max_turns=2,
            max_memory=2,
            print_compressor=False,
        ),
    )
    calls = []

    def fake_dispatch(session, user, *, reason="unknown"):
        del session, reason
        calls.append(user)
        return True

    runtime._dispatch = fake_dispatch
    obs = runtime.effort_observer

    def hold_fault(*_args, **_kwargs):
        obs.consecutive_out_of_bounds = obs.consecutive_steps
        obs.consecutive_in_bounds = 0
        return 0.0, np.zeros(1), np.zeros(12)

    def hold_clear(*_args, **_kwargs):
        obs.consecutive_out_of_bounds = 0
        obs.consecutive_in_bounds = 10**6
        return 0.0, np.zeros(1), np.zeros(12)

    obs.update = hold_fault
    for _ in range(4):
        runtime.step()
    assert len(calls) == 2
    assert "turn 1 of 2" in calls[0]
    assert "turn 2 of 2" in calls[1]
    assert "last turn" in calls[1]
    assert runtime.failure_episode is not None
    assert runtime.failure_episode.turns_sent == 2

    obs.update = hold_clear
    runtime.step()
    assert runtime.failure_episode is None

    obs.update = hold_fault
    runtime.step()
    assert len(calls) == 3
    assert "turn 1 of 2" in calls[2]


def test_walk_is_blocked_while_bias_is_above_threshold():
    controller = LocomotionController()
    controller.gait.block_walk = True
    applied = controller.set_task_targets(
        {
            "gait.mode": "walk",
            "gait.speed": 0.25,
            "gait.goal_x": 3.0,
            "gait.goal_y": 3.0,
        }
    )
    assert float(controller.gait.config.mode) == 0.0
    assert applied.get("gait.mode", 0.0) == 0.0
    controller.gait._steer_to_goal(np.zeros(2))
    assert float(controller.gait.config.mode) == 0.0

    controller.gait.block_walk = False
    applied = controller.set_task_targets({"gait.mode": "walk", "gait.speed": 0.25})
    assert applied["gait.mode"] == 1.0
