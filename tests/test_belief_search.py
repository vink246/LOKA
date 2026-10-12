"""Unbounded belief search, without a simulator."""

from types import SimpleNamespace

from loka.agent.belief_search import (
    BeliefSearchRunner,
    UnboundedSearch,
    parse_belief_searches,
    parse_search_command,
)
from loka.control.gait import MODE_STAND, MODE_WALK


def _run(search: UnboundedSearch, error_at):
    value = search.value
    decision = None
    for _ in range(search.max_probes + 1):
        decision = search.observe(error_at(value), abs(error_at(value)))
        if decision.done:
            return decision
        value = decision.value
    raise AssertionError("search did not finish")


def test_mass_search_finds_heavier_body():
    true = 14.598
    search = UnboundedSearch(
        9.598, mode="mul", expand=2.0, origin=9.598,
        error_tol=0.4, value_tol=0.1, lo=0.05, hi=80.0,
    )
    decision = _run(search, lambda value: true - value)
    assert decision.reason == "bias settled"
    assert abs(decision.value - true) < 0.5


def test_com_search_bisects_an_offset():
    true = 0.03
    search = UnboundedSearch(
        0.0, mode="add", initial_step=0.04, origin=0.0,
        error_tol=0.004, value_tol=0.005, unchanged_tol=0.001,
        lo=-0.4, hi=0.4,
    )
    decision = _run(search, lambda value: true - value)
    assert abs(decision.value - true) <= 0.005


def test_flat_residual_stops_and_reports_the_start():
    search = UnboundedSearch(
        1.0, mode="mul", expand=2.0, origin=1.0,
        error_tol=0.4, unchanged_tol=0.2, lo=0.0, hi=5.0,
    )
    decision = _run(search, lambda _value: 3.0)
    assert decision.done
    assert decision.reason == "bias did not move"
    assert decision.value == 1.0


def test_parse_search_commands():
    mass = parse_search_command("search mass")
    assert mass.kind == "mass" and mass.name == "torso_link"
    com = parse_search_command("search com y left")
    assert com.kind == "com" and com.axis == "y"
    assert com.name == "left_shoulder_roll_link"
    assert parse_search_command("search stop") == "stop"
    assert parse_search_command("mass 5") is None


def test_attempted_fixes_show_finished_search():
    from loka.agent.session import FailureEpisode

    episode = FailureEpisode(0.0)
    episode.interventions.append({
        "time": 1.0,
        "hypothesis": "torso mass",
        "search_results": ["mass torso_link done value=14.4 (bias settled)"],
    })
    text = episode.format_attempted_fixes()
    assert "mass torso_link done value=14.4 (bias settled)" in text
    episode.interrupted_walk = True
    stood = episode.build_initial_user_turn("tel", {}, 1.0, "cfg", just_stood=True)
    assert "already set gait.mode=stand" in stood
    assert "Do not command the stand" in stood
    assert "set gait.mode=walk" in stood
    later = episode.build_user_turn("tel", {}, 2.0, "cfg")
    assert "already set gait.mode=stand" not in later
    assert "set gait.mode=walk" in later
    quiet = FailureEpisode(0.0).build_initial_user_turn("tel", {}, 1.0, "cfg")
    assert "gait.mode=walk" not in quiet


def test_in_bounds_must_hold_before_clear():
    import numpy as np
    from loka.agent.anomaly import EffortDeviationObserver

    obs = EffortDeviationObserver()
    obs.dt = 0.002
    obs.update(np.zeros(2), np.zeros(12))
    assert obs.consecutive_in_bounds == 1
    assert (obs.consecutive_in_bounds * obs.dt) <= obs.clear_below_s
    obs.update(np.array([100.0, 0.0]), np.zeros(12))
    assert obs.consecutive_in_bounds == 0
    assert obs.consecutive_out_of_bounds == 1


def test_scratchpad_queues_mass_then_com_axes():
    requests = parse_belief_searches(
        {
            "Belief_Searches": [
                {"kind": "mass", "name": "torso"},
                {"kind": "com", "name": "torso_link", "axis": ["x", "y"]},
                {"kind": "gear", "name": "right_knee"},
            ]
        }
    )
    assert [(item.kind, item.name, item.axis) for item in requests] == [
        ("mass", "torso_link", None),
        ("com", "torso_link", "x"),
        ("com", "torso_link", "y"),
        ("gear", "right_knee", None),
    ]


def _runner(*, mode, walking, stop, searching_while_walking=True):
    gait = SimpleNamespace(
        config=SimpleNamespace(
            mode=mode, speed=0.15, stance_width=0.226, duty_factor=0.81,
        ),
        walking=walking,
        stop_in_progress=stop,
    )
    controller = SimpleNamespace(gait=gait, targets=None)

    def set_task_targets(updates):
        controller.targets = dict(updates)

    controller.set_task_targets = set_task_targets
    observer = SimpleNamespace(
        is_walking=searching_while_walking,
        step_period=0.70,
        window_size=100,
        dt=0.002,
        joint_bias=None,
    )
    runtime = SimpleNamespace(
        controller=controller,
        effort_observer=observer,
        sim=SimpleNamespace(data=SimpleNamespace(time=1.0), fell=False),
    )
    return BeliefSearchRunner(runtime), controller, observer


def test_search_waits_for_a_fault_stand_instead_of_overriding_it():
    runner, controller, observer = _runner(mode=MODE_STAND, walking=True, stop=True)
    runner._engage_measurement_gait()
    assert controller.targets is None
    assert runner._saved_gait is None
    assert runner._hold_for_stand

    runner.active = True
    runner._search = SimpleNamespace(value=9.6)
    runner._target = {}
    runner._request = SimpleNamespace(kind="mass", name="torso_link", axis=None)
    runner.tick()
    assert runner._hold_for_stand

    observer.is_walking = False
    runner.tick()
    assert not runner._hold_for_stand
    assert runner._ready_at > runner.runtime.sim.data.time


def test_search_gait_still_engages_during_a_live_walk():
    runner, controller, _observer = _runner(mode=MODE_WALK, walking=True, stop=False)
    runner._engage_measurement_gait()
    assert controller.targets == {
        "gait.speed": 0,
        "gait.stance_width": 0.28,
        "gait.duty_factor": 0.85,
    }
    assert runner._saved_gait["gait.speed"] == 0.15
    assert not runner._hold_for_stand
