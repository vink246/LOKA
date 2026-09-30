"""Closed-loop turning and gait-change gates.

Fast cases run in the default suite. The rest are ``slow``. Scenarios that
the current plant cannot pass are ``xfail(strict=True)`` until the phase
that fixes them removes the marker.
"""

from __future__ import annotations

import pytest

from loka.walk_bench import scenarios, run_scenario

# Closed-loop results on the yaw-aware plant. Anything not listed here is an
# open failure (strict xfail) so a surprise pass is visible.
PASSING = ("T1", "T2", "T3", "T5", "G1", "G2", "G3", "G4", "G5b", "S0", "P2")
# Other stacks have not cleared Gate 3. Keep them out of the strict-xfail list
# (group "alip") and do not pretend they pass the legacy set.
PASSING_BY_STACK = {
    "legacy_dcm": PASSING,
    "alip_footstep": (
        "T1", "T2", "T5", "G1", "G3", "S0", "B2", "N1", "S1", "P2",
        "P1_fwd_mid_8", "P1_back_mid_8", "P1_left_mid_8", "P1_right_mid_8",
    ),
}
FAST = ("T1", "T5", "T7", "G2", "G3")


def _by_name():
    return {row.name: row for row in scenarios()}


def _run(name: str):
    result = run_scenario(_by_name()[name])
    assert result.passed, f"{name}: {result.reasons}"


@pytest.mark.parametrize("name", [n for n in FAST if n in PASSING])
def test_fast_transition(name):
    _run(name)


@pytest.mark.parametrize("name", [n for n in FAST if n not in PASSING])
@pytest.mark.xfail(strict=True, reason="large heading changes and turn-in-place still fall or miss")
def test_fast_transition_open(name):
    _run(name)


@pytest.mark.slow
@pytest.mark.parametrize("name", [n for n in PASSING if n not in FAST])
def test_slow_transition_passing(name):
    _run(name)


_OPEN = [
    row.name for row in scenarios()
    if row.name not in PASSING and not row.name.startswith("P1_") and row.group != "alip"
]


@pytest.mark.slow
@pytest.mark.parametrize("name", _OPEN)
@pytest.mark.xfail(strict=True, reason="large turns and abrupt speed steps are still open")
def test_slow_transition_open(name):
    _run(name)


@pytest.mark.slow
@pytest.mark.parametrize(
    "name", [row.name for row in scenarios() if row.name.startswith("P1_")]
)
@pytest.mark.xfail(reason="push matrix is recorded; not every impulse is cleared yet")
def test_push_matrix(name):
    _run(name)
