"""Stance support repair when a declared near pad starts ahead of the robot."""

from __future__ import annotations

import numpy as np
import pytest

from loka.control.terrain import TerrainBelief, Volume, ensure_stance_support


def _short_near_gap_belief() -> TerrainBelief:
    """LOKA-style two-pad gap: near pad starts at x≈0.44, robot at x≈0."""
    belief = TerrainBelief()
    belief.set_volumes(
        [
            Volume(
                name="near",
                center=np.array([0.72, 0.0]),
                size=np.array([0.56, 1.5]),
                top=0.10,
                source="llm",
            ),
            Volume(
                name="far",
                center=np.array([2.09, 0.0]),
                size=np.array([1.98, 1.5]),
                top=0.10,
                source="llm",
            ),
        ]
    )
    return belief


def test_ensure_stance_support_extends_short_near_pad():
    belief = _short_near_gap_belief()
    feet = np.array([[-0.02, 0.10, 0.10], [-0.02, -0.10, 0.10]])
    assert belief.height_at(feet[0, :2]) == 0.0

    changed = ensure_stance_support(belief, feet)
    assert changed
    assert belief.height_at(feet[0, :2]) == pytest.approx(0.10)
    assert belief.height_at(feet[1, :2]) == pytest.approx(0.10)
    # Gap between pads must stay open.
    assert belief.height_at([1.05, 0.0]) == 0.0
    # Far pad untouched in height.
    assert belief.height_at([2.0, 0.0]) == pytest.approx(0.10)


def test_ensure_stance_support_noop_when_already_covered():
    belief = _short_near_gap_belief()
    feet = np.array([[0.6, 0.10, 0.10], [0.6, -0.10, 0.10]])
    assert not ensure_stance_support(belief, feet)
    assert belief.height_at(feet[0, :2]) == pytest.approx(0.10)


def test_ensure_stance_support_inserts_pad_when_nearest_is_far():
    belief = TerrainBelief()
    belief.set_volumes(
        [
            Volume(
                name="far_only",
                center=np.array([2.5, 0.0]),
                size=np.array([1.0, 1.5]),
                top=0.10,
                source="llm",
            )
        ]
    )
    feet = np.array([[0.0, 0.10, 0.10], [0.0, -0.10, 0.10]])
    assert ensure_stance_support(belief, feet)
    assert any(v.name == "stance_support" for v in belief.volumes)
    assert belief.height_at(feet[0, :2]) == pytest.approx(0.10)
    # Must not bridge into the far pad.
    assert belief.height_at([1.2, 0.0]) == 0.0


def test_ensure_stance_support_snaps_overestimated_pad_tops():
    """Vision declared 0.129 m; soles rest at 0.10 m — snap coplanar pads."""
    belief = TerrainBelief()
    belief.set_volumes(
        [
            Volume(
                name="near",
                center=np.array([0.6, 0.0]),
                size=np.array([0.8, 1.5]),
                top=0.129,
                source="llm",
            ),
            Volume(
                name="far",
                center=np.array([2.05, 0.0]),
                size=np.array([1.9, 1.5]),
                top=0.129,
                source="llm",
            ),
        ]
    )
    feet = np.array([[0.3, 0.10, 0.10], [0.3, -0.10, 0.10]])
    assert ensure_stance_support(belief, feet)
    assert belief.height_at(feet[0, :2]) == pytest.approx(0.10)
    assert belief.height_at([2.0, 0.0]) == pytest.approx(0.10)
    assert belief.height_at([1.05, 0.0]) == 0.0


def test_ensure_stance_support_does_not_flatten_next_stair_tread():
    belief = TerrainBelief()
    belief.set_volumes(
        [
            Volume(name="s0", center=np.array([0.15, 0.0]), size=np.array([0.3, 1.0]), top=0.07),
            Volume(name="s1", center=np.array([0.45, 0.0]), size=np.array([0.3, 1.0]), top=0.14),
        ]
    )
    feet = np.array([[0.15, 0.10, 0.07], [0.15, -0.10, 0.07]])
    assert not ensure_stance_support(belief, feet)
    assert belief.height_at([0.45, 0.0]) == pytest.approx(0.14)
