"""Standing LOKA policy: which controller knobs the LLM may touch.

The full ``loka.control.tuning`` catalogue remains available to the GUI /
human dashboard. The orchestrator only sees and applies the adaptive subset
below — robot-specific QP costs and saturation caps stay locked at their
tuned values.
"""

from __future__ import annotations

from loka.control import tuning

# Adaptive / belief-facing knobs only.
# Locked on purpose: mpc.weight_*, wbc.weight_contact / base task priorities,
# contact_kd, stand.max_*_acc, upper/swing posture — those define the certified
# standing plant and caused falls when the LLM rewrote them wholesale.
LLM_STAND_CONTROLLER_ALLOWLIST: frozenset[str] = frozenset(
    {
        # Friction belief (ice / slip)
        "mpc.friction_mu",
        "wbc.friction_mu",
        # CoM / torso impedance (soften under CoP saturation or mass shift)
        "wbc.kp_base_position",
        "wbc.kd_base_position",
        "wbc.kp_base_orientation",
        "wbc.kd_base_orientation",
        # Let legs bend for crouch / limp instead of fighting the height task
        "wbc.weight_posture_legs",
        "wbc.kp_posture_legs",
        "wbc.kd_posture_legs",
        # Swing-foot Cartesian tracking while walking
        "wbc.weight_swing_foot",
        "wbc.kp_swing_foot",
        "wbc.kd_swing_foot",
    }
)

_UNKNOWN = object()


def assert_allowlist_subset_of_tunables() -> None:
    missing = sorted(LLM_STAND_CONTROLLER_ALLOWLIST - set(tuning.BY_PATH))
    if missing:
        raise RuntimeError(f"Allowlist paths not in tuning catalogue: {missing}")


def filter_controller_targets(
    updates: dict[str, float],
    stack: str | None = None,
) -> tuple[dict[str, float], list[str]]:
    """Split updates into (allowed, rejected_paths).

    ``stack=None`` keeps the historical allowlist, so callers that have not
    been taught about stacks stay on ``legacy_dcm``.
    """
    allow = LLM_STAND_CONTROLLER_ALLOWLIST
    if stack is not None:
        from loka.control.stacks import SPECS

        allow = SPECS[stack].llm_allowlist
    allowed: dict[str, float] = {}
    rejected: list[str] = []
    for path, value in updates.items():
        if path in allow:
            allowed[path] = value
        else:
            rejected.append(path)
    return allowed, sorted(rejected)


def stand_llm_catalogue() -> str:
    """Prompt catalogue restricted to the standing allowlist."""
    assert_allowlist_subset_of_tunables()
    items = [tuning.BY_PATH[p] for p in sorted(LLM_STAND_CONTROLLER_ALLOWLIST)]
    width = max(len(t.path) for t in items)
    return "\n".join(
        f"{t.path:<{width}}  [{t.low:g}, {t.high:g}]  {t.summary}" for t in items
    )


def locked_controller_paths() -> list[str]:
    return sorted(set(tuning.BY_PATH) - LLM_STAND_CONTROLLER_ALLOWLIST)
