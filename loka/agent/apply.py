"""Apply LOKA scratchpad YAML to the G1 standing controller."""

from __future__ import annotations

from typing import Any

import mujoco

from loka.control.locomotion import LocomotionController
from loka.agent.directives import LISTENER_KINDS, parse_listeners
from loka.agent.model_state import resolve_mjcf_name
from loka.agent.context import format_stand_configuration
from loka.agent.policy import filter_controller_targets


_SCRATCHPAD_KEYS = {
    "semantic_state": "Semantic_State",
    "belief_searches": "Belief_Searches",
    "belief_search": "Belief_Search",
    "task_targets": "Task_Targets",
    "listeners": "Listeners",
    "controller_targets": "Controller_Targets",
    "planner_targets": "Planner_Targets",
    "model_mutations": "Model_Mutations",
    "mission_directives": "Mission_Directives",
    "primary_objective": "Primary_Objective",
}
_STATE_KEYS = {
    "hypothesis": "Hypothesis",
    "hypotheses": "Hypotheses",
    "analysis": "Analysis",
}


def canonicalize_scratchpad(scratchpad: dict) -> dict:
    """Rename YAML keys the model lowercased back to the names the runtime reads."""
    renamed = {}
    for key, value in scratchpad.items():
        canon = _SCRATCHPAD_KEYS.get(str(key).strip().lower(), key)
        if canon == "Semantic_State" and isinstance(value, dict):
            value = {
                _STATE_KEYS.get(str(inner).strip().lower(), inner): item
                for inner, item in value.items()
            }
        renamed[canon] = value
    scratchpad.clear()
    scratchpad.update(renamed)
    return scratchpad


def apply_stand_scratchpad(
    controller: LocomotionController,
    scratchpad: dict[str, Any],
    loka_state: dict[str, Any],
    *,
    plant_model: mujoco.MjModel | None = None,
    sim_time: float | None = None,
    episode=None,
) -> dict[str, Any]:
    """Parse and apply a stand-oriented scratchpad.

    Returns a summary dict of what changed. Mutates ``controller`` and
    ``loka_state`` in place. Belief mutations land on
    ``controller.robot.model`` (the controller's private prior). Optional
    ``plant_model`` is only used to resolve names if the belief model lacks
    them — mutations still write the controller model.

    ``Controller_Targets`` are hard-filtered by
    :data:`loka.agent.policy.LLM_STAND_CONTROLLER_ALLOWLIST`.
    """
    summary: dict[str, Any] = {
        "ok": False,
        "controller_targets": {},
        "rejected_controller_targets": [],
        "task_targets": {},
        "mutations": 0,
        "listeners": 0,
    }
    if isinstance(scratchpad, dict):
        canonicalize_scratchpad(scratchpad)
    if not scratchpad or "Semantic_State" not in scratchpad:
        print("\n[!] Failed to parse LOKA scratchpad (missing Semantic_State).")
        return summary

    print("\n" + "=" * 55)
    print("[ORCHESTRATOR] LOKA STAND UPDATE")
    print("=" * 55)
    sem = scratchpad.get("Semantic_State", {})
    print(f"Hypothesis: {sem.get('Hypothesis', 'None')}")
    print(f"Analysis:   {sem.get('Analysis', 'None')}\n")

    if episode is not None:
        episode.record_intervention(sim_time, scratchpad)

    # -- Controller_Targets → allowlisted update_weights --------------------
    raw_weights = scratchpad.get("Controller_Targets") or {}
    weight_updates: dict[str, float] = {}
    for name, value in raw_weights.items():
        try:
            weight_updates[str(name)] = float(value)
        except (TypeError, ValueError):
            print(f"     * [WARN] Non-numeric Controller_Targets '{name}' (ignored)")
    if weight_updates:
        stack = getattr(getattr(controller, "config", None), "stack", None)
        allowed, rejected = filter_controller_targets(weight_updates, stack=stack)
        summary["rejected_controller_targets"] = rejected
        if rejected:
            print("  -> Controller_Targets locked (ignored):")
            for path in rejected:
                print(f"     * {path}")
        if allowed:
            try:
                applied = controller.update_weights(**allowed)
                summary["controller_targets"] = applied
                print("  -> Controller_Targets applied:")
                for path, value in applied.items():
                    print(f"     * {path} = {value:.4g}")
            except KeyError as exc:
                print(f"  -> Controller_Targets rejected: {exc}")
                known = {
                    k: v for k, v in allowed.items() if k in controller.tunables()
                }
                if known:
                    applied = controller.update_weights(**known)
                    summary["controller_targets"] = applied
                    print("  -> Partial Controller_Targets applied:")
                    for path, value in applied.items():
                        print(f"     * {path} = {value:.4g}")
        elif rejected:
            print("  -> No allowlisted Controller_Targets to apply.")

    # Planner_Targets are MJPC-only; ignore with a note.
    if scratchpad.get("Planner_Targets"):
        print("  -> Planner_Targets ignored on stand plant (no MJPC agent).")

    # -- Task_Targets -------------------------------------------------------
    task_targets = scratchpad.get("Task_Targets") or {}
    if task_targets:
        applied_tasks = controller.set_task_targets(task_targets)
        summary["task_targets"] = applied_tasks
        if applied_tasks:
            print("  -> Task_Targets applied:")
            for name, value in applied_tasks.items():
                print(f"     * {name} = {value:.4g}")

    # -- Model_Mutations (belief) -------------------------------------------
    belief_model = controller.robot.model
    mutations = scratchpad.get("Model_Mutations") or []
    if mutations:
        print("  -> Model_Mutations queued onto controller belief:")
        for mut in mutations:
            obj_type = mut.get("object_type")
            name = mut.get("name")
            attr = mut.get("attribute")
            val = mut.get("value")
            obj_enum = {
                "actuator": mujoco.mjtObj.mjOBJ_ACTUATOR,
                "geom": mujoco.mjtObj.mjOBJ_GEOM,
                "body": mujoco.mjtObj.mjOBJ_BODY,
            }.get(obj_type)
            if obj_enum is None:
                print(f"     * [WARN] Unknown object_type '{obj_type}'")
                continue
            resolved_name, obj_id = resolve_mjcf_name(belief_model, obj_enum, name)
            if obj_id == -1 and plant_model is not None:
                resolved_name, obj_id = resolve_mjcf_name(plant_model, obj_enum, name)
            if obj_id == -1:
                print(f"     * [WARN] Could not find {obj_type} named '{name}'")
                continue
            loka_state.setdefault("mutations", []).append(
                {
                    "type": obj_type,
                    "id": obj_id,
                    "attr": attr,
                    "val": val,
                    "name": resolved_name,
                    "applied_at": sim_time,
                }
            )
            summary["mutations"] += 1
            print(f"     * {obj_type} '{resolved_name}' -> {attr} = {val}")

        from loka.agent.model_state import apply_loka_mutations

        nominal = loka_state.get("nominal_gears")
        if nominal is None:
            nominal = belief_model.actuator_gear[:, 0].copy()
            loka_state["nominal_gears"] = nominal
        apply_loka_mutations(belief_model, loka_state, nominal)

    # -- Listeners (wake conditions; omit the key to leave them armed) -----
    if "Listeners" in scratchpad:
        raw = scratchpad.get("Listeners") or []
        parsed = parse_listeners(raw, armed_at=float(sim_time or 0.0))
        skipped = 0
        if isinstance(raw, dict):
            raw_items = [raw]
        elif isinstance(raw, list):
            raw_items = raw
        else:
            raw_items = []
        for item in raw_items:
            if not isinstance(item, dict):
                skipped += 1
                continue
            kind = str(item.get("kind") or item.get("when") or "")
            if kind not in LISTENER_KINDS or not str(item.get("message") or "").strip():
                skipped += 1
                print(f"     * [WARN] Listener ignored (kind={kind!r})")
        loka_state["listeners"] = parsed
        summary["listeners"] = len(parsed)
        if parsed:
            print("  -> Listeners armed:")
            for listener in parsed:
                print(f"     * {listener.format_line()}")
        else:
            print("  -> Listeners cleared.")
        if skipped:
            print(f"     * [WARN] {skipped} listener(s) dropped")

    if scratchpad.get("Mission_Directives") or scratchpad.get("Primary_Objective"):
        print(
            "  -> Mission_Directives / Primary_Objective ignored "
            "(operator directives are locked)."
        )

    print(format_stand_configuration(controller))
    print("=" * 55 + "\n")
    summary["ok"] = True
    return summary
