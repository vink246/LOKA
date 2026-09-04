"""LLM orchestration: prompt assembly, YAML apply, planner recreation."""

import os

import mujoco
import yaml
from dotenv import load_dotenv
from mujoco_mpc import agent as mpc_agent
from openai import OpenAI

from loka.error_spec import format_error_spec, parse_error_tracking
from loka.mjcf_utils import (
    SUPPORTED_PLANNER_NUMERICS,
    set_model_numeric,
)
from loka.model_state import apply_loka_mutations
from loka.robot_context import (
    format_capabilities_block,
    format_current_error_tracking,
    format_primary_objective_block,
)

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

_PROMPT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "system_prompt.txt")


def resolve_mjcf_name(model, obj_enum, name):
    """Resolve an MJCF object name, tolerating LLM casing mistakes."""
    if not name:
        return None, -1

    obj_id = mujoco.mj_name2id(model, obj_enum, name)
    if obj_id != -1:
        return name, obj_id

    lowered = name.lower()
    if lowered != name:
        obj_id = mujoco.mj_name2id(model, obj_enum, lowered)
        if obj_id != -1:
            return lowered, obj_id

    if obj_enum == mujoco.mjtObj.mjOBJ_ACTUATOR:
        count = model.nu
    elif obj_enum == mujoco.mjtObj.mjOBJ_GEOM:
        count = model.ngeom
    elif obj_enum == mujoco.mjtObj.mjOBJ_BODY:
        count = model.nbody
    else:
        return None, -1

    for i in range(count):
        candidate = mujoco.mj_id2name(model, obj_enum, i)
        if candidate and candidate.lower() == lowered:
            return candidate, i

    return None, -1


def _snapshot_agent_params(agent):
    try:
        weights = {name: float(val) for name, val in agent.get_cost_weights().items()}
    except Exception:
        weights = {}
    try:
        tasks = dict(agent.get_task_parameters())
    except Exception:
        tasks = {}
    return weights, tasks


def reinit_agent_from_belief(model, agent, task_id):
    """Rebuild the MJPC server from the *belief* model only.

    The C++ agent holds a serialized copy of whatever MjModel we pass to Init.
    That copy must be nominal hardware plus LOKA Model_Mutations — never the
    hidden plant. Cost weights and task parameters are restored after Init.
    """
    weights, tasks = _snapshot_agent_params(agent)
    agent.close()
    agent = mpc_agent.Agent(task_id=task_id, model=model)
    if weights:
        try:
            agent.set_cost_weights(weights)
        except Exception:
            pass
    for name, value in tasks.items():
        try:
            if isinstance(value, str):
                agent.set_task_parameter(name, value)
            else:
                agent.set_task_parameter(name, float(value))
        except Exception:
            pass
    return agent


def recreate_agent_with_planner_settings(model, agent, settings, task_id):
    """Apply planner settings on the belief model and recreate the MPC agent."""
    applied = {}
    for name, value in settings.items():
        if name not in SUPPORTED_PLANNER_NUMERICS:
            continue
        set_model_numeric(model, name, float(value))
        applied[name] = float(value)

    if not applied:
        return agent

    return reinit_agent_from_belief(model, agent, task_id)


def _live_task_parameter_names(agent) -> set[str]:
    try:
        return set(agent.get_task_parameters().keys())
    except Exception:
        return set()


def load_system_prompt(
    robot_context=None,
    objective=None,
    capabilities=None,
    error_spec=None,
):
    """Assemble the system prompt from the generic template plus live model context."""
    with open(_PROMPT_PATH, encoding="utf-8") as handle:
        system_prompt = handle.read().rstrip()

    blocks = [system_prompt]
    if objective:
        blocks.append(format_primary_objective_block(objective))
    if capabilities:
        blocks.append(format_capabilities_block(capabilities))
    if error_spec is not None:
        blocks.append(format_current_error_tracking(error_spec))
    if robot_context:
        blocks.append(robot_context)
    return "\n\n".join(blocks)


def llm_worker(api_messages, result_queue):
    """Runs in a background thread to prevent freezing the physics loop."""
    try:
        response = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=api_messages,
            temperature=0.1,
        )
        yaml_text = response.choices[0].message.content
        yaml_text = yaml_text.replace("```yaml", "").replace("```", "").strip()
        scratchpad = yaml.safe_load(yaml_text)
        result_queue.put({"scratchpad": scratchpad, "raw_yaml": yaml_text})
    except Exception as e:
        print(f"\n[LLM Worker Error] {e}")
        result_queue.put(None)


def apply_scratchpad(
    agent,
    scratchpad,
    loka_state,
    model,
    episode=None,
    data=None,
    task_id="Walker",
):
    """Parses the LLM's YAML and safely applies it to the running MPC and loka_state."""
    if not scratchpad or "Semantic_State" not in scratchpad:
        print("\n[!] Failed to parse LOKA scratchpad.")
        return agent

    print("\n" + "=" * 55)
    print("[ORCHESTRATOR] LOKA COGNITIVE UPDATE RECEIVED")
    print("=" * 55)

    sem_state = scratchpad.get("Semantic_State", {})
    print(f"Hypothesis: {sem_state.get('Hypothesis', 'None')}")
    print(f"Analysis:   {sem_state.get('Analysis', 'None')}\n")

    if episode is not None:
        episode.record_intervention(data.time if data is not None else None, scratchpad)

    weights_dict = {}
    for weight_name, new_value in scratchpad.get("Controller_Targets", {}).items():
        try:
            weights_dict[weight_name] = float(new_value)
        except Exception:
            pass

    mutations = scratchpad.get("Model_Mutations", [])
    queued_mutations = 0
    if mutations:
        print("\n  -> Internal Model Mutations Queued:")
        for mut in mutations:
            obj_type = mut.get("object_type")
            name = mut.get("name")
            attr = mut.get("attribute")
            val = mut.get("value")

            obj_enum = None
            if obj_type == "actuator":
                obj_enum = mujoco.mjtObj.mjOBJ_ACTUATOR
            elif obj_type == "geom":
                obj_enum = mujoco.mjtObj.mjOBJ_GEOM
            elif obj_type == "body":
                obj_enum = mujoco.mjtObj.mjOBJ_BODY

            if obj_enum is not None:
                resolved_name, obj_id = resolve_mjcf_name(model, obj_enum, name)
                if obj_id != -1:
                    loka_state["mutations"].append({
                        "type": obj_type,
                        "id": obj_id,
                        "attr": attr,
                        "val": val,
                        "name": resolved_name,
                        "applied_at": data.time if data is not None else None,
                    })
                    queued_mutations += 1
                    if resolved_name != name:
                        print(
                            f"     * Queued {obj_type} '{resolved_name}' "
                            f"(resolved from '{name}') -> {attr} = {val}"
                        )
                    else:
                        print(f"     * Queued {obj_type} '{resolved_name}' -> {attr} = {val}")
                else:
                    print(f"     * [WARN] Could not find {obj_type} named '{name}'")

    # Belief model = nominal MJCF + LOKA mutations. Never serialize plant faults.
    nominal_gears = loka_state.get("nominal_gears")
    if nominal_gears is None:
        nominal_gears = model.actuator_gear[:, 0].copy()
    apply_loka_mutations(model, loka_state, nominal_gears)

    planner_targets = scratchpad.get("Planner_Targets", {})
    if planner_targets:
        print("  -> Planner Metaparameters Updated:")
        try:
            agent = recreate_agent_with_planner_settings(
                model, agent, planner_targets, task_id
            )
            for param_name, new_value in planner_targets.items():
                if param_name in SUPPORTED_PLANNER_NUMERICS:
                    print(f"     * {param_name} = {new_value}")
                else:
                    print(f"     * [WARN] Unknown planner setting '{param_name}' (ignored)")
            if data is not None:
                agent.set_state(
                    time=data.time,
                    qpos=data.qpos,
                    qvel=data.qvel,
                    act=data.act,
                )
        except Exception as e:
            print(f"     * Failed to apply planner settings: {e}")
    elif queued_mutations:
        try:
            agent = reinit_agent_from_belief(model, agent, task_id)
            if data is not None:
                agent.set_state(
                    time=data.time,
                    qpos=data.qpos,
                    qvel=data.qvel,
                    act=data.act,
                )
            print("  -> Planner internal model rebuilt from LOKA belief")
        except Exception as e:
            print(f"  -> Failed to rebuild planner belief: {e}")

    if weights_dict:
        try:
            agent.set_cost_weights(weights_dict)
            print("  -> Cost Weights Updated:")
            for weight_name, value in weights_dict.items():
                print(f"     * {weight_name} = {value}")
        except Exception as e:
            print(f"  -> Failed to apply cost weights: {e}")

    allowed_task_params = _live_task_parameter_names(agent)
    task_targets = scratchpad.get("Task_Targets", {})
    if task_targets:
        print("  -> Task Parameters Updated:")
        for param_name, new_value in task_targets.items():
            if allowed_task_params and param_name not in allowed_task_params:
                print(f"     * [WARN] Unknown task parameter '{param_name}' (ignored)")
                continue
            try:
                agent.set_task_parameter(param_name, float(new_value))
                print(f"     * {param_name} = {new_value}")
            except Exception as e:
                print(f"     * Failed to set {param_name}: {e}")

    error_tracking = scratchpad.get("Error_Tracking")
    if error_tracking is not None:
        print("  -> Error Tracking Updated:")
        try:
            new_spec = parse_error_tracking(error_tracking, model.nq, model.nv)
            loka_state["error_spec"] = new_spec
            print(format_error_spec(new_spec))
        except Exception as e:
            print(f"     * [WARN] Invalid Error_Tracking (ignored): {e}")

    print("=" * 55 + "\n")
    return agent
