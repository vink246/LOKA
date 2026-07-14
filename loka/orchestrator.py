import os
import yaml
import mujoco
from mujoco_mpc import agent as mpc_agent
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SUPPORTED_PLANNER_NUMERICS = frozenset({"agent_horizon", "sampling_exploration"})
SUPPORTED_TASK_PARAMETERS = frozenset({"Height Goal", "Speed Goal"})


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


def set_model_numeric(model, name, value):
    """Set an MJCF custom numeric on the MuJoCo model."""
    numeric_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, name)
    if numeric_id == -1:
        raise ValueError(f"Custom numeric '{name}' not found in model")
    addr = model.numeric_adr[numeric_id]
    model.numeric_data[addr] = float(value)


def get_model_numeric(model, name):
    """Read an MJCF custom numeric from the MuJoCo model."""
    numeric_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, name)
    if numeric_id == -1:
        return None
    addr = model.numeric_adr[numeric_id]
    return float(model.numeric_data[addr])


def recreate_agent_with_planner_settings(model, agent, settings, task_id):
    """Apply planner settings and recreate the MPC agent to pick them up."""
    applied = {}
    for name, value in settings.items():
        if name not in SUPPORTED_PLANNER_NUMERICS:
            continue
        set_model_numeric(model, name, float(value))
        applied[name] = float(value)

    if not applied:
        return agent

    agent.close()
    return mpc_agent.Agent(task_id=task_id, model=model)


def load_system_prompt(robot_context=None):
    with open("system_prompt.txt", encoding="utf-8") as handle:
        system_prompt = handle.read()
    if robot_context:
        system_prompt = f"{system_prompt.rstrip()}\n\n{robot_context}"
    return system_prompt


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


def apply_scratchpad(agent, scratchpad, loka_state, model, episode=None, data=None, task_id="Walker"):
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

    if weights_dict:
        try:
            agent.set_cost_weights(weights_dict)
            print("  -> Cost Weights Updated:")
            for weight_name, value in weights_dict.items():
                print(f"     * {weight_name} = {value}")
        except Exception as e:
            print(f"  -> Failed to apply cost weights: {e}")

    task_targets = scratchpad.get("Task_Targets", {})
    if task_targets:
        print("  -> Task Parameters Updated:")
        for param_name, new_value in task_targets.items():
            if param_name not in SUPPORTED_TASK_PARAMETERS:
                print(f"     * [WARN] Unknown task parameter '{param_name}' (ignored)")
                continue
            try:
                agent.set_task_parameter(param_name, float(new_value))
                print(f"     * {param_name} = {new_value}")
            except Exception as e:
                print(f"     * Failed to set {param_name}: {e}")

    mutations = scratchpad.get("Model_Mutations", [])
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
                    if resolved_name != name:
                        print(
                            f"     * Queued {obj_type} '{resolved_name}' "
                            f"(resolved from '{name}') -> {attr} = {val}"
                        )
                    else:
                        print(f"     * Queued {obj_type} '{resolved_name}' -> {attr} = {val}")
                else:
                    print(f"     * [WARN] Could not find {obj_type} named '{name}'")

    print("=" * 55 + "\n")
    return agent
