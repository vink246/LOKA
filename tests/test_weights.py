import os
import mujoco
import mujoco_mpc
from mujoco_mpc import agent as mpc_agent

def main():
    # 1. Locate and load the bundled XML
    mjpc_dir = os.path.dirname(mujoco_mpc.__file__)
    xml_path = os.path.join(mjpc_dir, "mjpc", "tasks", "walker", "task.xml")
    
    print(f"Loading model from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    
    # 2. Instantiate the MPC Agent
    agent = mpc_agent.Agent(task_id="Walker", model=model)
    
    # 3. Read and print default weights directly from the backend
    print("\n--- DEFAULT COST WEIGHTS ---")
    default_weights = agent.get_cost_weights()
    for name, value in default_weights.items():
        print(f"{name}: {value}")
        
    # 4. Mock the dictionary exactly as the LLM orchestrator would generate it
    mock_loka_targets = {
        "Height": 45.0,
        "Rotation": 8.0,
        "Speed": 2.5,
        "Control": 0.5
    }
    
    print("\n--- INJECTING MOCK LOKA WEIGHTS ---")
    print(f"Payload: {mock_loka_targets}")
    
    # 5. The critical test: executing the API binding
    try:
        agent.set_cost_weights(mock_loka_targets)
        print("Status: SUCCESS! set_cost_weights() executed without errors.")
    except Exception as e:
        print(f"Status: FAILED with error: {e}")
        return
        
    # 6. Verify the memory actually mutated in the C++ backend
    print("\n--- VERIFYING UPDATED WEIGHTS ---")
    updated_weights = agent.get_cost_weights()
    for name, value in updated_weights.items():
        print(f"{name}: {value}")

if __name__ == "__main__":
    main()