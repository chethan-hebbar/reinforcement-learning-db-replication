import requests
import time
import json
import argparse
import numpy as np
from stable_baselines3 import PPO

# --- Configuration ---
CONTROLLER_URL = "http://localhost:8080"
EVALUATION_DURATION_MINS = 20
POLLING_INTERVAL_SECS = 10
DECISION_INTERVAL_SECS = 60 # How often the RL agent makes a decision
NUM_KEYS = 5 # Should match the environment constants

# --- Helper Functions (adapted from our environment) ---

def get_system_state():
    """Fetches the current state of the entire cluster from the controller."""
    try:
        response = requests.get(f"{CONTROLLER_URL}/rl/system-state", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not get system state. Is the controller running? {e}")
        return None

def parse_state_to_observation(state_json):
    """Converts the JSON state into the NumPy vector for the agent."""
    if not state_json:
        return np.zeros(NUM_KEYS * 3 * 3, dtype=np.float32) # 3 nodes, 3 metrics

    key_order = [f"user_profile_{i}" for i in range(NUM_KEYS)]
    presence = np.zeros((3, NUM_KEYS), dtype=np.float32)
    read_counts = np.zeros((3, NUM_KEYS), dtype=np.float32)
    write_counts = np.zeros((3, NUM_KEYS), dtype=np.float32)

    for i, node_data in enumerate(state_json):
        for key_name, metrics in node_data.get('keyMetrics', {}).items():
            if key_name in key_order:
                key_idx = key_order.index(key_name)
                presence[i, key_idx] = 1.0
                read_counts[i, key_idx] = metrics.get('readCount', 0)
                write_counts[i, key_idx] = metrics.get('writeCount', 0)
    
    return np.concatenate([presence.flatten(), read_counts.flatten(), write_counts.flatten()])

def decode_action(action_id):
    """Converts an integer action back into a command."""
    is_evict = action_id >= (NUM_KEYS * 3)
    if is_evict:
        action_id -= (NUM_KEYS * 3)
    
    key_id = action_id // 3
    node_id = action_id % 3
    
    action_type = "EVICT" if is_evict else "REPLICATE"
    key_name = f"user_profile_{key_id}"
    node_name = f"replication-{['us', 'eu', 'ap'][node_id]}"
    return action_type, key_name, node_name

def execute_action(action_type, key, node):
    """Sends the chosen action to the controller."""
    payload = {"actionType": action_type, "key": key, "targetNode": node}
    try:
        requests.post(f"{CONTROLLER_URL}/rl/execute-action", json=payload, timeout=5)
        print(f"ACTION EXECUTED: {action_type} {key} on {node}")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not execute action: {e}")

def calculate_system_metrics(state_json):
    """Calculates aggregate latency and cost from the system state."""
    if not state_json:
        return 0, 0

    total_storage_cost = sum(node.get('storageCost', 0) for node in state_json)
    
    # This logic is a simplified simulation of latency. It does not use the
    # controller's reported latency but calculates it based on the state,
    # which is more stable for comparison.
    total_reads = 0
    predicted_latency_sum = 0
    presence_map = {}
    
    for i, node_data in enumerate(state_json):
        node_name = node_data['nodeId']
        for key_name, metrics in node_data.get('keyMetrics', {}).items():
            if key_name not in presence_map:
                presence_map[key_name] = set()
            presence_map[key_name].add(node_name)

    for node_data in state_json:
        requesting_node_id = node_data['nodeId']
        for key_name, metrics in node_data.get('keyMetrics', {}).items():
            reads = metrics.get('readCount', 0)
            total_reads += reads
            is_local = requesting_node_id in presence_map.get(key_name, set())
            latency = 10 if is_local else 150 # local vs remote latency
            predicted_latency_sum += reads * latency

    avg_latency = (predicted_latency_sum / total_reads) if total_reads > 0 else 0
    return avg_latency, total_storage_cost


# --- Main Evaluation Loop ---

def run_evaluation(mode, model_path=None):
    print(f"--- Starting Evaluation in '{mode.upper()}' Mode ---")
    
    model = None
    if mode == 'rl':
        if not model_path:
            print("ERROR: Must provide --model_path for 'rl' mode.")
            return
        print(f"Loading trained model from {model_path}...")
        model = PPO.load(model_path)

    results = []
    start_time = time.time()
    last_decision_time = 0

    while time.time() - start_time < EVALUATION_DURATION_MINS * 60:
        current_time = time.time()
        
        # --- RL Agent makes a decision every DECISION_INTERVAL_SECS ---
        if mode == 'rl' and current_time - last_decision_time >= DECISION_INTERVAL_SECS:
            print("\n--- RL Agent making a decision ---")
            last_decision_time = current_time
            state_json = get_system_state()
            if state_json:
                observation = parse_state_to_observation(state_json)
                action, _ = model.predict(observation, deterministic=True)
                action_type, key, node = decode_action(action.item())
                execute_action(action_type, key, node)
            print("---------------------------------\n")

        # --- Poll for metrics every POLLING_INTERVAL_SECS ---
        state_json = get_system_state()
        avg_latency, total_cost = calculate_system_metrics(state_json)
        
        elapsed_time = current_time - start_time
        print(f"Time: {int(elapsed_time)}s, Avg Latency: {avg_latency:.2f}ms, Total Cost: ${total_cost:.2f}")
        
        results.append({
            "time": elapsed_time,
            "avg_latency": avg_latency,
            "total_cost": total_cost
        })

        time.sleep(POLLING_INTERVAL_SECS)

    # Save results to a file
    output_filename = f"evaluation_results_{mode}.json"
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"--- Evaluation Finished. Results saved to {output_filename} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation for the adaptive replication system.")
    parser.add_argument("--mode", type=str, required=True, choices=['static', 'rl'], help="The evaluation mode to run.")
    parser.add_argument("--model_path", type=str, default="ppo_replication_policy.zip", help="Path to the saved RL model.")
    args = parser.parse_args()
    
    run_evaluation(args.mode, args.model_path)