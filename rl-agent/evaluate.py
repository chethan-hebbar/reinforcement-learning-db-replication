import requests
import time
import json
import argparse
import numpy as np
from sb3_contrib import MaskablePPO 

CONTROLLER_URL = "http://localhost:8080"
EVALUATION_DURATION_MINS = 60

# Synchronized frequency with GNN (1 second)
POLLING_INTERVAL_SECS = 1
DECISION_INTERVAL_SECS = 1

NUM_KEYS = 20
NODE_PREFIXES = ['us', 'eu', 'ap', 'sa', 'jp']
NUM_NODES = len(NODE_PREFIXES)


def get_system_state():
    """Fetches the current state of the entire cluster from the controller."""
    try:
        response = requests.get(f"{CONTROLLER_URL}/rl/system-state", timeout=2)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not get system state: {e}")
        return None

def parse_state_to_observation(state_json):
    """
    Converts the JSON state into the NumPy vector.
    """
    total_size = NUM_KEYS * NUM_NODES * 3
    if not state_json:
        return np.zeros(total_size, dtype=np.float32)

    # Define consistent ordering
    node_order = [f"replication-{r}" for r in NODE_PREFIXES]
    key_order = [f"user_profile_{i}" for i in range(NUM_KEYS)]

    presence = np.zeros((NUM_NODES, NUM_KEYS), dtype=np.float32)
    read_counts = np.zeros((NUM_NODES, NUM_KEYS), dtype=np.float32)
    write_counts = np.zeros((NUM_NODES, NUM_KEYS), dtype=np.float32)

    # Map API Node IDs to our index
    node_map = {}
    for node_data in state_json:
        node_id = node_data['nodeId']
        for target in node_order:
            if target == node_id:
                node_map[target] = node_data
                break

    for i, node_name in enumerate(node_order):
        node_data = node_map.get(node_name)
        if not node_data: continue

        for key_name, metrics in node_data.get('keyMetrics', {}).items():
            if key_name in key_order:
                key_idx = key_order.index(key_name)
                
                presence[i, key_idx] = 1.0
                read_counts[i, key_idx] = np.log1p(metrics.get('readCount', 0))
                write_counts[i, key_idx] = np.log1p(metrics.get('writeCount', 0))
    
    return np.concatenate([
        presence.flatten(),
        read_counts.flatten(),
        write_counts.flatten()
    ])

def get_action_mask(state_json):
    """
    Reconstructs the validity mask so MaskablePPO doesn't pick invalid actions.
    """
    limit = NUM_KEYS * NUM_NODES
    mask = np.zeros(limit * 2, dtype=bool)
    
    node_order = [f"replication-{r}" for r in NODE_PREFIXES]
    key_order = [f"user_profile_{i}" for i in range(NUM_KEYS)]
    
    # Identify what exists where
    presence_map = set()
    for i, node_name in enumerate(node_order):
        node_data = next((n for n in state_json if n['nodeId'] == node_name), None)
        if node_data:
            for key_name in node_data.get('keyMetrics', {}).keys():
                if key_name in key_order:
                    k_idx = key_order.index(key_name)
                    presence_map.add((k_idx, i))

    for k in range(NUM_KEYS):
        for n in range(NUM_NODES):
            flat_idx = (k * NUM_NODES) + n
            if (k, n) in presence_map:
                # Exists -> Can Evict (Index + Limit)
                mask[limit + flat_idx] = True
            else:
                # Not Exists -> Can Replicate (Index)
                mask[flat_idx] = True
                
    return mask

def decode_action(action_id):
    """Converts an integer action back into a command."""
    limit = NUM_KEYS * NUM_NODES
    is_evict = action_id >= limit
    
    if is_evict:
        action_id -= limit
    
    key_id = action_id // NUM_NODES
    node_id = action_id % NUM_NODES
    
    action_type = "EVICT" if is_evict else "REPLICATE"
    
    key_name = f"user_profile_{key_id}"
    node_name = f"replication-{NODE_PREFIXES[node_id]}"
    
    return action_type, key_name, node_name

def execute_action(action_type, key, node):
    """Sends the chosen action to the controller."""
    payload = {"actionType": action_type, "key": key, "targetNode": node}
    try:
        requests.post(f"{CONTROLLER_URL}/rl/execute-action", json=payload, timeout=1)
        print(f"ACTION: {action_type} {key} on {node}")
    except requests.exceptions.RequestException:
        pass 

def calculate_system_metrics(state_json):
    """Calculates aggregate latency and cost from the system state."""
    if not state_json:
        return 0, 0

    total_storage_cost = sum(node.get('storageCost', 0) for node in state_json)
    
    total_reads = 0
    predicted_latency_sum = 0
    presence_map = {}
    
    for node_data in state_json:
        node_id = node_data['nodeId']
        for key_name in node_data.get('keyMetrics', {}).keys():
            if key_name not in presence_map:
                presence_map[key_name] = set()
            presence_map[key_name].add(node_id)

    for node_data in state_json:
        requesting_node_id = node_data['nodeId']
        for key_name, metrics in node_data.get('keyMetrics', {}).items():
            reads = metrics.get('readCount', 0)
            total_reads += reads
            is_local = requesting_node_id in presence_map.get(key_name, set())
            latency = 10 if is_local else 150 
            predicted_latency_sum += reads * latency

    avg_latency = (predicted_latency_sum / total_reads) if total_reads > 0 else 0
    return avg_latency, total_storage_cost



def run_evaluation(mode, model_path=None):
    print(f"--- Starting Evaluation in '{mode.upper()}' Mode (5 Nodes / 20 Keys) ---")
    
    model = None
    if mode == 'rl':
        if not model_path:
            print("ERROR: Must provide --model_path for 'rl' mode.")
            return
        print(f"Loading trained model from {model_path}...")
        try:
            model = MaskablePPO.load(model_path)
        except Exception as e:
            print(f"ERROR loading model: {e}")
            return

    results = []
    start_time = time.time()
    last_decision_time = 0

    while time.time() - start_time < EVALUATION_DURATION_MINS * 60:
        loop_start = time.time()
        state_json = get_system_state()
        
        # RL Agent Decision
        if mode == 'rl' and state_json:
            if loop_start - last_decision_time >= DECISION_INTERVAL_SECS:
                last_decision_time = loop_start
                
                observation = parse_state_to_observation(state_json)
                
                # --- Generate Mask for Prediction ---
                # This ensures the agent doesn't try to evict keys that don't exist
                # or replicate keys that are already there.
                action_masks = get_action_mask(state_json)
                
                action, _ = model.predict(observation, action_masks=action_masks, deterministic=True)
                action_type, key, node = decode_action(action.item())
                execute_action(action_type, key, node)

        # Metrics Collection
        avg_latency, total_cost = calculate_system_metrics(state_json)
        
        elapsed_time = loop_start - start_time
        print(f"Time: {int(elapsed_time)}s, Avg Latency: {avg_latency:.2f}ms, Total Cost: ${total_cost:.2f}")
        
        results.append({
            "time": elapsed_time,
            "avg_latency": avg_latency,
            "total_cost": total_cost
        })

        time.sleep(POLLING_INTERVAL_SECS)

    # Save results
    output_filename = f"evaluation_results_{mode}_20keys.json"
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"--- Evaluation Finished. Results saved to {output_filename} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=['static', 'rl'])
    # Default to the masked model name you used
    parser.add_argument("--model_path", type=str, default="ppo_replication_policy.zip")
    args = parser.parse_args()
    
    run_evaluation(args.mode, args.model_path)