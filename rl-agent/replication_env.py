import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests

NUM_NODES = 5
NUM_KEYS = 20
CONTROLLER_URL = "http://localhost:8080"

# Match your Docker Compose service names
NODE_PREFIXES = ['us', 'eu', 'ap', 'sa', 'jp'] 

# Reward Weights (Matching your GNN config for fair comparison)
LATENCY_WEIGHT = 0.1
COST_WEIGHT = 0.9

class ReplicationEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(ReplicationEnv, self).__init__()

        # Total actions = (replicate + evict) for every (key * node) combo
        # Action space size = 20 * 5 * 2 = 200
        self.action_space = spaces.Discrete(NUM_KEYS * NUM_NODES * 2)

        # State vector: [presence_matrix, read_counts, write_counts]
        # Size = 3 * (20 * 5) = 300 inputs
        state_size = NUM_KEYS * NUM_NODES * 3
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(state_size,), dtype=np.float32)

        print(f"ReplicationEnv initialized. State Size: {state_size}, Action Size: {self.action_space.n}")

    def _decode_action(self, action_id):
        # Logic: 
        # 0..99   = REPLICATE (Keys 0-19 on Node 0, then Node 1...)
        # 100..199 = EVICT
        
        limit = NUM_KEYS * NUM_NODES
        is_evict = action_id >= limit
        
        if is_evict:
            action_id -= limit
        
        # Calculate indices
        key_id = action_id // NUM_NODES
        node_id = action_id % NUM_NODES
        
        action_type = "EVICT" if is_evict else "REPLICATE"
        
        key_name = f"user_profile_{key_id}"
        node_name = f"replication-{NODE_PREFIXES[node_id]}"
        
        return action_type, key_name, node_name

    def _get_system_state(self):
        try:
            response = requests.get(f"{CONTROLLER_URL}/rl/system-state", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching system state: {e}")
            return None

    def _parse_state_to_observation(self, state_json):
        # We must enforce a consistent order for the vector
        node_order = [f"replication-{r}" for r in NODE_PREFIXES]
        key_order = [f"user_profile_{i}" for i in range(NUM_KEYS)]

        presence = np.zeros((NUM_NODES, NUM_KEYS), dtype=np.float32)
        read_counts = np.zeros((NUM_NODES, NUM_KEYS), dtype=np.float32)
        write_counts = np.zeros((NUM_NODES, NUM_KEYS), dtype=np.float32)

        # Map node IDs from JSON to our index (0-4)
        # Handle exact match or prefix match
        node_map = {}
        for i, target_name in enumerate(node_order):
            for node_data in state_json:
                # Check if "replication-us" is in "replication-us" (Docker) or "us-east-1" (App)
                if node_data['nodeId'] == target_name:
                    node_map[target_name] = node_data
                    break

        for i, node_name in enumerate(node_order):
            node_data = node_map.get(node_name)
            if not node_data: continue

            for key_name, metrics in node_data.get('keyMetrics', {}).items():
                if key_name in key_order:
                    key_idx = key_order.index(key_name)
                    
                    presence[i, key_idx] = 1.0
                    # Apply log normalization to match GNN (optional but recommended)
                    read_counts[i, key_idx] = np.log1p(metrics.get('readCount', 0))
                    write_counts[i, key_idx] = np.log1p(metrics.get('writeCount', 0))
        
        return np.concatenate([
            presence.flatten(),
            read_counts.flatten(),
            write_counts.flatten()
        ])

    def _execute_action(self, action_type, key, node):
        payload = {"actionType": action_type, "key": key, "targetNode": node}
        try:
            requests.post(f"{CONTROLLER_URL}/rl/execute-action", json=payload, timeout=1)
            return True
        except: return False

    def _calculate_reward(self, state_json):
        if not state_json: return -100.0

        total_storage_cost = 0
        total_reads = 0
        predicted_latency_sum = 0
        
        presence_map = {}
        for node_data in state_json:
            node_id = node_data['nodeId']
            total_storage_cost += node_data.get('storageCost', 0)
            for k in node_data.get('keyMetrics', {}).keys():
                if k not in presence_map: presence_map[k] = set()
                presence_map[k].add(node_id)

        for node_data in state_json:
            req_node = node_data['nodeId']
            for k, m in node_data.get('keyMetrics', {}).items():
                reads = m.get('readCount', 0)
                total_reads += reads
                is_local = req_node in presence_map.get(k, set())
                predicted_latency_sum += reads * (10 if is_local else 150)

        avg_lat = (predicted_latency_sum / total_reads) if total_reads > 0 else 0
        
        # Scale reward down slightly to prevent huge numbers with 20 keys
        reward = -1 * ((LATENCY_WEIGHT * avg_lat) + (COST_WEIGHT * total_storage_cost))
        return reward / 20.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state_json = self._get_system_state()
        if state_json is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}
        return self._parse_state_to_observation(state_json), {}

    def step(self, action):
        action_type, key, node = self._decode_action(action)
        
        # Execute
        self._execute_action(action_type, key, node)
        
        # New State
        new_state = self._get_system_state()
        obs = self._parse_state_to_observation(new_state)
        reward = self._calculate_reward(new_state)
        
        return obs, reward, False, False, {}
    
    def action_masks(self):
        # Create a boolean mask of valid actions
        # Action 0..(N*K-1) = REPLICATE
        # Action (N*K)..(2*N*K-1) = EVICT
        
        # This requires fetching state, which is slow, but necessary for Masking.
        # Ideally, cache the state from the last step() call.
        state_json = self._get_system_state() 
        
        presence_map = set()
        
        node_order = [f"replication-{r}" for r in NODE_PREFIXES]
        key_order = [f"user_profile_{i}" for i in range(NUM_KEYS)]
        
        for i, node_name in enumerate(node_order):
            # Find node data
            node_data = next((n for n in state_json if n['nodeId'] == node_name), None)
            if node_data:
                for key_name in node_data.get('keyMetrics', {}).keys():
                    if key_name in key_order:
                        k_idx = key_order.index(key_name)
                        presence_map.add((k_idx, i))

        mask = [False] * (NUM_KEYS * NUM_NODES * 2)
        
        limit = NUM_KEYS * NUM_NODES
        
        for k in range(NUM_KEYS):
            for n in range(NUM_NODES):
                flat_idx = (k * NUM_NODES) + n
                
                is_present = (k, n) in presence_map
                
                if is_present:
                    # If present, we can EVICT (Action index + limit)
                    mask[limit + flat_idx] = True
                    # We CANNOT Replicate (already there)
                else:
                    # If not present, we can REPLICATE (Action index)
                    mask[flat_idx] = True
                    # We CANNOT Evict (not there)
                    
        return mask