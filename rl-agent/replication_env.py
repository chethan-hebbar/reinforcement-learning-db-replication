import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests

# --- Constants ---
NUM_NODES = 3
NUM_KEYS = 5
CONTROLLER_URL = "http://localhost:8080"
LATENCY_WEIGHT = 0.4
COST_WEIGHT = 0.6

class ReplicationEnv(gym.Env):
    """Custom Environment for Adaptive Data Replication."""
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(ReplicationEnv, self).__init__()

        # === DEFINE ACTION SPACE ===
        # Total actions = (replicate key K to node N) + (evict key K from node N)
        # N_actions = (NUM_KEYS * NUM_NODES) + (NUM_KEYS * NUM_NODES)
        self.action_space = spaces.Discrete(NUM_KEYS * NUM_NODES * 2)

        # === DEFINE STATE (OBSERVATION) SPACE ===
        # State vector: [presence_matrix, read_counts, write_counts] flattened
        # Size = 3 * (NUM_KEYS * NUM_NODES)
        state_size = NUM_KEYS * NUM_NODES * 3
        # We use low=0 and high=np.inf because counts can grow indefinitely.
        # We also use np.float32 as recommended by Stable-Baselines3.
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(state_size,), dtype=np.float32)

        print("ReplicationEnv initialized.")
        print(f"Action Space: {self.action_space.n} discrete actions")
        print(f"Observation Space Shape: {self.observation_space.shape}")
    
    # We will also create a helper to decode actions
    def _decode_action(self, action_id):
        is_evict = action_id >= (NUM_KEYS * NUM_NODES)
        if is_evict:
            action_id -= (NUM_KEYS * NUM_NODES)
        
        key_id = action_id // NUM_NODES
        node_id = action_id % NUM_NODES
        
        action_type = "EVICT" if is_evict else "REPLICATE"
        # We will need a consistent way to map node_id to a real URL/name
        # For now, we use placeholders.
        key_name = f"user_profile_{key_id}"
        node_name = f"replication-{['us', 'eu', 'ap'][node_id]}"
        return action_type, key_name, node_name
    
    def _get_system_state(self):
        try:
            response = requests.get(f"{CONTROLLER_URL}/rl/system-state", timeout=5)
            response.raise_for_status() # Raise an exception for bad status codes
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching system state: {e}")
            return None

    def _parse_state_to_observation(self, state_json):
        """Converts the JSON state response into a flattened NumPy array."""
        # Define the order of nodes and keys for consistency
        node_order = [f"replication-{r}" for r in ['us', 'eu', 'ap']]
        key_order = [f"user_profile_{i}" for i in range(NUM_KEYS)]

        # Initialize arrays for each part of the state vector
        presence = np.zeros((NUM_NODES, NUM_KEYS), dtype=np.float32)
        read_counts = np.zeros((NUM_NODES, NUM_KEYS), dtype=np.float32)
        write_counts = np.zeros((NUM_NODES, NUM_KEYS), dtype=np.float32)

        # Create a map for quick lookup: 'us-east-1' -> 0, 'eu-west-1' -> 1, etc.
        # This is a bit fragile, depends on container names. A better way would be
        # to have the node ID in the response match our expected order.
        # Let's assume the order from the API is us, eu, ap for now.
        node_id_map = {node_data['nodeId'].split('-')[0]: i for i, node_data in enumerate(state_json)}
    
        for i, node_data in enumerate(state_json):
            # Assuming the order is consistent, otherwise use a map
            node_idx = i 
            for key_name, metrics in node_data.get('keyMetrics', {}).items():
                if key_name in key_order:
                    key_idx = key_order.index(key_name)
                
                    # Presence: if a key has metrics, it is present.
                    presence[node_idx, key_idx] = 1.0
                    read_counts[node_idx, key_idx] = metrics.get('readCount', 0)
                    write_counts[node_idx, key_idx] = metrics.get('writeCount', 0)
    
        # Flatten the arrays and concatenate them to form the final state vector
        return np.concatenate([
            presence.flatten(),
            read_counts.flatten(),
            write_counts.flatten()
        ])


    def _execute_action(self, action_type, key, node):
        """Makes an API call to the controller to execute a replication action."""
        payload = {
            "actionType": action_type,
            "key": key,
            "targetNode": node
        }
        try:
            response = requests.post(f"{CONTROLLER_URL}/rl/execute-action", json=payload, timeout=5)
            response.raise_for_status()
            return True # Success
        except requests.exceptions.RequestException as e:
            print(f"Error executing action: {e}")
            return False # Failure

    def _calculate_reward(self, state_json):
        """Calculates the reward based on the current system state."""
        if not state_json:
            return -1000 # Heavy penalty if the system is down

        total_storage_cost = 0
        total_reads = 0
        predicted_latency_sum = 0
        
        # We need to know which keys are on which nodes for the latency calculation
        presence_map = {} # e.g., {'key_0': {'node_0', 'node_1'}, ...}
        for i, node_data in enumerate(state_json):
            node_name = f"node_{i}" # This is a placeholder
            total_storage_cost += node_data.get('storageCost', 0)
            for key_name in node_data.get('keyMetrics', {}).keys():
                if key_name not in presence_map:
                    presence_map[key_name] = set()
                presence_map[key_name].add(node_name)

        # Now calculate the predicted latency
        for i, node_data in enumerate(state_json):
            node_name = f"node_{i}"
            for key_name, metrics in node_data.get('keyMetrics', {}).items():
                reads = metrics.get('readCount', 0)
                total_reads += reads
                
                # Is the key present locally on the node that is getting the reads?
                is_local_read = key_name in presence_map and node_name in presence_map[key_name]
                
                if is_local_read:
                    predicted_latency_sum += reads * 10 # 10ms for local
                else:
                    predicted_latency_sum += reads * 150 # 150ms for remote

        avg_predicted_latency = (predicted_latency_sum / total_reads) if total_reads > 0 else 0
        
        # The reward is negative because we want to MINIMIZE cost and latency.
        # The agent's goal is to maximize reward, so it will learn to make this value less negative.
        reward = -1 * ( (LATENCY_WEIGHT * avg_predicted_latency) + (COST_WEIGHT * total_storage_cost) )
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Recommended by Gymnasium
        print("Resetting environment...")
    
        state_json = self._get_system_state()
        if state_json is None:
        # If the controller is down, return a zero state and handle it in the training loop
            print("Failed to get initial state from controller.")
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            observation = self._parse_state_to_observation(state_json)
    
        return observation, {}

    def step(self, action):
        # 1. Decode the integer action into a meaningful command
        action_type, key, node = self._decode_action(action)
        print(f"Step: Executing action -> {action_type} {key} on {node}")

        # 2. Execute the action by calling the controller
        success = self._execute_action(action_type, key, node)
        if not success:
            # If the action fails, penalize heavily and end the episode
            reward = -1000
            new_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return new_obs, reward, True, False, {"error": "Action execution failed"}

        # 3. Get the new state of the system AFTER the action
        # In a real system, we would wait for the workload to run for a bit.
        # Here, we get the state immediately to see the result of our action.
        new_state_json = self._get_system_state()
        new_observation = self._parse_state_to_observation(new_state_json)

        # 4. Calculate the reward based on the new state
        reward = self._calculate_reward(new_state_json)
        print(f"Step: New State Reward -> {reward}")

        # For this environment, an episode doesn't have a natural "end" (terminated=True).
        # We will let the training loop decide when to stop.
        terminated = False
        truncated = False
        info = {}

        return new_observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass