import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests
import time
from graph_utils import parse_system_state_to_graph

CONTROLLER_URL = "http://localhost:8080"

LATENCY_WEIGHT = 0.1
COST_WEIGHT = 0.9

MAX_KEYS = 25
MAX_SERVERS = 10
MAX_EDGES = MAX_KEYS * MAX_SERVERS

class ReplicationEnvGNN(gym.Env):
    def __init__(self, config=None):
        self.action_space = spaces.Discrete(MAX_KEYS * MAX_SERVERS)
        
        self.observation_space = spaces.Dict({
            "x_keys": spaces.Box(-np.inf, np.inf, shape=(MAX_KEYS, 3), dtype=np.float32),
            "x_servers": spaces.Box(-np.inf, np.inf, shape=(MAX_SERVERS, 2), dtype=np.float32),
            "edge_index": spaces.Box(-1, max(MAX_KEYS, MAX_SERVERS), shape=(2, MAX_EDGES), dtype=np.int64),
            "edge_attr": spaces.Box(-np.inf, np.inf, shape=(MAX_EDGES, 2), dtype=np.float32),
            "real_counts": spaces.Box(0, MAX_EDGES, shape=(3,), dtype=np.int32), 
            "action_mask": spaces.Box(0, 1, shape=(MAX_KEYS * MAX_SERVERS,), dtype=np.float32)
        })
        
        self.current_key_names = []
        self.current_server_ids = [
            'replication-us', 
            'replication-eu', 
            'replication-ap',
            'replication-sa', 
            'replication-jp'
        ]
        
        self.steps = 0
        self.max_steps = 200

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        while True:
            state_json = self._fetch_state()
            has_keys = any(len(n.get('keyMetrics', {})) > 0 for n in state_json)
            if has_keys: break
            time.sleep(0.2)
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        truncated = (self.steps >= self.max_steps)
        num_servers = len(self.current_server_ids)
        key_idx = int(action) // num_servers
        server_idx = int(action) % num_servers

        if key_idx >= len(self.current_key_names):
            # Invalid action penalty
            #print(f"[AGENT] Action {action}: INVALID KEY INDEX {key_idx} (Max {len(self.current_key_names)})")
            return self._get_obs(), -20.0, False, truncated, {}
        
        k_name = self.current_key_names[key_idx]
        s_name = self.current_server_ids[server_idx]

        target_key = self.current_key_names[key_idx]
        target_node = self.current_server_ids[server_idx]

        # Determine Action Type
        state_json = self._fetch_state()
        exists = False
        for node in state_json:
            if target_node in node['nodeId'] or node['nodeId'] in target_node:
                if target_key in node.get('keyMetrics', {}):
                    exists = True
                    break
        
        action_type = "EVICT" if exists else "REPLICATE"
        #print(f"[DEBUG] {target_key} on {target_node} Exists? {exists} -> Action: {action_type}")
        
        try:
            resp =requests.post(f"{CONTROLLER_URL}/rl/execute-action", 
                          json={"actionType": action_type, "key": target_key, "targetNode": target_node}, 
                          timeout=1)
            time.sleep(0.01) 
            
        except Exception as e:
            print(f"API ERROR: {e}")

        obs = self._get_obs()
        
        # Calculate Scaled Reward
        # Raw reward is usually around -16.0 (Safe) to -10.0 (Optimized)
        # We scale it down to keep gradients stable.
        reward_raw = self._calculate_reward(self._last_state_json)
        reward_scaled = reward_raw / 20.0 
        
        return obs, reward_scaled, False, truncated, {}

    def _fetch_state(self):
        try:
            return requests.get(f"{CONTROLLER_URL}/rl/system-state", timeout=2).json()
        except: return []

    def _get_obs(self):
        state_json = self._fetch_state()
        self._last_state_json = state_json
        
        x_k, x_s, e_i, e_a, k_names = parse_system_state_to_graph(state_json)
        self.current_key_names = k_names
        
        nk, ns = x_k.shape[0], x_s.shape[0]
        ne = e_i.shape[1]

        obs = {
            "x_keys": np.pad(x_k, ((0, MAX_KEYS - nk), (0,0))),
            "x_servers": np.pad(x_s, ((0, MAX_SERVERS - ns), (0,0))),
            "edge_index": np.pad(e_i, ((0,0), (0, MAX_EDGES - ne)), constant_values=-1),
            "edge_attr": np.pad(e_a, ((0, MAX_EDGES - ne), (0,0))),
            "real_counts": np.array([nk, ns, ne], dtype=np.int32),
            "action_mask": np.zeros(MAX_KEYS * MAX_SERVERS, dtype=np.float32)
        }
        
        if nk > 0 and ns > 0:
            obs["action_mask"][:nk * ns] = 1.0
        else:
            obs["action_mask"][0] = 1.0 
        
        return obs

    def _calculate_reward(self, state_json):
        if not state_json: return -100.0
        
        total_cost = sum(n.get('storageCost', 0) for n in state_json)
        total_reads = 0
        latency_sum = 0
        
        presence_map = {}
        for n in state_json:
            for k in n.get('keyMetrics', {}):
                if k not in presence_map: presence_map[k] = set()
                presence_map[k].add(n['nodeId'])
                
        for n in state_json:
            node_id = n['nodeId']
            for k, m in n.get('keyMetrics', {}).items():
                reads = m.get('readCount', 0)
                total_reads += reads
                is_local = node_id in presence_map.get(k, set())
                latency_sum += reads * (10 if is_local else 150)
                
        avg_lat = (latency_sum / total_reads) if total_reads > 0 else 0

        # Only print every ~50 steps to avoid spamming too much
        if self.steps % 200 == 0:
            print(f"\n[ENV DEBUG] Latency: {avg_lat:.2f} | Cost: {total_cost:.2f} | Reads: {total_reads}")
        
        # Using Cost-Conscious weights to match best MLP result
        reward = -1 * ((LATENCY_WEIGHT * avg_lat) + (COST_WEIGHT * total_cost))
        
        if np.isnan(reward) or np.isinf(reward): return -100.0
        return reward