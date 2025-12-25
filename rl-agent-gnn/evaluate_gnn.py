import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from gnn_environment import ReplicationEnvGNN
from gnn_model import ReplicationGNN
from ray import tune
import numpy as np
import time
import json
import os


CHECKPOINT_PATH = os.path.abspath("./manual_checkpoints")
EVAL_DURATION_MINUTES = 60

def run_evaluation():
    ray.init(ignore_reinit_error=True)

    tune.register_env("replication_gnn_env", lambda config: ReplicationEnvGNN(config))
    ModelCatalog.register_custom_model("replication_gnn_model", ReplicationGNN)

    print(f"Loading agent from: {CHECKPOINT_PATH}")
    agent = Algorithm.from_checkpoint(CHECKPOINT_PATH)

    env = ReplicationEnvGNN()
    obs, info = env.reset()

    results = []
    start_time = time.time()
    
    print("--- Starting GNN Evaluation ---")

    while time.time() - start_time < EVAL_DURATION_MINUTES * 60:
        # Ask Agent for Action
        # explore=False makes it deterministic (best action only)
        action = agent.compute_single_action(obs, explore=False)
        
        # Execute in Env
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Log Metrics (Reusing logic from previous eval scripts)
        # We fetch state directly to calculate metrics for the plot
        state_json = env._fetch_state()
        
        total_cost = sum(n.get('storageCost', 0) for n in state_json)
        # Calculate Latency (Simplified approximation for plotting)
        total_reads = 0
        lat_sum = 0
        presence = {}
        for n in state_json:
            for k in n.get('keyMetrics', {}):
                if k not in presence: presence[k] = set()
                presence[k].add(n['nodeId'])
        for n in state_json:
            nid = n['nodeId']
            for k, m in n.get('keyMetrics', {}).items():
                r = m.get('readCount', 0)
                total_reads += r
                is_local = nid in presence.get(k, set())
                lat_sum += r * (10 if is_local else 150)
        
        avg_lat = (lat_sum / total_reads) if total_reads > 0 else 0
        
        elapsed = time.time() - start_time
        print(f"Time: {int(elapsed)}s | Latency: {avg_lat:.1f}ms | Cost: ${total_cost:.1f} | Action: {action}")

        results.append({
            "time": elapsed,
            "avg_latency": avg_lat,
            "total_cost": total_cost
        })
        
        time.sleep(1)

    with open("evaluation_results_gnn.json", "w") as f:
        json.dump(results, f)
    print("Evaluation Complete. Results saved.")

if __name__ == "__main__":
    run_evaluation()