import os
import shutil
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from gnn_environment import ReplicationEnvGNN
from gnn_model import ReplicationGNN
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

def train_manual():
    ray.init(ignore_reinit_error=True)
    register_env("replication_gnn_env", lambda config: ReplicationEnvGNN(config))
    ModelCatalog.register_custom_model("replication_gnn_model", ReplicationGNN)

    print("Building PPO Configuration...")
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment("replication_gnn_env")
        .framework("torch")
        .training(
            model={
                "custom_model": "replication_gnn_model",
            },
            train_batch_size=4800,
            minibatch_size=256,
            lr=0.01,
            grad_clip=0.5,
            vf_clip_param=50.0,
            entropy_coeff=0.05,
        )
        .env_runners(num_env_runners=0)
        .resources(num_gpus=0)
    )

    # Build the Algorithm Instance
    # This actually creates the PPO object. No "Tuner" involved.
    algo = config.build()
    print("Algorithm built successfully.")

    # Manual Training Loop
    # We control exactly what happens step-by-step
    num_iterations = 15
    best_reward = -float('inf')
    
    # Directory to save checkpoints
    save_dir = os.path.abspath("./manual_checkpoints")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    print("\n--- Starting Training Loop ---")

    for i in range(num_iterations):
        result = algo.train()
        
        # Ray nesting varies by version, safe get:
        if 'env_runners' in result:
            reward = result['env_runners']['episode_reward_mean']
        else:
            reward = result['episode_reward_mean']
        
        print(f"Iter: {i+1:03d} | Reward: {reward}")

        # Checkpoint Logic
        if reward is not None and not isinstance(reward, str) and reward > best_reward:
            best_reward = reward
            print(f"   --> New Best Reward! Saving Checkpoint...")
            save_obj = algo.save(save_dir)
            
            # Extract the actual path string depending on what Ray returned
            if isinstance(save_obj, str):
                save_path = save_obj
            elif hasattr(save_obj, "checkpoint") and save_obj.checkpoint:
                save_path = save_obj.checkpoint.path
            elif hasattr(save_obj, "path"):
                save_path = save_obj.path
            else:
                save_path = str(save_obj)

            print(f"   --> Saved to: {save_path}")
            
            with open("best_checkpoint_path.txt", "w") as f:
                f.write(save_path)
    
    print("\nTraining Finished.")
    algo.stop()

if __name__ == "__main__":
    train_manual()