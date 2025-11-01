import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from replication_env import ReplicationEnv

# --- Optional: A Custom Callback for Logging ---
# This is useful to see the reward at each step during training.
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # self.locals is a dictionary with all the variables from the training loop
        reward = self.locals['rewards'][0]
        self.logger.record('custom/step_reward', reward)
        return True

# --- Main Training Block ---
if __name__ == "__main__":
    print("--- Starting Reinforcement Learning Training ---")

    # 1. Instantiate the custom environment
    # `make_vec_env` is a helper that wraps the environment, which is good practice
    env = make_vec_env(ReplicationEnv, n_envs=1)

    # 2. Define the RL model
    # We will use the Proximal Policy Optimization (PPO) algorithm.
    # "MlpPolicy" means it will use a standard Multi-Layer Perceptron (a neural network) as its brain.
    # We've added a learning_rate and a tensorboard_log for monitoring.
    model = PPO("MlpPolicy",
                env,
                verbose=1, # Prints out training progress
                learning_rate=0.0005,
                tensorboard_log="./ppo_replication_tensorboard/")

    # 3. Start the training process
    # The agent will interact with the environment for 10,000 "timesteps".
    # A timestep is one full cycle: get state -> agent predicts action -> env executes action -> get new state/reward.
    training_timesteps = 25_000
    print(f"Starting training for {training_timesteps} timesteps...")
    start_time = time.time()
    
    # The learn() method is where all the magic happens.
    model.learn(total_timesteps=training_timesteps, callback=RewardLoggerCallback())
    
    end_time = time.time()
    print(f"--- Training Finished ---")
    print(f"Total training time: {end_time - start_time:.2f} seconds")

    # 4. Save the trained agent's policy (its "brain") to a file
    model_path = "ppo_replication_policy.zip"
    model.save(model_path)

    print(f"Trained model saved to: {model_path}")
    print("\nTo view training logs, run the following command in your terminal:")
    print(f"tensorboard --logdir ./ppo_replication_tensorboard/")