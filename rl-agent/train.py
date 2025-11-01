import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from replication_env import ReplicationEnv

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        self.logger.record('custom/step_reward', reward)
        return True

if __name__ == "__main__":
    print("--- Starting Reinforcement Learning Training ---")

    env = make_vec_env(ReplicationEnv, n_envs=1)

    model = PPO("MlpPolicy",
                env,
                verbose=1, # Prints out training progress
                learning_rate=0.0005,
                tensorboard_log="./ppo_replication_tensorboard/")

    training_timesteps = 25_000
    print(f"Starting training for {training_timesteps} timesteps...")
    start_time = time.time()
    
    model.learn(total_timesteps=training_timesteps, callback=RewardLoggerCallback())
    
    end_time = time.time()
    print(f"--- Training Finished ---")
    print(f"Total training time: {end_time - start_time:.2f} seconds")

    model_path = "ppo_replication_policy.zip"
    model.save(model_path)

    print(f"Trained model saved to: {model_path}")
    print("\nTo view training logs, run the following command in your terminal:")
    print(f"tensorboard --logdir ./ppo_replication_tensorboard/")