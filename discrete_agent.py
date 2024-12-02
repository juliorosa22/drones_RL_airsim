from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from airgym.envs.quad_drone_env import QuadAirSimDroneEnv


# Initialize the environment with the target position
env = QuadAirSimDroneEnv(ip_address="127.0.0.1", img_shape=(64, 64, 1), path_file='training_paths.json',csv_file_log='dqn_episodes.csv')

# Ensure the environment meets the requirements
check_env(env)




# Set up callbacks
# Checkpoint callback to save model periodically
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/checkpoints/',
                                         name_prefix='dqn_model_checkpoint')

# Evaluation callback to monitor training performance
#eval_callback = EvalCallback(env, best_model_save_path="./logs/best_model",
#                             log_path="./logs/eval_results", eval_freq=100, n_eval_episodes=10, deterministic=True)

# Create a DQN model with MultiInputPolicy and set up TensorBoard
#model = DQN("MlpPolicy", env, batch_size=64, learning_rate=1e-3, verbose=1, tensorboard_log="./logs/")
model=PPO.load("pretrained_PPO_agent", env, batch_size=128, learning_rate=1e-3, verbose=1, tensorboard_log="./logs/")
# Train the model with specified callbacks
model.learn(total_timesteps=2e5, log_interval=500, tb_log_name="ppo_training_log",
            callback=[checkpoint_callback], progress_bar=True)

# Save the final model after training completes
model.save("ppo_drone_agent")
