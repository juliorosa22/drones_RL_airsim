from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import numpy as np
from airgym.envs.ddpg_drone_env import DDPGAirSimDroneEnv
from stable_baselines3.common.env_checker import check_env

# Initialize the environment with the target position
env = DDPGAirSimDroneEnv(ip_address="127.0.0.1", img_shape=(64, 64, 1), start_position=[-20, 0 , -5], target_position=[-45, 0, -5])
check_env(env)
# Add noise for exploration in DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Set up callbacks
# Checkpoint callback to save model periodically
checkpoint_callback = CheckpointCallback(save_freq=100, save_path='./logs/checkpoints/',
                                         name_prefix='new_ddpg_model_checkpoint')

# Evaluation callback to monitor training performance
eval_callback = EvalCallback(env, best_model_save_path="./logs/best_model",
                             log_path="./logs/results", eval_freq=100, n_eval_episodes=5, deterministic=True)

# Create DDPG model with MultiInputPolicy, and set up TensorBoard for real-time monitoring
model = DDPG("MultiInputPolicy", env, action_noise=action_noise, batch_size=64,
             learning_rate=1e-3, verbose=1, tensorboard_log="./logs/")

# Train the model with specified callbacks
model.learn(total_timesteps=50000, log_interval=100, tb_log_name="new_ddpg_airsim_run",
            callback=[checkpoint_callback, eval_callback], progress_bar=True)

# Save the final model after training completes
model.save("ddpg_airsim_drone")
