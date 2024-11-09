from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
#from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import numpy as np
from airgym.envs.drone_env import AirSimDroneEnv

# Initialize the environment with the target position
env = AirSimDroneEnv(ip_address="127.0.0.1",img_shape=(64,64,1),start_position=[0,0,-10],target_position=[-15, 30, -10])

# Add noise for exploration in DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

# Set up callbacks
#eval_callback = EvalCallback(env, best_model_save_path="./logs/best_model",
#                            log_path="./logs/results", eval_freq=1000, n_eval_episodes=5)
#checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./logs/',
#                                         name_prefix='ddpg_model_checkpoint')

# Create DDPG model
model = DDPG("MultiInputPolicy", env, action_noise=action_noise,batch_size=128,learning_rate=2e-3,verbose=1)

# Train the model
model.learn(total_timesteps=100000,log_interval=1000,tb_log_name="ddpg_airsim_run",progress_bar=True)

# Save the model
model.save("ddpg_airsim_drone")
