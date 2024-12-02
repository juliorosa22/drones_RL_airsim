from stable_baselines3 import DQN, PPO
from generate_expert_data import ExpertDataCollector
import gymnasium as gym
from gymnasium import spaces
import numpy as np
LIDAR_FEAT_SIZE=300
class ExpertImitationEnv(gym.Env):
    def __init__(self, observations, actions):
        super(ExpertImitationEnv, self).__init__()
        self.observations = observations
        self.actions = actions
        self.current_index = 0
        self.max_velocity=10
        # Define observation and action space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observations.shape[1],),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(7)  # Number of discrete actions

    def reset(self, **kwargs):
       
        self.current_index = 0
        return self.observations[self.current_index],{}

    def parse_flattened_obs(self,flattened_obs):
            """
            Convert a flattened observation array back into a dictionary.

            Args:
                flattened_obs (np.ndarray): Flattened observation array.

            Returns:
                dict: Parsed observation with keys for drone_position, orientation, linear_velocity, and relative_yaw.
            """
            # Indices for splitting the flattened observation
            position_end = 3  # First 3 elements are drone position
            orientation_end = position_end + 3  # Next 3 elements are orientation (roll, pitch, yaw)
            velocity_end = orientation_end + 3  # Next 3 elements are linear velocity 
            # Parse data
            drone_position = flattened_obs[:position_end]
            orientation = flattened_obs[position_end:orientation_end]
            linear_velocity = flattened_obs[orientation_end:velocity_end]
            lidar_points=flattened_obs[velocity_end:]

            # Construct the dictionary
            parsed_obs = {
                "drone_position": drone_position,
                "orientation": orientation,
                "linear_velocity": linear_velocity,
                "lidar_points":lidar_points
            }

            return parsed_obs

    
    def step(self, action):
        # Use the expert's action as the ground truth
        done = self.current_index >= len(self.actions) - 1
        reward = 1.0 if action == self.actions[self.current_index] else -1.0  # Reward based on matching expert action
        obs = self.observations[self.current_index] if not done else np.zeros_like(self.observations[0])
        # reward=0
        # if not done:
        #     dic_obs=self.parse_flattened_obs(obs)
        #     reward=self._compute_reward(dic_obs)
        
        self.current_index += 1
        return obs, reward, done, False, {}


#pre-training phase
ip_client="127.0.0.1"
collector = ExpertDataCollector(ip_address=ip_client, csv_file="expert_data.csv",path_file='training_paths.json',enabled_drone=False)
#collector.generate_expert_obs()
observations,actions=collector.read_xp_data()
print(len(actions))

# # Create the custom imitation environment
imitation_env = ExpertImitationEnv(observations, actions)

# # Initialize the model
model = PPO("MlpPolicy", imitation_env,learning_rate=1e-3,verbose=1)

# # Pre-train the model using the expert data
model.learn(total_timesteps=len(observations) * 50)  # Train for multiple passes over the expert data

# # Save the pre-trained model
model.save("pretrained_PPO_agent")
print("model saved")

