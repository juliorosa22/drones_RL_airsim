from stable_baselines3 import DQN, PPO
from generate_expert_data import ExpertDataCollector
from path_handler import PathHandler
import gymnasium as gym
from gymnasium import spaces
import numpy as np
LIDAR_FEAT_SIZE=600
class ExpertImitationEnv(gym.Env):
    def __init__(self, observations,actions):
        super(ExpertImitationEnv, self).__init__()
        self.observations = observations
        self.path_handler=PathHandler("training_paths.json")
        self.min_dist=2
        self.done_count=0
        self.min_dist_obs=2.0
        #self.target_positions=self.extract_target_positions()
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

    def parse_lidarData(self, data):
        # reshape array of floats to array of [X,Y,Z]
        points = np.array(data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0]/3), 3))
        return points

    def reset(self, **kwargs):
       
        self.current_index = 0
        return self.observations[self.current_index],{}

    def parse_flattened_obs(self,flattened_obs):
    
        # Indices for splitting the flattened observation
        position_end = 3  # First 3 elements are drone position
        orientation_end = position_end + 3  # Next 3 elements are orientation (roll, pitch, yaw)
        velocity_end = orientation_end + 3  # Next 3 elements are linear velocity
        

        # Parse data
        drone_position = flattened_obs[:position_end]
        orientation = flattened_obs[position_end:orientation_end]
        target_position = flattened_obs[orientation_end:velocity_end]
        lidar_distances=flattened_obs[velocity_end:]

        # Construct the dictionary
        parsed_obs = {
            "drone_position": drone_position,
            "orientation": orientation,
            "target_position": target_position,
            "lidar_distances":lidar_distances
        }

        return parsed_obs

    def extract_target_positions(self):
        target_positions = []
        paths=self.path_handler.paths
        for path in paths:
            for point in path["points"]:
                target_positions.append(point["target_position"])
        return np.array(target_positions)
    
    def is_position_close(self,position, threshold=2.0):
        """
        Check if a position is close to any target positions.
        
        :param position: List or array of the current position [x, y, z].
        :param target_positions: Array of target positions.
        :param threshold: Distance threshold for proximity.
        :return: True if close to any target position, False otherwise.
        """
        position = np.array(position)
        distances = np.linalg.norm(self.target_positions - position, axis=1)
        return np.any(distances < threshold)

    def estimate_reward(self,obs):
        pos=obs["drone_position"]

        is_near_target = self.is_position_close(pos)
        proximity_reward=0
        if is_near_target:
            proximity_reward=1e6

        lidar_points=obs["lidar_points"]
        check_lidar=np.any(lidar_points!=0)
        lidar_points=self.parse_lidarData(lidar_points)
        lidar_penalty=0
        if check_lidar:
            distances_to_points = np.linalg.norm(lidar_points, axis=1)
            average_distance = np.mean(distances_to_points)                
            #penalty for proximity with obstacles
            if average_distance < self.min_dist_obs * 1.5:
                lidar_penalty=-1e3

        total_rwd=proximity_reward+lidar_penalty
        return total_rwd

    def step(self, action):
        
        # Use the expert's action as the ground truth
        done = self.current_index >= len(self.actions) - 1
        reward = 1.0 if action == self.actions[self.current_index] else -1.0  # Reward based on matching expert action
        self.current_index += 1
        obs = self.observations[self.current_index] if not done else np.zeros_like(self.observations[0])
        return obs, reward, done, False, {}


#pre-training phase
ip_client="127.0.0.1"
collector = ExpertDataCollector(ip_address=ip_client, csv_file="expert_data_6.csv",path_file='training_paths.json',enabled_drone=False)
#collector.generate_expert_obs()
#collector.run_new_expert()
run= True
if run:
    observations,actions=collector.read_xp_data()
    print(f"{len(actions),len(observations)}")

    # # # # # # # Create the custom imitation environment
    imitation_env = ExpertImitationEnv(observations, actions)

    # # # # # # # Initialize the model
    model = DQN("MlpPolicy", imitation_env,learning_rate=1e-3,verbose=1)

    # # # # # # # Pre-train the model using the expert data
    model.learn(total_timesteps=len(observations) * 200)  # Train for multiple passes over the expert data

    # # # # # # # Save the pre-trained model
    model.save("imitation_DQN_agent")
    print("model saved")

