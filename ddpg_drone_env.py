import gymnasium as gym
import numpy as np
from gymnasium import spaces
import airsim
from airgym.envs.airsim_env import AirSimEnv
import cv2
from scipy.spatial.transform import Rotation as R

class DDPGAirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, img_shape, start_position, target_position, max_velocity=5, max_yaw_rate=45):
        super().__init__()

        # Connect to AirSim
        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()
        self.img_shape = img_shape
        self.target_position = np.array(target_position)
        self.start_position = np.array(start_position)
        self.max_distance = float(1.5 * np.linalg.norm(self.start_position - self.target_position))
        self.previous_distance_to_target=np.linalg.norm(self.start_position - self.target_position)
        self.max_velocity = max_velocity  # Maximum velocity in m/s
        self.max_yaw_rate = max_yaw_rate  # Maximum yaw change in degrees per step
        self.previous_min_depth = None
        self.previous_velocity = np.zeros(3, dtype=np.float32)

        # Setup flight
        self._setup_flight()

        # Define observation space (depth image + drone state + relative orientation)
        self.observation_space = spaces.Dict({
            #"depth_image": spaces.Box(low=0, high=255, shape=self.img_shape, dtype=np.uint8),
            "drone_position": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            #"target_position": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            #"orientation": spaces.Box(low=-np.pi, high=np.pi, shape=(3,), dtype=np.float32),  # roll, pitch, yaw in radians
            "linear_velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            #"relative_distance": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "relative_yaw": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),  # Yaw angle to align with target
        })

        # Define action space (limited [-1, 1] for vx, vy, vz, and yaw rate)
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    def __del__(self):
        self.client.reset()

    def _setup_flight(self):
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        x,y,z=float(self.start_position[0]),float(self.start_position[1]),float(self.start_position[2])
        self.client.moveToPositionAsync(x,y,z, 10).join()

    def reset(self, **kwargs):
        if 'seed' in kwargs:
            seed = kwargs['seed']
        
        # Reset environment and set to start position
        self.client.reset()
        self._setup_flight()
        self.previous_distance_to_target = np.linalg.norm(self.start_position - self.target_position)
        self.previous_min_depth = None
        self.previous_velocity = np.zeros(3, dtype=np.float32)
        # Get initial observation
        return self._get_observation(), {}

    def step(self, action):
        # Scale action values to the desired range
        vx = float(action[0] * self.max_velocity)
        vy = float(action[1] * self.max_velocity)
        vz = float(action[2] * self.max_velocity)
        yaw_rate = float(action[3] * self.max_yaw_rate)  # Max yaw rate change in degrees per step

        self.client.rotateByYawRateAsync(yaw_rate,duration=0.3).join()
        self.client.moveByVelocityAsync(vx,vy,vz,duration=0.3).join()
        # Get new state and compute reward
        obs = self._get_observation()
        reward, done , target_dist= self._compute_reward_and_done(obs)
        # Truncate if reward drops below threshold
        terminated = done
        truncated = target_dist > self.max_distance and not terminated
        return obs, reward, terminated, truncated, {}

    def get_current_yaw(self):
        # Get current yaw from quaternion orientation
        orientation = self.client.getMultirotorState().kinematics_estimated.orientation
        rotation = R.from_quat([orientation.x_val, orientation.y_val, orientation.z_val, orientation.w_val])
        _, _, yaw = rotation.as_euler('xyz', degrees=False)
        return yaw

    def _get_relative_yaw_to_target(self, current_position, current_yaw):
        # Compute direction vector from drone to target
        direction_vector = self.target_position - current_position
        target_yaw = np.arctan2(direction_vector[1], direction_vector[0])  # Yaw angle to face the target

        # Calculate the relative yaw between drone's current yaw and the target yaw
        relative_yaw = target_yaw - current_yaw
        # Normalize relative yaw to the range [-pi, pi]
        relative_yaw = (relative_yaw + np.pi) % (2 * np.pi) - np.pi
        return relative_yaw

    def _get_observation(self):
        # Get depth image and resize
        #depth_image = self._get_depth_image()
        #depth_image = depth_image.reshape(self.img_shape)

        # Get drone's current position, orientation, linear velocity
        drone_state = self.client.getMultirotorState()
        drone_position = drone_state.kinematics_estimated.position
        current_position = np.array([drone_position.x_val, drone_position.y_val, drone_position.z_val])

        # Orientation (roll, pitch, yaw) from quaternion
        orientation_quat = drone_state.kinematics_estimated.orientation
        orientation_euler = self.get_euler_angles(orientation_quat)

        # Linear velocity
        linear_velocity = drone_state.kinematics_estimated.linear_velocity
        current_velocity = np.array([linear_velocity.x_val, linear_velocity.y_val, linear_velocity.z_val])

        # Relative distance to target (component-wise difference)
        #relative_distance = self.target_position - current_position

        # Relative yaw to face the target
        current_yaw = orientation_euler[2]  # Extract the yaw angle
        relative_yaw = self._get_relative_yaw_to_target(current_position, current_yaw)

        return {
            #"depth_image": depth_image,
            "drone_position": current_position,
            #"target_position": self.target_position,
            #"orientation": np.array(orientation_euler),  # roll, pitch, yaw
            "linear_velocity": current_velocity,
            #"relative_distance": relative_distance,
            "relative_yaw": np.array([relative_yaw]),  # Yaw difference to align with the target
        }

    def get_euler_angles(self, quaternion):
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        rotation = R.from_quat([quaternion.x_val, quaternion.y_val, quaternion.z_val, quaternion.w_val])
        return rotation.as_euler('xyz', degrees=False)  # Returns roll, pitch, yaw

    def _get_obstacle_proximity_features(self):
        # Get depth image and preprocess
        depth_image = self._get_depth_image()

        # 1. Minimum Depth Value (nearest obstacle)
        min_depth = np.min(depth_image)

        # 2. Quadrant-Based Minimum Depths
        h, w = depth_image.shape
        quadrants = [
            depth_image[:h//2, :w//2],     # Top-left
            depth_image[:h//2, w//2:],     # Top-right
            depth_image[h//2:, :w//2],     # Bottom-left
            depth_image[h//2:, w//2:],     # Bottom-right
        ]
        quadrant_depths = [np.min(quad) for quad in quadrants]

        return min_depth, quadrant_depths

    def _get_depth_image(self):
        # Get depth image from AirSim
        responses = self.client.simGetImages([airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, True, False)])
        depth_img = responses[0]
        depth_data = airsim.list_to_2d_float_array(depth_img.image_data_float, depth_img.height, depth_img.width)
        depth_data = (255.0 * depth_data / np.max(depth_data)).astype(np.uint8)  # Normalize
        return cv2.resize(depth_data, (self.img_shape[0], self.img_shape[1]))

    def _compute_reward_and_done(self, obs):
        # Initialize done flag
        done = False
        
        # Calculate distance to target
        drone_position, current_velocity, relative_yaw = obs["drone_position"], obs["linear_velocity"], obs["relative_yaw"]
        distance_to_target = np.linalg.norm(drone_position - self.target_position)
        distance_reward = self.previous_distance_to_target - distance_to_target
        self.previous_distance_to_target = distance_to_target

        # Penalize yaw misalignment when no obstacle is close
        yaw_penalty = -5 * abs(relative_yaw) if distance_to_target > 5.0 else -abs(relative_yaw)

        # Reward for moving away from the nearest obstacle
        min_depth, quadrant_depths = self._get_obstacle_proximity_features()
        obstacle_reward = 0
        if min_depth < 5.0 and self.previous_min_depth is not None and min_depth > self.previous_min_depth:
            obstacle_reward = 10 * (min_depth - self.previous_min_depth)
        self.previous_min_depth = min_depth

        # Smooth movement penalty for excessive velocity changes (uncomment if needed)
        # velocity_change_penalty = -np.linalg.norm(current_velocity - self.previous_velocity)
        self.previous_velocity = current_velocity

        # Proximity bonus for getting closer to the target
        proximity_bonus = 10 / (0.1 + distance_to_target) if distance_to_target < 3.0 else 0
        if distance_to_target < 1.5:
            proximity_bonus += 100  # Large reward for reaching the target
            done = True

        # Collision penalty
        collision = self.client.simGetCollisionInfo().has_collided
        collision_penalty = -1e6 if collision else 0
        if collision:
            done = True

        # Time penalty to encourage faster completion
        time_penalty = -0.1

        # Combine all rewards and penalties with appropriate scaling
        reward = (
            10 * distance_reward +
            yaw_penalty +
            obstacle_reward +
            # velocity_change_penalty +  # Uncomment if needed
            proximity_bonus +
            collision_penalty +
            time_penalty
        )

        # Determine episode termination
        done = done or distance_to_target < 1
        return reward, done, distance_to_target
    
    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
