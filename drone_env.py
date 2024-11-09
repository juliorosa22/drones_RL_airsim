import gymnasium as gym
import numpy as np
from gymnasium import spaces
import airsim
from airgym.envs.airsim_env import AirSimEnv
import cv2
from scipy.spatial.transform import Rotation as R

class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, img_shape, start_position, target_position, max_velocity=5, max_yaw_rate=45):
        super().__init__()

        # Connect to AirSim
        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()
        self.img_shape = img_shape
        self.target_position = np.array(target_position)
        self.start_position = np.array(start_position)
        self.min_threshold = float(-2 * np.linalg.norm(self.start_position - self.target_position))
        self.max_velocity = max_velocity  # Maximum velocity in m/s
        self.max_yaw_rate = max_yaw_rate  # Maximum yaw change in degrees per step

        # Setup flight
        self._setup_flight()

        # Define observation space (depth image + drone state + relative orientation)
        self.observation_space = spaces.Dict({
            "depth_image": spaces.Box(low=0, high=255, shape=self.img_shape, dtype=np.uint8),
            #"drone_position": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            #"target_position": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            #"orientation": spaces.Box(low=-np.pi, high=np.pi, shape=(3,), dtype=np.float32),  # roll, pitch, yaw in radians
            "linear_velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "relative_distance": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
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
        self.client.moveToPositionAsync(*self.start_position, 10).join()

    def reset(self, **kwargs):
        if 'seed' in kwargs:
            seed = kwargs['seed']
        
        # Reset environment and set to start position
        self.client.reset()
        self._setup_flight()

        # Get initial observation
        return self._get_observation(), {}

    def step(self, action):
        # Scale action values to the desired range
        vx = float(action[0] * self.max_velocity)
        vy = float(action[1] * self.max_velocity)
        vz = float(action[2] * self.max_velocity)
        yaw_rate = float(action[3] * self.max_yaw_rate)  # Max yaw rate change in degrees per step

        # Get current yaw, adjust it by yaw_rate
        #current_yaw = self.get_current_yaw()
        #new_yaw = current_yaw + np.deg2rad(yaw_rate)  # Convert degrees to radians

        # Move by velocity and set yaw
        #self.client.moveByVelocityZAsync(vx, vy, vz, yaw_mode=new_yaw, duration=0.3).join()
        self.client.rotateByYawRateAsync(yaw_rate,duration=0.3).join()
        self.client.moveByVelocityAsync(vx,vy,vz,duration=0.3).join()
        # Get new state and compute reward
        obs = self._get_observation()
        reward, done = self._compute_reward_and_done(obs["drone_position"])

        # Truncate if reward drops below threshold
        terminated = done
        truncated = reward < self.min_threshold and not terminated

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
        depth_image = self._get_depth_image()
        depth_image = depth_image.reshape(self.img_shape)

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
        relative_distance = self.target_position - current_position

        # Relative yaw to face the target
        current_yaw = orientation_euler[2]  # Extract the yaw angle
        relative_yaw = self._get_relative_yaw_to_target(current_position, current_yaw)

        return {
            "depth_image": depth_image,
            #"drone_position": current_position,
            #"target_position": self.target_position,
            #"orientation": np.array(orientation_euler),  # roll, pitch, yaw
            "linear_velocity": current_velocity,
            "relative_distance": relative_distance,
            "relative_yaw": np.array([relative_yaw]),  # Yaw difference to align with the target
        }

    def get_euler_angles(self, quaternion):
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        rotation = R.from_quat([quaternion.x_val, quaternion.y_val, quaternion.z_val, quaternion.w_val])
        return rotation.as_euler('xyz', degrees=False)  # Returns roll, pitch, yaw

    def _get_depth_image(self):
        # Get depth image from AirSim
        responses = self.client.simGetImages([airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, True, False)])
        depth_img = responses[0]
        depth_data = airsim.list_to_2d_float_array(depth_img.image_data_float, depth_img.height, depth_img.width)
        depth_data = (255.0 * depth_data / np.max(depth_data)).astype(np.uint8)  # Normalize
        return cv2.resize(depth_data, (self.img_shape[0], self.img_shape[1]))

    #TO DO improve the reward function
    def _compute_reward_and_done(self, drone_position, current_velocity, relative_yaw):
        # Calculate distance to target
        distance_to_target = np.linalg.norm(drone_position - self.target_position)

        # Reward for getting closer to the target (based on change in distance)
        distance_reward = self.previous_distance_to_target - distance_to_target
        self.previous_distance_to_target = distance_to_target

        # Penalize for misalignment in yaw
        alignment_reward = -abs(relative_yaw)  # Negative reward for misalignment

        # Penalize large changes in velocity to encourage smooth movement
        velocity_change_penalty = -np.linalg.norm(current_velocity - self.previous_velocity)
        self.previous_velocity = current_velocity

        # Proximity bonus
        proximity_bonus = 0
        if distance_to_target < 3.0:
            proximity_bonus = 10  # Bonus for getting close to the target
        if distance_to_target < 1.5:
            proximity_bonus = 100  # Large reward for reaching the target
            done = True

        # Collision penalty
        collision = self.client.simGetCollisionInfo().has_collided
        collision_penalty = -1e6 if collision else 0
        if collision:
            done = True

        # Time penalty
        time_penalty = -0.1  # Small penalty to encourage faster completion

        # Combine all rewards
        reward = (distance_reward+alignment_reward+velocity_change_penalty+ proximity_bonus+ collision_penalty+ time_penalty)

        # Episode termination
        done = collision or distance_to_target < 0.5

        return reward, done

    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
