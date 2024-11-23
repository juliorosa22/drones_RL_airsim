import gymnasium as gym
import numpy as np
from gymnasium import spaces
import airsim
from airgym.envs.airsim_env import AirSimEnv
import cv2
from scipy.spatial.transform import Rotation as R

class DDPGAirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, img_shape, start_position, target_position,prox_tresh=5 ,max_velocity=10, max_yaw_rate=np.pi/4):
        super().__init__()

        # Connect to AirSim
        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()
        self.img_shape = img_shape
        self.target_position = np.array(target_position)
        self.start_position = np.array(start_position)
        self.max_velocity = max_velocity  # Maximum velocity in m/s
        self.max_distance = float(1.5 * np.linalg.norm(self.start_position - self.target_position))
        self.max_yaw_rate = max_yaw_rate  # Maximum yaw change in degrees per step
        self.drone_orientation=np.zeros(3, dtype=np.float32)
        self.previous_distance_to_target=np.linalg.norm(self.start_position - self.target_position)
        self.previous_altitude = 0.0
        self.previous_min_depth = 0
        self.previous_perc_range=0.0
        self.previous_position=np.zeros(3, dtype=np.float32)
        self.prox_tresh=prox_tresh #paramater that defines when consider change yaw aligment due proximity to some obstacle in front of the drone
        self.previous_velocity = np.zeros(3, dtype=np.float32)
        self.collision_counter=0

        # Setup flight
        self._setup_flight()

        # Define observation space (depth image + drone state + relative orientation)
        self.observation_space = spaces.Dict({
            #"depth_image": spaces.Box(low=0, high=255, shape=self.img_shape, dtype=np.uint8),
            "drone_position": spaces.Box(low=np.array([-100, -100, -100]), high=np.array([100, 100, 100]), dtype=np.float32),
            #"target_position": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "orientation": spaces.Box(low=np.array([-np.pi,-np.pi,-np.pi]), high=np.array([np.pi,np.pi,np.pi]), dtype=np.float32),  # roll, pitch, yaw in radians
            "linear_velocity": spaces.Box(low=np.array([-self.max_velocity, -self.max_velocity, -self.max_velocity]), high=np.array([self.max_velocity, self.max_velocity, self.max_velocity]), dtype=np.float32),
            #"relative_distance": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "relative_yaw": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),  # Yaw angle to align with target
        })

        # Define action space (limited [-1, 1] for vx, vy, vz, and yaw ratem(rads/sec))
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    def __del__(self):
        self.client.reset()

    def _set_previous_states(self):
        obs=self._get_obs()
        depth_image=self._get_depth_image()
        self.start_position = obs["drone_position"]
        self.previous_position=obs["drone_position"]
        self.previous_altitude = float(self.start_position[2])
        self.previous_velocity=obs["linear_velocity"]
        self.previous_min_depth, self.previous_perc_range = self._get_obstacle_proximity_features(depth_image)

    def _setup_flight(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        #self.client.takeoffAsync().join()
        x,y,z=float(self.start_position[0]),float(self.start_position[1]),float(self.start_position[2])
        self.client.moveToPositionAsync(x,y,z, 10).join()
        self._set_previous_states()
        
    def reset(self, **kwargs):
        if 'seed' in kwargs:
            seed = kwargs['seed']
        
        # Reset environment and set to start position
        self.collision_counter=0
        self._setup_flight()
        # Get initial observation
        return self._get_obs(), {}
        
    def get_euler_angles(self, quaternion):
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        rotation = R.from_quat([quaternion.x_val, quaternion.y_val, quaternion.z_val, quaternion.w_val])
        return rotation.as_euler('xyz', degrees=False)  # Returns roll, pitch, yaw

    def _get_relative_yaw_to_target(self, current_position, current_yaw):
        # Compute direction vector from drone to target
        direction_vector = self.target_position - current_position
        target_yaw = np.arctan2(direction_vector[1], direction_vector[0])  # Yaw angle to face the target

        # Calculate the relative yaw between drone's current yaw and the target yaw
        relative_yaw = target_yaw - current_yaw
        # Normalize relative yaw to the range [-pi, pi]
        relative_yaw = (relative_yaw + np.pi) % (2 * np.pi) - np.pi
        return relative_yaw.astype(np.float32)

    def _get_depth_image(self):
        # Get depth image from AirSim
        responses = self.client.simGetImages([airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, True, False)])
        depth_img = responses[0]
        depth_data = airsim.list_to_2d_float_array(depth_img.image_data_float, depth_img.height, depth_img.width)
        depth_data = (255.0 * depth_data / np.max(depth_data)).astype(np.uint8)  # Normalize
        return cv2.resize(depth_data, (self.img_shape[0], self.img_shape[1]))

    def _get_obstacle_proximity_features(self, depth_image):
        # Remove the last dimension if the image has it (e.g., shape [H, W, 1])
        #depth_image = np.squeeze(depth_image, axis=-1)
        # Get image dimensions
        h, w = depth_image.shape
        # Define cropping region for the central portion (e.g., 50% of height and width)
        crop_height = int(h * 0.5)
        crop_width = int(w * 0.5)
        start_y = (h - crop_height) // 2
        start_x = (w - crop_width) // 2
        
        # Crop the central portion of the depth image
        central_depth_image = depth_image[start_y:start_y + crop_height, start_x:start_x + crop_width]
        #print("central depth image")
        #print(central_depth_image)
        # 1. Minimum Depth Value in the central portion (nearest obstacle)
        min_depth = np.min(central_depth_image)

        # 2. Quadrant-Based Minimum Depths within the cropped region
        #ch, cw = central_depth_image.shape
        #quadrants = [
        #    central_depth_image[:ch//2, :cw//2],     # Top-left
        #    central_depth_image[:ch//2, cw//2:],     # Top-right
        #    central_depth_image[ch//2:, :cw//2],     # Bottom-left
        #    central_depth_image[ch//2:, cw//2:],     # Bottom-right
        #]
        #quadrant_depths = [np.min(quad) for quad in quadrants]

        # 3. Percentage of pixels within 90% of the minimum value
        threshold = min_depth * 1.05  # 95% of minimum value (min_depth + 10%)
        pixels_within_range = np.sum(central_depth_image <= threshold)
        total_pixels = central_depth_image.size
        percentage_within_range = (pixels_within_range / total_pixels) * 100

        return min_depth, percentage_within_range

    def _get_obs(self):
        # Get depth image and resize
        #depth_image = self._get_depth_image()
        #depth_image = np.expand_dims(depth_image, axis=-1)
        # Get drone's current position, orientation, linear velocity
        drone_state = self.client.getMultirotorState()
        drone_position = drone_state.kinematics_estimated.position
        current_position = np.array([drone_position.x_val, drone_position.y_val, drone_position.z_val],dtype=np.float32)

        # Orientation (roll, pitch, yaw) from quaternion
        orientation_quat = drone_state.kinematics_estimated.orientation
        orientation_euler = self.get_euler_angles(orientation_quat)

        # Linear velocity
        linear_velocity = drone_state.kinematics_estimated.linear_velocity
        current_velocity = np.array([linear_velocity.x_val, linear_velocity.y_val, linear_velocity.z_val],dtype=np.float32)

        # Relative distance to target (component-wise difference)
        #relative_distance = self.target_position - current_position

        # Relative yaw to face the target
        current_yaw = orientation_euler[2]  # Extract the yaw angle
        relative_yaw = self._get_relative_yaw_to_target(current_position, current_yaw)
        drone_orientation=np.array(orientation_euler,dtype=np.float32)
        
        self.drone_orientation=drone_orientation
        return {
            #"depth_image": depth_image,
            "drone_position": current_position,
            #"target_position": self.target_position,
            "orientation": drone_orientation,  # roll, pitch, yaw
            "linear_velocity": current_velocity,
            #"relative_distance": relative_distance,
            "relative_yaw": np.array([relative_yaw]),  # Yaw difference to align with the target
        }

    def _compute_reward(self, obs):
        # Initialize done flag
        done = False

        # Calculate distance to target
        drone_position, current_velocity, relative_yaw = obs["drone_position"], obs["linear_velocity"], obs["relative_yaw"]
        distance_to_target = np.linalg.norm(drone_position - self.target_position)
        delta = 1 if self.previous_distance_to_target - distance_to_target > 0 else -1
        distance_reward = 100 * delta
        self.previous_distance_to_target = distance_to_target
        

        ##### This reward tries to incentivize the drone movement in its camera-facing position
        orientation = obs["orientation"]  # Assuming the drone's orientation is provided in roll, pitch, yaw
        yaw = orientation[2]  # Extract yaw in radians
        facing_direction = np.array([np.cos(yaw), np.sin(yaw), 0])  # Facing direction vector in X-Y plane
        # Calculate movement direction based on positions
        movement_direction = drone_position - self.previous_position
        if np.linalg.norm(movement_direction) > 1e-6:  # Avoid division by zero
            movement_direction /= np.linalg.norm(movement_direction)  # Normalize
        else:
            movement_direction = np.zeros_like(movement_direction)
        # Check alignment between movement and facing direction
        alignment = np.dot(facing_direction[:2], movement_direction[:2])  # Consider only X-Y plane for alignment
        alignment_reward = 0
        if alignment > 0.8:  # Movement is closely aligned with the facing direction
            print("Movement in direction of camera")
            alignment_reward = 50  # Positive reward for aligned movement
            if delta > 0:
                distance_reward*=1.5 #bonus in case the movement was in the direction of the camera facing and the distance was reduced
        elif alignment < 0.5:
            alignment_reward = -30
        # Update the previous position for the next step
        
        

        # Penalty for larger changes in Z altitude
        altitude_change = abs(drone_position[2] - self.previous_altitude)
        altitude_penalty = -3 * altitude_change  # Scale the penalty as needed
        
        

        # Calculate the misalignment change and update the previous yaw
        yaw_penalty = -2 * abs(relative_yaw) if abs(relative_yaw) > np.pi / 2 else -abs(relative_yaw)
        

        # Reward for moving away from the nearest obstacle
        depth_image = self._get_depth_image()
        min_depth, perc_range_obstacle = self._get_obstacle_proximity_features(depth_image)
        obstacle_reward = 0
        if min_depth < self.prox_tresh:  # This indicates that drone is close to some obstacle
            if perc_range_obstacle > 0.9:  # The main portion of the central image is occupied by an obstacle
                yaw_penalty *= 0.1  # Downscale the yaw misalignment penalty temporarily
                self.previous_perc_range = perc_range_obstacle
                #print("drone very close to obstacle")
            if self.previous_perc_range > 0.9 and self.previous_perc_range > perc_range_obstacle:
                print("obstacle reward achieved")
                obstacle_reward = 30
            
        

        # Smooth movement penalty for excessive velocity changes
        velocity_change_penalty = -np.linalg.norm(current_velocity - self.previous_velocity)
        
        

        # Collision penalty
        collision = self.client.simGetCollisionInfo().has_collided
        collision_penalty = 0
        if collision:
            self.collision_counter += 1
            print(f"collision counter:{self.collision_counter}")
            collision_penalty = -1e3 * self.collision_counter
        

        # Terminate the episode if too many collisions
        done = True if self.collision_counter > 1000 else False

        # Time penalty to encourage faster completion
        time_penalty = -0.1 * distance_to_target
        

        # Proximity bonus
        proximity_bonus = 0
        if distance_to_target < 5.0:
            proximity_bonus = 1e3
            if distance_to_target < 1.5:
                proximity_bonus = 1e6
                done = True
        print(f"Proximity Bonus: {proximity_bonus}")

        self.previous_altitude = drone_position[2]
        self.previous_position = drone_position
        self.previous_velocity = current_velocity

        # Combine all rewards and penalties with appropriate scaling
        reward = (
            distance_reward +
            yaw_penalty +
            obstacle_reward +
        #    velocity_change_penalty +
            proximity_bonus +
            alignment_reward +
            altitude_penalty +
            collision_penalty +
            time_penalty
        )
        #print(f"Total:{reward} | distance:{distance_reward} | yaw pen:{yaw_penalty} | obstacle:{obstacle_reward}")
        #print(f"velocity:{velocity_change_penalty} |proximity:{proximity_bonus} |alignment:{alignment_reward}")
        #print(f"altitude pen:{altitude_penalty} | collision pen:{collision_penalty} | time: {time_penalty}")
        print(f"Total Reward: {reward}")
        print("######################")

        return float(reward), done, distance_to_target

    
    def _do_action(self,action):
        # Scale action values to the desired range
        vx = float(action[0] *0.5*self.max_velocity)
        vy = float(action[1] *0.5*self.max_velocity)
        vz = float(action[2] *0.1* self.max_velocity)
        delta_yaw_target_rad = float(action[3])  # Max yaw rate change in degrees per step
        # Convert yaw_angle to degrees for AirSim
        yaw_action = np.degrees(delta_yaw_target_rad+self.drone_orientation[2])#gets the current drone orientation and sums the delta rad to align with the target an then transforms to degree to use in airsim
        
        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=float(yaw_action))
        self.client.moveByVelocityAsync(vx, vy, vz, duration=0.5, yaw_mode=yaw_mode).join()

        #self.client.rotateByYawRateAsync(yaw_rate,duration=0.5).join()
        #self.client.moveByVelocityAsync(c_vx+a_vx,c_vy+a_vy,c_vz+a_vz,duration=0.5).join()

    def step(self, action):
        self._do_action(action)
        # Get new state and compute reward
        obs = self._get_obs()
        reward, done , target_dist= self._compute_reward(obs)
        # Truncate if reward drops below threshold
        terminated = True if done else False
        truncated = True if target_dist > self.max_distance and not terminated else False
        return obs, reward, terminated, truncated, {}

    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
