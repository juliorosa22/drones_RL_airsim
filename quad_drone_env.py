import gymnasium as gym
import numpy as np
from gymnasium import spaces
import airsim
from airgym.envs.airsim_env import AirSimEnv
import cv2
from scipy.spatial.transform import Rotation as R
from path_handler import PathHandler

class QuadAirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, img_shape, path_file,min_dist=2 ,max_velocity=10):
        super().__init__()
        # Connect to AirSim
        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()
        self.img_shape = img_shape
        #paths handler
        self.path_handler=PathHandler(path_file)
        #drone initial state
        self.start_position=np.zeros(3, dtype=np.float32)
        self.target_position_position=np.zeros(3, dtype=np.float32)
        self.max_distance = np.zeros(3, dtype=np.float32)
        self.max_velocity = max_velocity  # Maximum velocity in m/s
        
        self.drone_orientation=np.zeros(3, dtype=np.float32)
        #self.previous_distance_to_target=np.linalg.norm(self.start_position - self.target_position)
        self.previous_altitude = 0.0
        #self.previous_min_depth = 0
        self.previous_perc_range=0.0
        self.previous_position=np.zeros(3, dtype=np.float32)
        self.min_dist_obs=min_dist #minimum distance threshold to penalize the drone when getting closer to obstacles
        self.previous_velocity = np.zeros(3, dtype=np.float32)
        #simulation counters
        self.collision_counter=0
        self.step_count=0
        self.update_path()
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
        #change the action space
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    def __del__(self):
        self.client.reset()

    def _set_previous_states(self):
        obs=self._get_obs()
        #depth_image=self._get_depth_image()
        self.start_position = obs["drone_position"]
        self.previous_position=obs["drone_position"]
        self.previous_altitude = float(self.start_position[2])
        self.previous_velocity=obs["linear_velocity"]
        #self.previous_min_depth, self.previous_perc_range = self._get_obstacle_proximity_features(depth_image)

    def update_path(self):
        path = self.path_handler.get_next_path()
        print(f"Next Path selected:{path['path_id']}")
        point=self.path_handler.get_next_point()
        self.target_position = np.array(point['target_position'],dtype=np.float32)
        self.start_position = np.array(point['start_position'],dtype=np.float32)
        self.max_distance= 1.5*np.linalg.norm(self.target_position - self.start_position)

    def _setup_flight(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        #self.client.takeoffAsync().join()
        state = self.client.getMultirotorState()
        if state.landed_state == airsim.LandedState.Landed:
            x,y,z=float(self.start_position[0]),float(self.start_position[1]),float(self.start_position[2])
            self.client.moveToPositionAsync(x,y,z, 0.75*self.max_velocity).join()
        self._set_previous_states()
        
    def reset(self, **kwargs):
        if 'seed' in kwargs:
            seed = kwargs['seed']
        
        # Reset environment and set to start position
        self.collision_counter=0
        self.step_count=0
        self.update_path()
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

    def parse_lidarData(self, data):
        # reshape array of floats to array of [X,Y,Z]
        points = np.array(data.point_cloud, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0]/3), 3))
        return points

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

    def new_reward(self,obs):
        done=False
        # Extract required observation data
        drone_position = obs["drone_position"]
        previous_position = self.previous_position
        target_position = self.target_position
        lidar_data = self.client.getLidarData()
        
        # Initialize rewards and penalties
        close_target_reward=0.0 #(a): range 0 or 1
        direction_reward=0.0 #(b): range[-1, 1]
        displacement_reward = 0.0 #(c): range 0 or 0.5
        yaw_reward=0 #(d): range 0 or [2/3,1]
        foward_reward =0.0 #(e): range [-1, 1]
        lidar_reward = 0 #(f): range[0,1] based on the percentage of distances above the threshold
        lidar_penalty = 0 # (g): range[0,1] based on the percentage of distances below the threshold
        collision_penalty = 0
        
        # ---- 1. Distance Based ----
        #(a)----- proximity reward based on current distance from target
        dist_target = np.linalg.norm(target_position-drone_position)
        start_dist_target=np.linalg.norm(target_position-self.start_position)
        if dist_target < 3:
            close_target_reward=1
            if dist_target < 1:
                done=True
        else:
            close_target_reward=0

        #(b)-----  Scalar product reward for movement in the target direction
        movement_vector = drone_position - previous_position
        prev_dist_vector = target_position - previous_position
        # Normalize vectors to calculate scalar product
        if np.linalg.norm(movement_vector) > 0 and np.linalg.norm(prev_dist_vector) > 0:
            movement_vector_normalized = movement_vector / np.linalg.norm(movement_vector)
            prev_distance_vector_normalized = prev_dist_vector / np.linalg.norm(prev_dist_vector)
            direction_reward = np.dot(movement_vector_normalized, prev_distance_vector_normalized)
    

        #(c)-----  Reward based on the displacement
        previous_distance_to_target = np.linalg.norm(self.previous_position - target_position)
        current_distance_to_target = np.linalg.norm(drone_position - target_position)
        delta_displacement= previous_distance_to_target - current_distance_to_target
        displacement_reward= 0.5 if delta_displacement > 0 else 0

        #(d)-----  Drone yaw alignment reward, measures if the drone is aligned with the target
        relative_yaw=obs["relative_yaw"]
        yaw_reward= (np.pi-abs(relative_yaw))/np.pi if -(np.pi/3) <= relative_yaw <= (np.pi/3) else 0
        
        #(e)-----  This reward tries to incentivize the drone movement in its camera-facing position
        orientation = obs["orientation"]  # Assuming the drone's orientation is provided in roll, pitch, yaw
        yaw = orientation[2]  # Extract yaw in radians
        facing_direction = np.array([np.cos(yaw), np.sin(yaw), 0])  # Facing direction vector in X-Y plane
        # Calculate movement direction based on positions
        movement_vector_norm = np.zeros_like(movement_vector)
        if np.linalg.norm(movement_vector) > 1e-6:  # Avoid division by zero
            movement_vector_norm = movement_vector/np.linalg.norm(movement_vector)  # Normalize
            
        # Check alignment between movement and facing direction
        foward_reward = np.dot(facing_direction[:2], movement_vector_norm[:2])  # Consider only X-Y plane for alignment
        


        # ---- 2. LiDAR Based ----
        # Parse LiDAR data into 3D point cloud
        if len(lidar_data.point_cloud) < 3:
            lidar_points = np.empty((0, 3))  # No points received, empty array
        else:
            lidar_points = self.parse_lidarData(lidar_data)

        if lidar_points.size > 0:
            # (a) Reward for open spaces in front
            distances_to_points = np.linalg.norm(lidar_points, axis=1)
            front_area_indices = lidar_points[:, 0] > 0  # Points in front of the drone
            front_distances = distances_to_points[front_area_indices]

            if len(front_distances) > 0:
                average_front_distance = np.mean(front_distances)
                #if the current mean distance is bigger than the minimum threshold apply reward based on how many distances above the mean are in the front of the drone
                #(f) -------- Reward for movement in wide open space
                if average_front_distance > self.min_dist_obs:
                    above_mean_count = np.sum(front_distances > average_front_distance*0.9)
                    # Calculate the percentage of distances above the mean
                    lidar_reward = (above_mean_count / len(front_distances))
                # (g)------ Penalty for proximity to obstacles        
                if average_front_distance <= self.min_dist_obs:                    
                    below_mean_count = np.sum(front_distances < average_front_distance)
                    # Calculate the percentage of distances above the mean
                    lidar_penalty = (below_mean_count / len(front_distances))
                    
        estimated_reward = ( 
            6*close_target_reward+
            displacement_reward+
            direction_reward+
            foward_reward+
            yaw_reward+
            lidar_reward-lidar_penalty                        
        )
        
        #when the penalties are bigger than the rewards
        if estimated_reward < 0:
            estimated_reward=np.log(estimated_reward)*np.log(self.step_count)
        else:
            estimated_reward = np.power(2,int(abs(estimated_reward)))
        # ---- 3. Collision Penalty ----
        collision = self.client.simGetCollisionInfo().has_collided
        if collision:
            collision_penalty = -np.power(2,int(abs(estimated_reward))+1)   # Proportional to previous rewards/penalties

        
        total_reward = float(estimated_reward+collision_penalty)

        # Update state for next reward calculation
        self.previous_position = drone_position

        return total_reward, done, dist_target
    
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

    def update_points(self):
        new_points=self.path_handler.get_next_point()    
        if new_points is None:
            return True
        self.start_position = np.array(new_points['start_position'],dtype=np.float32)
        self.target_position = np.array(new_points['target_position'],dtype=np.float32)
        return False

    def step(self, action):
        done_point,done_path=False, False
        self.step_count+=1

        self._do_action(action)
        # Get new state and compute reward
        obs = self._get_obs()
        reward, done_point,dist_target = self.new_reward(obs)
        if done_point:
            done_path=self.update_points()
        
        terminated = True if done_path else False # Finish the current episode in case all the points in the path were used
        # Truncate if reward drops below threshold
        truncated = True if dist_target > self.max_distance else False
        return obs, reward, terminated, truncated, {}

    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
