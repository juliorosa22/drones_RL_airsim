import gymnasium as gym
import numpy as np
from gymnasium import spaces
import airsim
from airgym.envs.airsim_env import AirSimEnv
from scipy.spatial.transform import Rotation as R
from path_handler import PathHandler
import time
class QuadAirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, img_shape, path_file,max_yaw_rate=45,min_dist=2 ,max_velocity=10):
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
        self.max_yaw_rate=max_yaw_rate
        self.drone_orientation=np.zeros(3, dtype=np.float32)
        
        #self.previous_altitude = 0.0
        
        
        self.previous_position=np.zeros(3, dtype=np.float32)
        self.min_dist_obs=min_dist #minimum distance threshold to penalize the drone when getting closer to obstacles
        self.previous_velocity = np.zeros(3, dtype=np.float32)
        
        
        self.update_path()
        self._setup_flight()
        # Define observation space (drone position, orientation,linear velocity, relative yaw to target)
        self.observation_space = spaces.Dict({
            "drone_position": spaces.Box(low=np.array([-100, -100, -100]), high=np.array([100, 100, 100]), dtype=np.float32),
            "orientation": spaces.Box(low=np.array([-np.pi,-np.pi,-np.pi]), high=np.array([np.pi,np.pi,np.pi]), dtype=np.float32),  # roll, pitch, yaw in radians
            "linear_velocity": spaces.Box(low=np.array([-self.max_velocity, -self.max_velocity, -self.max_velocity]), high=np.array([self.max_velocity, self.max_velocity, self.max_velocity]), dtype=np.float32),
            "relative_yaw": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),  # Yaw angle to align with target
        })
        #Information used to decide when stop the episode due agent random behavior
        self.sim_info = {
            "collision_counter":0,
            "step_counter":0,
            "dist_to_target":0.0,
            #"Truncate": False,
            "reward_components":[],
            "reward_names":[],
            }
        

        # Define action space (limited [-1, 1] for vx, vy, vz, and yaw ratem(rads/sec))
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    def __del__(self):
        print("del called")
        self.client.reset()
      

    def update_path(self):
        path = self.path_handler.get_next_path()
        print(f"Next Path selected:{path['path_id']}")
        point=self.path_handler.get_next_point()
        self.target_position = np.array(point['target_position'],dtype=np.float32)
        self.start_position = np.array(point['start_position'],dtype=np.float32)
        self.max_distance= 3*np.linalg.norm(self.target_position - self.start_position)
        self.previous_position=self.start_position

    def _setup_flight(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        #state = self.client.getMultirotorState()
        #if state.landed_state == airsim.LandedState.Landed:
        x,y,z=float(self.start_position[0]),float(self.start_position[1]),float(self.start_position[2])
        self.client.moveToPositionAsync(x,y,z, 0.75*self.max_velocity).join()

        
    def reset(self, **kwargs):
        if 'seed' in kwargs:
            seed = kwargs['seed']
        
        # Reset environment and set to start position
        self.sim_info['collision_counter']=0
        self.sim_info['step_counter']=0
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

    
    def _get_obs(self):
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
            "drone_position": current_position,
            "orientation": drone_orientation,  # roll, pitch, yaw
            "linear_velocity": current_velocity,
            "relative_yaw": np.array([relative_yaw]),  # Yaw difference to align with the target
        }

    def parse_lidarData(self, data):
        # reshape array of floats to array of [X,Y,Z]
        points = np.array(data.point_cloud, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0]/3), 3))
        return points

    def _show_sim_infos(self):
        #TODO
        print("Showing simulation information")
        

    def _compute_reward(self,obs):
        done=False
        # Extract required observation data
        drone_position = obs["drone_position"]
        previous_position = self.previous_position
        target_position = self.target_position
        lidar_data = self.client.getLidarData()
        
        # Initialize rewards and penalties
        close_target_reward=0.0 #(a): range 0 or 1
        dist_reward=0
        direction_reward=0.0 #(b): range[-1, 1]
        displacement_reward = 0.0 #(c): range 0 or 0.5
        yaw_reward=0 #(d): range 0 or [2/3,1]
        foward_reward =0.0 #(e): range [-1, 1]
        lidar_reward = 0 #(f): range[0,1] based on the percentage of distances above the threshold
        lidar_penalty = 0 # (g): range[0,1] based on the percentage of distances below the threshold
        collision_penalty = 0
        
        ##weights for each component of reward based on distance to target
        displ_weight=0.2
        yaw_weight=0.1
        forward_weight=0.05
        lidar_pen_weight=0.3

        # ---- 1. Distance Based ----
        #(a)----- proximity reward based on current distance from target
        dist_target = np.linalg.norm(target_position-drone_position)
        dist_reward=-dist_target
        if dist_target < 3:
            close_target_reward=1e2
            if dist_target < 1:
                close_target_reward=1e6
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
        displacement_reward= displ_weight*dist_target if delta_displacement > 0 else 0
        
        #(d)-----  Drone yaw alignment reward, measures if the drone is aligned with the target
        relative_yaw=obs["relative_yaw"]
        #yaw_reward= (np.pi-abs(relative_yaw))/np.pi if -(np.pi/3) <= relative_yaw <= (np.pi/3) else 0
        yaw_reward= yaw_weight*dist_target if -(np.pi/3) <= relative_yaw <= (np.pi/3) else 0
    
        #(e)-----  This reward tries to incentivize the drone movement in its camera-facing position
        orientation = obs["orientation"]  # Assuming the drone's orientation is provided in roll, pitch, yaw
        yaw = orientation[2]  # Extract yaw in radians
        facing_direction = np.array([np.cos(yaw), np.sin(yaw), 0])  # Facing direction vector in X-Y plane
        # Calculate movement direction based on positions
        movement_vector_norm = np.zeros_like(movement_vector)
        if np.linalg.norm(movement_vector) > 1e-6:  # Avoid division by zero
            movement_vector_norm = movement_vector/np.linalg.norm(movement_vector)  # Normalize
            
        # Check alignment between movement and facing direction
        dot_prod_foward = np.dot(facing_direction[:2], movement_vector_norm[:2])  # Consider only X-Y plane for alignment
        foward_reward = forward_weight*dist_target if dot_prod_foward > 0.5 else 0


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
                    lidar_reward = 50*(above_mean_count / len(front_distances))
                # (g)------ Penalty for proximity to obstacles        
                if average_front_distance <= self.min_dist_obs * 1.1:                    
                    below_mean_count = np.sum(front_distances < average_front_distance)
                    # Calculate the percentage of distances above the mean
                    #lidar_penalty = 50*(below_mean_count / len(front_distances))
                    lidar_penalty = -lidar_pen_weight*dist_target
        

         # ---- 3. Collision Penalty ----
        collision = self.client.simGetCollisionInfo().has_collided
        if collision:
            self.sim_info['collision_counter']+=1
            self.collision_counter+=1
            collision_penalty = -(0.5*dist_target+1e2)*np.log(self.collision_counter)
            

        
        total_reward = ( 
            close_target_reward+
            dist_reward+
            displacement_reward+
            #yaw_reward+
            #foward_reward+
            lidar_penalty+
            collision_penalty
        )
        
            
        total_reward = float(total_reward)
        #update the simulation information for logs or truncate the episode
        self.sim_info['reward_components']=[close_target_reward,dist_reward,dist_reward,displacement_reward,lidar_penalty,collision_penalty]
        self.sim_info['reward_names']=['close_target_reward','dist_reward','displacement_reward','lidar_penalty','collision_penalty']
        self.sim_info['dist_to_target']=dist_target

        # Update state for next reward calculation
        self.previous_position = drone_position

        return total_reward, done
    
    def _do_action(self,action):
        # Scale action values to the desired range
        vx = float(action[0] *0.5*self.max_velocity)
        vy = float(action[1] *0.5*self.max_velocity)
        vz = float(action[2] *0.05* self.max_velocity)
        delta_yaw_target_rad = float(action[3]) * np.radians(self.max_yaw_rate)  # Max yaw rate change in degrees per step
        # Convert yaw_angle to degrees for AirSim
        new_yaw_target_rad = self.drone_orientation[2] + delta_yaw_target_rad
        new_yaw_target_rad = (new_yaw_target_rad + np.pi) % (2 * np.pi) - np.pi
        yaw_action = np.degrees(new_yaw_target_rad)#gets the current drone orientation and sums the delta rad to align with the target an then transforms to degree to use in airsim
        print(f"Yaw:{yaw_action}")
        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=float(yaw_action))
        #self.client.moveByVelocityAsync(vx, vy, vz, duration=1, yaw_mode=0).join()
        self.client.moveByVelocityAsync(vx,vy,vz,duration=0.5).join()
        #time.sleep(0.5)
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
        self.sim_info["step_counter"]+=1
        step_count=self.sim_info["step_counter"]
        self._do_action(action)
        # Get new state and compute reward
        obs = self._get_obs()
        print(f"Step: {step_count}")
        reward, done_point = self._compute_reward(obs)
        
        if done_point:
            done_path=self.update_points()
        
        terminated = True if done_path else False # Finish the current episode in case all the points in the path were used
        # Truncate if reward drops below threshold
        truncated = True if self.sim_info["collision_counter"] > 200 or  self.sim_info["dist_to_target"] > self.max_distance  else False
        if truncated:
            print("Episode Truncated")
        return obs, reward, terminated, truncated, {}

    def close(self):
        print("Close Called")
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
