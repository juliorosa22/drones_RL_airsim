import gymnasium as gym
import numpy as np
from gymnasium import spaces
import airsim
from airgym.envs.airsim_env import AirSimEnv
from scipy.spatial.transform import Rotation as R
from path_handler import PathHandler
import time
import os
import csv
MAX_STEPS_PER_EPISODE=500
LIDAR_FEAT_SIZE=300
class QuadAirSimDroneEnv(AirSimEnv):
    
    def __init__(self, ip_address, img_shape, path_file,csv_file_log,max_yaw_rate=45,min_dist=2,max_velocity=10):
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
        
        self.previous_position=np.zeros(3, dtype=np.float32)
        self.min_dist_obs=min_dist #minimum distance threshold to penalize the drone when getting closer to obstacles
        self.previous_velocity = np.zeros(3, dtype=np.float32)
        self.update_path()
        self._setup_flight()
        

        # Flatten the observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9+LIDAR_FEAT_SIZE,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(7)

        #Information used to decide when stop the episode due agent random behavior
        self.sim_info = {
            "collision_counter":0,
            "step_counter":0,
            "dist_to_target":0.0,
            "Truncate": False,
            "reward_components":[],
            "reward_names":[],
            }
        
        #Log handlers
        self.csv_file_path = csv_file_log
        self._initialize_csv()
        self.episode_log = []  # Store all steps during the episode
        self.episode_id = 0  # Unique identifier for the episode where the agent reaches at least one target point

    #-------PATH Handler Functions      
    #This function gets the new path for the next episode training
    def update_path(self):
        path = self.path_handler.get_next_path()
        print(f"Next Path selected:{path['path_id']}")
        point=self.path_handler.get_next_point()
        self.target_position = np.array(point['target_position'],dtype=np.float32)
        self.start_position = np.array(point['start_position'],dtype=np.float32)
        self.max_distance= 3*np.linalg.norm(self.target_position - self.start_position)
        self.previous_position=self.start_position

    #this function gets the next target point in the current path for the drone reach
    #return True if the current path is finished or False if there is still point in path
    def update_points(self):
        new_points=self.path_handler.get_next_point()    
        if new_points is None:
            return True
        self.start_position = np.array(new_points['start_position'],dtype=np.float32)
        self.target_position = np.array(new_points['target_position'],dtype=np.float32)
        return False
    
    #### Auxiliary simulation functions
    #setup the drone before the training of the agent begins
    def _setup_flight(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        #state = self.client.getMultirotorState()
        #if state.landed_state == airsim.LandedState.Landed:
        x,y,z=float(self.start_position[0]),float(self.start_position[1]),float(self.start_position[2])
        self.client.moveToPositionAsync(x,y,z, 0.75*self.max_velocity).join()

    #Display Simulation Info
    def _show_sim_infos(self):
        
        reward_str=''
        rewards=self.sim_info["reward_components"]
        if rewards[0]>0:
           reward_str+=f"target close:{rewards[0]}|"
        reward_str+=f"dist rwd:{rewards[1]}|"
        reward_str+=f"displ rwd:{rewards[2]}|"
        reward_str+=f"lidar pen:{rewards[3]}|"
        reward_str+=f"colli pen:{rewards[4]}|"
        print(reward_str)
    
    #Logger functions to track the success of training
    def _initialize_csv(self):

        # Create the CSV file with headers if it doesn't exist
        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "Episode_ID","Path_ID", "Target_ID", 
                    "Drone_Position_X", "Drone_Position_Y", "Drone_Position_Z", "Reward"
                ])

    def _log_step(self, obs, reward):
        """Log data for the current step into the episode log."""
        path_id = self.path_handler.paths[self.path_handler.current_path_index]["path_id"]
        target_id = self.path_handler.current_point_index
        drone_x, drone_y, drone_z = obs["drone_position"]

        # Use the current `episode_id` if it's set; otherwise, use 0 (target not yet reached)
        self.episode_log.append({
            "Episode_ID": self.episode_id or 0,  # 0 until the target is reached
            "Path_ID": path_id,
            "Target_ID": target_id,
            "Drone_Position_X": drone_x,
            "Drone_Position_Y": drone_y,
            "Drone_Position_Z": drone_z,
            "Reward": reward
        })
        #update the next episode id
        
        #writes the current episode where the drone reached the target into the csv file
    
    def _log_episode_to_csv(self):
        """Write all logged steps in the episode to the CSV file."""
        # Update all rows in the episode log to use the finalized
        for row in self.episode_log:
            row["Episode_ID"] = self.episode_id

        with open(self.csv_file_path, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=[
                "Episode_ID","Path_ID", "Target_ID", 
                "Drone_Position_X", "Drone_Position_Y", "Drone_Position_Z", "Reward"
            ])
            writer.writerows(self.episode_log)
        self.episode_log = []  # Clear the log for the next episode

    ###### Auxiliary Component Reward Functions
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

    def parse_lidarData(self, data):
        # reshape array of floats to array of [X,Y,Z]
        points = np.array(data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0]/3), 3))
        return points

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


    ### Fundamental Functions for the Enviroment and Agent

    def __del__(self):
        print("del called")
        self.client.reset()

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

                
        drone_orientation=np.array(orientation_euler,dtype=np.float32)
        
        lidar_data = self.client.getLidarData()
        lidar_points=np.zeros(LIDAR_FEAT_SIZE,dtype=np.float32)
        size_lidar_read=len(lidar_data.point_cloud)
        if size_lidar_read > LIDAR_FEAT_SIZE:        
            lidar_points[:] = np.array(lidar_data.point_cloud[:LIDAR_FEAT_SIZE], dtype=np.float32)
        elif size_lidar_read > 3:
            lidar_points[:size_lidar_read]=np.array(lidar_data.point_cloud[:], dtype=np.float32)
        return np.concatenate([
            current_position,  # Position (x, y, z)
            drone_orientation,  # Orientation (roll, pitch, yaw)
            current_velocity,  # Linear velocity (vx, vy, vz)
            lidar_points,
        ])

    def _compute_reward(self,obs):
        done=False
        # Extract required observation data
        drone_position = obs["drone_position"]
        previous_position = self.previous_position
        target_position = self.target_position
        #lidar_data = self.client.getLidarData()
        
        # Initialize rewards and penalties
        close_target_reward=0.0 #(a): range 0 or 1
        dist_reward=0
        displacement_reward = 0.0 #(c): range 0 or 0.5
        lidar_penalty = 0.0 # (g): range[0,1] based on the percentage of distances below the threshold
        collision_penalty = 0.0
        
        ##weights for each component of reward based on distance to target
        displ_weight=0.1
        lidar_pen_weight=0.25

        # ---- 1. Distance Based ----
        #(a)----- proximity reward based on current distance from target
        dist_target = np.linalg.norm(target_position-drone_position)
        dist_reward=-dist_target
        if dist_target < 5:
            close_target_reward=1e3
            if dist_target < 3:
                close_target_reward=1e6
                done=True
        else:
            close_target_reward=0

         
        #(c)-----  Reward based on the displacement
        previous_distance_to_target = np.linalg.norm(previous_position - target_position)
        current_distance_to_target = np.linalg.norm(drone_position - target_position)
        delta_displacement= previous_distance_to_target - current_distance_to_target
        displacement_reward= displ_weight*dist_target if delta_displacement > 0 else 0
        
             

        # ---- 2. LiDAR Based ----
        # Parse LiDAR data into 3D point cloud
        lidar_points=obs["lidar_points"]
        check_lidar=np.any(lidar_points!=0)
        lidar_points=self.parse_lidarData(lidar_points)
        if check_lidar:
            distances_to_points = np.linalg.norm(lidar_points, axis=1)
            average_distance = np.mean(distances_to_points)                
            #penalty for proximity with obstacles
            if average_distance < self.min_dist_obs * 1.1:
                #print("Close to obstacle")                   
                lidar_penalty = -lidar_pen_weight*dist_target
         # ---- 3. Collision Penalty ----
        collision = self.client.simGetCollisionInfo().has_collided
        if collision:
            
            self.sim_info['collision_counter']+=1
            collision_counter=self.sim_info['collision_counter']
            print(f"collision: {collision_counter}")
            collision_penalty = -(0.5*dist_target+1e2)*np.log(collision_counter)
            

        
        time_penalty = -0.01 * self.sim_info['step_counter']

        total_reward = ( 
            close_target_reward+
            dist_reward+
            displacement_reward+
            lidar_penalty+
            collision_penalty+
            time_penalty
        )
        
            
        total_reward = float(total_reward)
        #update the simulation information for logs or truncate the episode
        self.sim_info['reward_components']=[close_target_reward,dist_reward,displacement_reward,lidar_penalty,collision_penalty,time_penalty]
        self.sim_info['reward_names']=['close_target_reward','dist_reward','displacement_reward','lidar_penalty','collision_penalty','time_penalty']
        self.sim_info['dist_to_target']=dist_target

        # Update state for next reward calculation
        self.previous_position = drone_position

        return total_reward, done
    
    def interpret_action(self, action):
        """
        Interpret discrete action into velocity and yaw commands.
        Actions:
        0: Hover
        1: Move Forward
        2: Move Backward
        3: Move Right
        4: Move Left
        5: Rotate Right (90 degrees)
        6: Rotate Left (90 degrees)
        """
        angle=45
        vx, vy, vz = 0.0, 0.0, 0.0
        yaw_action = None

        if action == 0:  # Hover
            pass  # No movement
        elif action == 1:  # Move Forward
            vx = -self.max_velocity
            vz=self.max_velocity
        elif action == 2:  # Move Backward
            vx = self.max_velocity
            vz=self.max_velocity
        elif action == 3:  # Move Right
            vy = -self.max_velocity
            vz=self.max_velocity
        elif action == 4:  # Move Left
            vy = self.max_velocity
            vz=self.max_velocity
        elif action == 5:  # Rotate Right (90 degrees)
            yaw_action= angle
            #yaw_action = np.degrees(self.drone_orientation[2] + np.pi / 2)  # Add 90 degrees
        elif action == 6:  # Rotate Left (90 degrees)
            yaw_action=-angle
            #yaw_action = np.degrees(self.drone_orientation[2] - np.pi / 2)  # Subtract 90 degrees

        return vx, vy, vz, yaw_action


    def _do_action(self,action):
        vx, vy, vz, yaw_action = self.interpret_action(action)
        act_time=0.5
        if yaw_action is not None:  # Rotation
            self.client.rotateByYawRateAsync(yaw_rate=yaw_action,duration=act_time).join()
        else:  # Movement in the XY plane
            self.client.moveByVelocityAsync(0.3*vx, 0.3*vy, 0*vz,duration=act_time).join()
    
    def step(self, action):
        done_point,done_path=False, False
        self.sim_info["step_counter"]+=1
        step_count=self.sim_info["step_counter"]
        self._do_action(action)
        # Get new state and compute reward
        vec_obs=self._get_obs()
        obs = self.parse_flattened_obs(vec_obs)
        
        reward, done_point = self._compute_reward(obs)
        #insert the current observation and reward into the current list in case the drone reaches the target in the current episode
        self._log_step(obs,reward)
        #it means that the drone reached the target position
        if done_point:
            #writes the current log into csv file
            self._log_episode_to_csv()
            done_path=self.update_points()
        
        terminated = True if done_path else False # Finish the current episode in case all the points in the path were used
        # Truncate if reward drops below threshold
        truncated=False
        if self.sim_info["collision_counter"] > 100 or step_count > 500 or  self.sim_info["dist_to_target"] > self.max_distance:
            truncated=True
            print("Episode Truncated")
        
        #updates the current episode id
        if terminated or truncated:
            self.episode_id+=1
        
            
            
        return vec_obs, reward, terminated, truncated, {}

    def close(self):
        print("Close Called")
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
