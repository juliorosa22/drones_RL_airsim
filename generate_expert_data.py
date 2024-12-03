import csv
import airsim
from path_handler import PathHandler
from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd
import time
LIDAR_FEAT_SIZE=600
class ExpertDataCollector:
    def __init__(self, ip_address, csv_file,path_file,enabled_drone=True):
        
        self.csv_file = csv_file
        self.path_handler=PathHandler(path_file)
        self.min_dist=2
        self.max_vel=10
        self.target_pos=np.zeros(3,dtype=np.float32)
        self.client = airsim.MultirotorClient(ip=ip_address)
        self.expert_points=[]
        self.expert_actions=[]
        
        if enabled_drone:
            self.client.confirmConnection()
    

    def parse_flattened_obs(self,flattened_obs):
    
        # Indices for splitting the flattened observation
        position_end = 3  # First 3 elements are drone position
        orientation_end = position_end + 3  # Next 3 elements are orientation (roll, pitch, yaw)
        velocity_end = orientation_end + 3  # Next 3 elements are linear velocity
        

        # Parse data
        drone_position = flattened_obs[:position_end]
        orientation = flattened_obs[position_end:orientation_end]
        target_position = flattened_obs[orientation_end:velocity_end]
        lidar_positions=flattened_obs[velocity_end:]

        # Construct the dictionary
        parsed_obs = {
            "drone_position": drone_position,
            "orientation": orientation,
            "target_position": target_position,
            "lidar_points":lidar_positions
        }

        return parsed_obs
    
    
    def parse_lidarData(self, data):
            # reshape array of floats to array of [X,Y,Z]
            points = np.array(data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0]/3), 3))
            return points

    def collect_data(self, state, action):
        with open(self.csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([state, action])

    def _setup_flight(self,position):
            self.client.reset()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            
            #state = self.client.getMultirotorState()
            #if state.landed_state == airsim.LandedState.Landed:
            x,y,z=float(position[0]),float(position[1]),float(position[2])
            self.client.moveToPositionAsync(x,y,z, 0.75*5).join()
            time.sleep(2)

    def safe_move(self,vel_vec,z_fixed,act_time):
        vx = vel_vec[0]
        vy=vel_vec[1]
        #self.client.moveToPositionAsync(float(drone_position[0]),float(drone_position[1]),float(z_fixed),0.75*self.max_vel)
        self.client.moveByVelocityAsync(float(vx), float(vy), 0,duration=act_time).join()
        drone_state = self.client.getMultirotorState()
        state_pos = drone_state.kinematics_estimated.position
        current_z = float(state_pos.z_val)
        while current_z < 1.3*z_fixed:
            #print("correcting z")
            self.client.moveByVelocityAsync(0, 0, 0.03*self.max_vel,duration=act_time).join()
            drone_state = self.client.getMultirotorState()
            state_pos = drone_state.kinematics_estimated.position
            current_z = float(state_pos.z_val)
            #print(f"{current_z}")
    
    
    def check_lidar_safe(self,obs,drone_ground,avoid_obs):
        z_fixed=-5
        obs_d=self.parse_flattened_obs(np.copy(obs))
        lidar_points = obs_d["lidar_points"]
        lidar_points = self.parse_lidarData(lidar_points)
        distances_to_points = np.linalg.norm(lidar_points, axis=1)
        average_distance = np.mean(distances_to_points)
        collision = self.client.simGetCollisionInfo().has_collided
        action_change=False
        count=0
        change="backward"
        while (average_distance < self.min_dist*1.5 or collision) and count < 5:
            count+=1
            action_change=True
            #print(f"Close to target: {average_distance}")
            id,action=self.get_action(change)
            drone_ground+=action[:-1]
            x,y,_=drone_ground[:]
            self.client.moveToPositionAsync(float(x),float(y),z_fixed,5).join()
            self.expert_points.append(obs)
            self.expert_actions.append(id)            
            
            
            # id,action=self.get_action(avoid_obs)
            # #print(f"id:{id} | action:{action}")
            # drone_ground+=action[:-1]
            # x,y,_=drone_ground[:]
            # self.client.moveToPositionAsync(float(x),float(y),z_fixed,5).join()
            
            obs=self._get_obs()
            obs_d = self.parse_flattened_obs(np.copy(obs))
            distances_to_points = np.linalg.norm(self.parse_lidarData(obs_d["lidar_points"]),axis=1)
            average_distance = np.mean(distances_to_points)
            collision = self.client.simGetCollisionInfo().has_collided
            change = avoid_obs if change=="backward" else "backward"


        return action_change,drone_ground            

    def run_new_expert(self):
        # Example: Scripted expert behavior
        act_time=0.5
        perc_vel=0.3
        z_fixed=-5.0
        move_away='left'
        move_close='right'
        allow_collision=False
        
        collision_count=0
        hover_count=0
        for i in range(len(self.path_handler.paths)):
            path=self.path_handler.get_next_path()
            point=self.path_handler.get_next_point()
            self._setup_flight(point['start_position'])
            
            while point is not None:
                start_pos=np.array(point['start_position'],dtype=np.float32)
                target_pos=np.array(point['target_position'],dtype=np.float32)
                drone_pos_ground=np.copy(start_pos)
                print(f"start:{start_pos} | target: {target_pos}")
                self.target_pos=np.copy(target_pos)
                while abs(drone_pos_ground[0] - target_pos[0]) > 1:  
                    #print("1 While")
                    obs=self._get_obs()
                    
                    id,action=self.get_action("forward")
                    action_changed,drone_pos_ground=self.check_lidar_safe(obs,np.copy(drone_pos_ground),move_away)
                    if action_changed:
                        print("change in action")
                        break                    
                    
                    drone_pos_ground+=action[:-1]
                    x,y,_= drone_pos_ground[:]
                    #print(f"x:{x} | y: {y}")
                    self.client.moveToPositionAsync(float(x),float(y),z_fixed,5).join()
                    self.expert_points.append(obs)
                    self.expert_actions.append(id)
                        
                        
                #Movement to deslocate from the current initial position
                delta_pos=10 if move_away =='left' else -10
                        
                    
                if move_away=="right":
                    while drone_pos_ground[1] > (start_pos[1]+delta_pos):
                        #print("4 while")
                        obs=self._get_obs()
                        #obs=self.parse_flattened_obs(np.copy(obs_vec))
                        
                        id,action=self.get_action(move_away)
                        action_changed,drone_pos_ground=self.check_lidar_safe(obs,drone_pos_ground,move_away)
                        if not action_changed:
                            drone_pos_ground+=action[:-1]
                            x,y,_= drone_pos_ground[:]
                            self.client.moveToPositionAsync(float(x),float(y),z_fixed,5).join()
                            self.expert_points.append(obs)
                            self.expert_actions.append(id)
                    
                    
                else:
                    while drone_pos_ground[1] < (start_pos[1]+delta_pos):
                        #print("4 while")
                        obs=self._get_obs()
                        
                        
                        id,action=self.get_action(move_away)
                        action_changed,drone_pos_ground=self.check_lidar_safe(obs,drone_pos_ground,move_away)
                        if not action_changed:
                            drone_pos_ground+=action[:-1]
                            x,y,_= drone_pos_ground[:]
                            self.client.moveToPositionAsync(float(x),float(y),z_fixed,5).join()
                            self.expert_points.append(obs)
                            self.expert_actions.append(id)
                    move_away='left'
                    move_close='right'



                # movement to approximate to next target position
                while drone_pos_ground[0] > target_pos[0] :
                    #print("3 while")
                    obs=self._get_obs()
                    #obs=self.parse_flattened_obs(np.copy(obs_vec))
                    
                    id,action=self.get_action("forward")
                    action=np.array(action,dtype=np.float32)
                    action_changed,drone_pos_ground=self.check_lidar_safe(obs,drone_pos_ground,move_close)
                    if not action_changed:
                        drone_pos_ground+=action[:-1]
                        x,y,_= drone_pos_ground[:]
                        self.client.moveToPositionAsync(float(x),float(y),z_fixed,5).join()
                        #self.expert_points.append(obs_vec)
                        #self.expert_actions.append(id)
                    
                    

                ##movement to close the drone y componet to target y component
                if move_close=="right":
                    while drone_pos_ground[1] > target_pos[1]:
                        #print("5 while")
                        obs=self._get_obs()
                        #obs=self.parse_flattened_obs(np.copy(obs_vec))
                        
                        id,action=self.get_action(move_close)
                        action=np.array(action,dtype=np.float32)
                        #v_vec=action[:-1]*self.max_vel*perc_vel
                        action_changed,drone_pos_ground=self.check_lidar_safe(obs,drone_pos_ground,move_close)
                        if not action_changed:
                            drone_pos_ground+=action[:-1]
                            x,y,_= drone_pos_ground[:]
                            self.client.moveToPositionAsync(float(x),float(y),z_fixed,5).join()
                            self.expert_points.append(obs)
                            self.expert_actions.append(id)
                    move_away='right'
                    move_close='left'
                    
                else:
                    while drone_pos_ground[1] < target_pos[1]:
                        #print("6 while")
                        obs=self._get_obs()
                        #obs=self.parse_flattened_obs(np.copy(obs_vec))
                        
                        id,action=self.get_action(move_close)
                        action=np.array(action,dtype=np.float32)
                       
                        action_changed,drone_pos_ground=self.check_lidar_safe(obs,drone_pos_ground,move_close)
                        if not action_changed:
                            drone_pos_ground+=action[:-1]
                            x,y,_= drone_pos_ground[:]
                            self.client.moveToPositionAsync(float(x),float(y),z_fixed,5).join()
                            self.expert_points.append(obs)
                            self.expert_actions.append(id)
                    move_away='left'
                    move_close='right'
                ##hover in the actual target position before the next point
                while hover_count < 10:
                    hover_count+=1
                    obs=self._get_obs()
                    #obs=self.parse_flattened_obs(np.copy(obs_vec))
                    
                    id,action=self.get_action("hover")
                    action_changed,drone_pos_ground=self.check_lidar_safe(obs,drone_pos_ground,move_close)
                    if not action_changed:
                        drone_pos_ground+=action[:-1]
                        x,y,_= drone_pos_ground[:]
                        self.client.moveToPositionAsync(float(x),float(y),z_fixed,5).join()
                        #self.expert_points.append(obs)
                        #self.expert_actions.append(id)

                hover_count=0
                collision_count=0
                #allow_collision = not allow_collision
            

                point=self.path_handler.get_next_point()
            
            self.client.reset()

        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        
        data = {
            "observation": [xp_obs.tolist() for xp_obs in self.expert_points],  # Convert np.array to list
            "action": self.expert_actions
        }
        print(f"size action:{len(self.expert_actions)} | size obs:{len(self.expert_points)}")
        df = pd.DataFrame(data)
        # Save to CSV
        df.to_csv(self.csv_file, index=False)
        print(f"Expert data saved to {self.csv_file}")
        act_ids=np.array(self.expert_actions)
        result ={i: np.sum(act_ids == i) for i in range(7)}
        print(f"Counts for each id: {result}")


    def move_position(self,position,vel,act_time,error=0):
        next_position=vel[:-1]*act_time+position+error
        #print(f"position:{position}, next:{next_position}")
        return next_position

    def get_action(self,command):
        if command == "forward":
            return 1,np.array([-1,0,0,0],dtype=np.float32)
        elif command == "backward":
            return 2,np.array([1,0,0,0],dtype=np.float32)
        elif command == "right":
            return 3,np.array([0,-1,0,0],dtype=np.float32)
        elif command == "left":
            return 4,np.array([0,1,0,0],dtype=np.float32)
        elif command == "rot_right":
            return 5,np.array([0,0,0,1],np.float32)
        elif command == "rot_left":
            return 6,np.array([0,0,0,-1],dtype=np.float32)
        elif command == "hover":
            return 0,np.array([0,0,0,0],dtype=np.float32)
        else:
            return None

    def get_euler_angles(self, quaternion):
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        rotation = R.from_quat([quaternion.x_val, quaternion.y_val, quaternion.z_val, quaternion.w_val])
        return rotation.as_euler('xyz', degrees=False)  # Returns roll, pitch, yaw
   
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
        vec_obs= np.concatenate([
            current_position,  # Position (x, y, z)
            drone_orientation,  # Orientation (roll, pitch, yaw)
            self.target_pos,  # Linear velocity (vx, vy, vz)
            lidar_points,
        ])
        if vec_obs.shape[0] != (9+LIDAR_FEAT_SIZE):
            print(f"obs_shape:{vec_obs.shape}")
            print("ERROR")
        return vec_obs

    def _generate_expert_points(self):
        # Define how the expert acts given the state
        vel_step=1.0
        move_away='left'
        move_close='right'
        expert_points=[[] for i in range(len(self.path_handler.paths))]
        expert_actions=[[] for i in range(len(self.path_handler.paths))]
        hover_count=0
        for i in range(len(self.path_handler.paths)):
            path=self.path_handler.get_next_path()
            point=self.path_handler.get_next_point()
            
            while point is not None:
                start_pos=np.array(point['start_position'],dtype=np.float32)
                target_pos=np.array(point['target_position'],dtype=np.float32)
                drone_pos=np.copy(start_pos)
                if self.path_handler.current_point_index == 0:
                    while abs(drone_pos[0] - (start_pos[0]-5)) > 0:  
                        #print("1 While")
                        #if drone_pos[0]>-100:
                        #    print(f"{abs(drone_pos[0] - (start_pos[0]-5))}")
                        #    print(f"drone_pos:{drone_pos[0]} | tgt:{start_pos[0]-5}")
                        id,action=self.get_action("forward")
                        #action=np.array(action,dtype=np.float32)
                        
                        expert_points[i].append((np.copy(drone_pos),np.copy(target_pos)))
                        expert_actions[i].append(id)
                        #self.collect_data(drone_pos,action)
                        drone_pos+=action[:-1]#self.move_position(drone_pos,action*vel_step,1)
                        
                #Movement to deslocate from the current initial position
                delta_pos=10 if move_away =='left' else -10
                        
                while abs(drone_pos[1] - (start_pos[1]+delta_pos)) > 0:
                    #print("2 while")
                    #print(f"{abs(drone_pos[1] - start_pos[1]+delta_pos)}")
                    id,action=self.get_action(move_away)
                    #action=np.array(action,dtype=np.float32)
                    #self.collect_data(drone_pos,action)
                    expert_points[i].append((np.copy(drone_pos),np.copy(target_pos)))
                    expert_actions[i].append(id)
                    drone_pos+=action[:-1]#self.move_position(drone_pos,action*vel_step,1)
                # movement to approximate to next target position
                while abs(drone_pos[0] - target_pos[0]) > 0:
                    #print("3 while")                      
                    id,action=self.get_action("forward")
                    #action=np.array(action,dtype=np.float32)
                    #self.collect_data(drone_pos,action)
                    expert_points[i].append((np.copy(drone_pos),np.copy(target_pos)))
                    expert_actions[i].append(id)
                    drone_pos+=action[:-1]#self.move_position(drone_pos,action*vel_step,1)

                ##movement to close the drone y componet to target y component
                while abs(drone_pos[1] - target_pos[1]) > 0:
                    #print("4 while")                      
                    id,action=self.get_action(move_close)
                    #action=np.array(action,dtype=np.float32)
                    expert_points[i].append((np.copy(drone_pos),np.copy(target_pos)))
                    expert_actions[i].append(id)
                    drone_pos+=action[:-1]#self.move_position(drone_pos,action*vel_step,1)
                    

                
                if move_close =='right':                            
                    move_away='right'
                    move_close='left'
                else:
                    move_away='left'
                    move_close='right'

                point=self.path_handler.get_next_point()
                #action=self.get_action("rot_right")    
                
                

        
        return expert_points,expert_actions  # Example action

    def generate_expert_obs(self):
        xp_points,xp_actions=self._generate_expert_points() 
        xp_obs=[]
        xp_act=[]
        for path,action_path in zip(xp_points,xp_actions):
            self._setup_flight(path[0])
            #print(f"len:{len(path)}")
            for position,target,action in zip(path,action_path):
                #print(f"moving to position {position}")
                x,y,z=float(position[0]),float(position[1]),float(position[2])
                self.client.moveToPositionAsync(x,y,z, 5).join()
                obs=self._get_obs(target)
                print(obs.shape)
                xp_obs.append(obs)
                xp_act.append(action)
                #print(f"Observation:{obs} | id :{action}")
                #time.sleep(0.5)
            
            self.client.reset()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        data = {
            "observation": [obs.tolist() for obs in xp_obs],  # Convert np.array to list
            "action": xp_act
        }
    
        df = pd.DataFrame(data)
        # Save to CSV
        df.to_csv(self.csv_file, index=False)
        print(f"Expert data saved to {self.csv_file}")

    def read_xp_data(self):
        # Load expert data
        df = pd.read_csv(self.csv_file)
        print(list(df))
        df["observation"] = df["observation"].apply(lambda x: np.fromstring(x.strip("[]"), sep=","))

        # Convert to NumPy arrays for training
        #obs=df["observation"].values
       
        observations = np.stack(df["observation"].values)
        actions = df["action"].values
        return observations,actions



