from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from airgym.envs.quad_drone_env import QuadAirSimDroneEnv


# Initialize the environment with the target position


# Ensure the environment meets the requirements



def train_agent(model_name,tb_log_prefix='tb_log',type_model=1,check_point_prefix='check_point'):
    env = QuadAirSimDroneEnv(ip_address="127.0.0.1", img_shape=(64, 64, 1), path_file='training_paths.json',csv_file_log=model_name+'_episodes.csv')
    check_env(env)

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/checkpoints/',
                                         name_prefix=model_name+check_point_prefix)
    model = None
    if type_model ==1:
        print("PPO Model")
        #model = PPO.load('imitation_'+model_name, env, batch_size=64, learning_rate=1e-3, verbose=1, tensorboard_log="./logs/")
        model = PPO("MlpPolicy", env, batch_size=256, learning_rate=1e-3, verbose=1, tensorboard_log="./logs/")
    else:
        print("DQN Model")
        #model=DQN.load('imitation_'+model_name, env, batch_size=64, learning_rate=1e-3, verbose=1, tensorboard_log="./logs/")        
        model=DQN("MlpPolicy", env, batch_size=256, learning_rate=1e-3, verbose=1, tensorboard_log="./logs/")
    
    model.learn(total_timesteps=(3*1e5), log_interval=1000, tb_log_name=model_name+tb_log_prefix,
            callback=[checkpoint_callback], progress_bar=True)
    model.save(model_name+"_agent")

train_agent('DQN_23fev',type_model=2)
# Set up callbacks

# Evaluation callback to monitor training performance
#eval_callback = EvalCallback(env, best_model_save_path="./logs/best_model",
#                             log_path="./logs/eval_results", eval_freq=100, n_eval_episodes=10, deterministic=True)

# Create a DQN model with MultiInputPolicy and set up TensorBoard
#model = DQN("MlpPolicy", env, batch_size=64, learning_rate=1e-3, verbose=1, tensorboard_log="./logs/")



# Save the final model after training completes

