import gymnasium as gym
from stable_baselines3 import PPO, DQN
import numpy as np
import pandas as pd
# Replace with your evaluation environment class
from airgym.envs.quad_drone_env import QuadAirSimDroneEnv

# Load evaluation environment
eval_env = QuadAirSimDroneEnv(ip_address="127.0.0.1", img_shape=(64, 64, 1) ,path_file='training_paths.json',csv_file_log='improved_ppo_episodes.csv')

# Load trained model
model_list = [PPO.load("improved_ppo_final_model") ,DQN.load("improved_dqn_final_model")] # Replace "ppo_drone_agent" with your model file name
names=['PPO','DQN']
metrics = []
# Evaluation parameters
for model,name in zip(model_list,names):
    n_eval_episodes = 6
    max_steps = 50
    #eval_env.client.simPause(True)
    # Metrics for evaluation
    episode_rewards = []
    successes = 0
    best_collision_scores=[]
    steps_per_episode=[]
    print(f"Evaluation for :{name}")
    for episode in range(n_eval_episodes):
        obs,_ = eval_env.reset()
        episode_reward = 0
        print(f"Episode:{episode + 1}")
        for step in range(max_steps):
            #print(f"Step:{step}")
            # Get the action from the trained model
            action, _states = model.predict(obs, deterministic=False)

            # Take a step in the environment
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward

            # Check if the episode has ended
            if terminated or truncated:
                
                
                if terminated:
                    best_collision_scores.append(eval_env.sim_info['collision_counter'])
                    steps_per_episode.append(step+1)
                    successes += 1  # Count successful episodes
                break
            
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{n_eval_episodes} - Reward: {episode_reward}")

 # Calculate metrics
    
    rewards=np.array(episode_rewards)
    avg_reward = rewards.mean()
    max_reward = np.max(rewards)
    success_rate = successes / n_eval_episodes * 100
    collisions = np.array(best_collision_scores) if len(best_collision_scores) > 0 else 0.0
    min_collision = np.min(collisions)
    avg_steps = np.array(steps_per_episode).mean() if len(steps_per_episode)> 0 else 0.0
    # Print evaluation results
    print(f"Evaluation results for {name}:")
    print(f"Average Reward: {avg_reward}")
    print(f"Max Acc Reward: {max_reward}")
    print(f"Success Rate: {success_rate}%")
    print(f"Min Collision: {min_collision}")
    print(f"Average Success Steps: {avg_steps}")

    # Store metrics for the model
    metrics.append({
        "model": name,
        "avg_rwd": avg_reward,
        "max_episode_rwd":max_reward,
        "success_rate": success_rate,
        "Min_Collision": min_collision,
        "Avg_Steps":avg_steps
    })
        
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("evaluation_metrics.csv", index=False)
print("Metrics saved to 'evaluation_metrics.csv'")

