import gymnasium as gym
from stable_baselines3 import PPO, DQN

# Replace with your evaluation environment class
from airgym.envs.quad_drone_env import QuadAirSimDroneEnv

# Load evaluation environment
eval_env = QuadAirSimDroneEnv(ip_address="127.0.0.1", img_shape=(64, 64, 1) ,path_file='training_paths.json',csv_file_log='improved_ppo_episodes.csv')

# Load trained model
model = PPO.load("improved_ppo_final_model")  # Replace "ppo_drone_agent" with your model file name
#model = DQN.load("improved_dqn_final_model")  # Replace "ppo_drone_agent" with your model file name

# Evaluation parameters
n_eval_episodes = 60
max_steps = 500

# Metrics for evaluation
episode_rewards = []
successes = 0
best_collison_score=[]
for episode in range(n_eval_episodes):
    obs,_ = eval_env.reset()
    episode_reward = 0
    for step in range(max_steps):
        # Get the action from the trained model
        action, _states = model.predict(obs, deterministic=False)

        # Take a step in the environment
        obs, reward, terminated, truncated, info = eval_env.step(action)
        episode_reward += reward

        # Check if the episode has ended
        if terminated or truncated:
            if terminated:
                best_collison_score.append(eval_env.sim_info['collision_counter'])
                successes += 1  # Count successful episodes
            break

    episode_rewards.append(episode_reward)
    print(f"Episode {episode + 1}/{n_eval_episodes} - Reward: {episode_reward}")

# Print evaluation results
print(f"Average Reward: {sum(episode_rewards) / len(episode_rewards)}")
print(f"Success Rate: {successes / n_eval_episodes * 100}%")
