import gymnasium as gym
from stable_baselines3 import PPO, DQN
import pandas as pd
# Replace with your evaluation environment class
from airgym.envs.quad_drone_env import QuadAirSimDroneEnv

# Load evaluation environment
eval_env = QuadAirSimDroneEnv(ip_address="127.0.0.1", img_shape=(64, 64, 1) ,path_file='training_paths.json',csv_file_log='just_test.csv')

# Load trained model

ppo_model = PPO.load("PPO_22fev25_agent",verbose=1)  # Replace with your PPO model file name
dqn_model = DQN.load("DQN_23fev_agent",verbose=1)  # Replace with your DQN model file name

# Load models
n_eval_episodes = 100
max_steps = 1000

# Function to evaluate a model
def evaluate_model(model, model_name, eval_env, n_eval_episodes, max_steps):
    episode_rewards = []
    success_rates = []
    collision_scores = []
    step_counts = []
    position_vectors = []
    best_episode_index = -1
    best_reward = float('-inf')
    best_positions = []
    best_collision_count = 0
    best_steps = 0

    for episode in range(n_eval_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0
        step_count = 0
        episode_positions = []

        for step in range(max_steps):
            print(f"Step number:{step+1}")
            # Get the action from the trained model
            action, _states = model.predict(obs, deterministic=False)
            print(f"action:{action}")
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            step_count += 1
            position = obs[:3]  # Assuming position is the first 3 elements in observation
            episode_positions.append(position)

            if terminated or truncated:

                if terminated or eval_env.sim_info['done_points_counter']>=1 and eval_env.sim_info["collision_counter"] < 100:
                    success_rates.append(1)  # Mark as success
                else:
                    success_rates.append(0)  # Mark as failure

                collision_scores.append(eval_env.sim_info['collision_counter'])
                
                break

        # Update metrics
        episode_rewards.append(episode_reward)
        step_counts.append(step_count)
        position_vectors.append(episode_positions)

        # Check for the best episode
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_collision_count = eval_env.sim_info['collision_counter']
            best_steps = step_count
            best_positions = episode_positions
            best_episode_index = episode

        print(f"{model_name} - Episode {episode + 1}/{n_eval_episodes} - Reward: {episode_reward}")

    # Calculate summary metrics
    mean_collisions = sum(collision_scores) / len(collision_scores) if len(collision_scores)>0 else 0.0
    success_rate = sum(success_rates) / len(success_rates) * 100 if len(success_rates)>0 else 0.0

    # Save results to summary DataFrame
    summary = pd.DataFrame({
        "Model": [model_name],
        "Best Reward": [best_reward],
        "Best Episode Collisions": [best_collision_count],
        "Best Episode Steps": [best_steps],
        "Mean Collisions": [mean_collisions],
        "Success Rate (%)": [success_rate]
    })

    # Save position vectors of the best episode separately
    best_positions_df = pd.DataFrame(best_positions, columns=["X", "Y", "Z"])
    best_positions_file = f"{model_name}_best_episode_positions_6mar25.csv"
    best_positions_df.to_csv(best_positions_file, index=False)

    print(f"Best episode positions saved to {best_positions_file}")

    return summary

# Evaluate PPO and DQN models

dqn_summary = evaluate_model(dqn_model, "DQN", eval_env, n_eval_episodes, max_steps)
ppo_summary = evaluate_model(ppo_model, "PPO", eval_env, n_eval_episodes, max_steps)
# Combine summaries and save
all_summaries = pd.concat([dqn_summary,ppo_summary], ignore_index=True)
all_summaries.to_csv("evaluation_summary_trash.csv", index=False)

print("Evaluation completed. Summary saved to evaluation_summary.csv.")