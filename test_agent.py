from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from airgym.envs.quad_drone_env import QuadAirSimDroneEnv
from stable_baselines3.common.env_checker import check_env





# Initialize the environment with the target position
env = QuadAirSimDroneEnv(ip_address="127.0.0.1", img_shape=(64, 64, 1), path_file='training_paths.json')


# Initialize the environment


# Load the saved model
model = TD3.load("td3_drone_agent")

# Evaluate the policy
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)

print(f"Mean reward: {mean_reward}, Standard deviation of reward: {std_reward}")

# Optionally, run one episode to visualize the behavior
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)  # Use the trained model to get actions
    obs, reward, done, truncated, info = env.step(action)
    env.render()  # If your environment has a render method