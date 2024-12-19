import torch
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
from config import *
from utils import DQNNetwork, AddChannelDimensionWrapper, select_action

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing on {'GPU' if torch.cuda.is_available() else 'CPU'}.")

# Environment setup with rendering
env = gym.make(ENV_NAME, render_mode="human")
env = GrayscaleObservation(env, keep_dim=True)
env = ResizeObservation(env, shape=OBSERVATION_SHAPE)
env = AddChannelDimensionWrapper(env)
env = FrameStackObservation(env, stack_size=STACK_SIZE, padding_type="zero")

# Model initialization
policy_net = DQNNetwork(input_channels=STACK_SIZE, num_actions=env.action_space.n).to(device)

# Load the trained model
try:
    policy_net.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
    policy_net.eval()  # Set the model to evaluation mode
    print(f"Loaded pretrained model from {PRETRAINED_MODEL_PATH}")
except FileNotFoundError:
    print(f"No pretrained model found at {PRETRAINED_MODEL_PATH}. Please train a model first.")
    exit()

# Testing parameters
num_runs = 10
total_rewards = []

# Run the model for `num_runs` episodes
for run in range(num_runs):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Select action using the trained policy
        action = select_action(state, policy_net, epsilon=0.0, action_space=env.action_space, device=device)  # No exploration

        # Perform action and render the environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = next_state

    total_rewards.append(total_reward)
    print(f"Run {run + 1}/{num_runs} - Total Reward: {total_reward}")

# Print average reward over all runs
average_reward = sum(total_rewards) / num_runs
print(f"Average Reward over {num_runs} runs: {average_reward}")

# Close the environment
env.close()
