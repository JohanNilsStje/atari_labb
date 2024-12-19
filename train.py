import torch
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
from torch.utils.tensorboard import SummaryWriter
import copy
from config import *
from utils import DQNNetwork, ReplayBuffer, AddChannelDimensionWrapper, select_action, optimize_model

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {'GPU' if torch.cuda.is_available() else 'CPU'}.")

# Environment setup
env = gym.make(ENV_NAME)
env = GrayscaleObservation(env, keep_dim=True)
env = ResizeObservation(env, shape=OBSERVATION_SHAPE)
env = AddChannelDimensionWrapper(env)
env = FrameStackObservation(env, stack_size=STACK_SIZE, padding_type="zero")

# Model and buffer initialization
policy_net = DQNNetwork(input_channels=STACK_SIZE, num_actions=env.action_space.n).to(device)
target_net = copy.deepcopy(policy_net)
target_net.eval()
replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
writer = SummaryWriter(TENSORBOARD_LOG_DIR)

# Try to load pretrained model
start_episode = 0
steps_done = 0
try:
    policy_net.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
    print(f"Loaded pretrained model from {PRETRAINED_MODEL_PATH}")
except NameError:
    print(f"No pretrained model found at {PRETRAINED_MODEL_PATH}. Starting training from scratch.")

# Training loop
for episode in range(start_episode, start_episode + NUM_EPISODES):
    state, info = env.reset()
    done = False
    step = 0
    total_reward = 0
    epsilon = max(EPSILON_START * (EPSILON_DECAY ** episode), EPSILON_MIN)  # Decay epsilon per episode

    while not done and step < MAX_STEPS:
        # Select action
        action = select_action(state, policy_net, epsilon, env.action_space, device)

        # Perform action in environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Store transition in replay buffer
        replay_buffer.push(state, action, reward, next_state, done)

        # Update state
        state = next_state

        # Optimize the model
        optimize_model(policy_net, target_net, replay_buffer, BATCH_SIZE, device, optimizer, GAMMA, writer, steps_done)

        steps_done += 1
        step += 1

        # Update target network
        if steps_done % TARGET_UPDATE_FREQUENCY == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(f"Target network updated at step {steps_done}.")

    # Log results to TensorBoard
    writer.add_scalar("Reward/episode", total_reward, episode)
    writer.add_scalar("Epsilon/episode", epsilon, episode)
    print(f"Episode {episode + 1}/{start_episode + NUM_EPISODES} - Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

    # Save the model periodically
    if (episode + 1) % 100 == 0:
        save_path = MODEL_SAVE_PATH.format(episode=episode + 1)
        torch.save(policy_net.state_dict(), save_path)
        print(f"Model saved at {save_path}.")

# Close TensorBoard writer and environment
writer.close()
env.close()