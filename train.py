import torch
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from gymnasium.spaces import Box
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import copy
import torch.profiler
from config import *
from utils import DQNNetwork, ReplayBuffer, select_action, optimize_model, CustomFrameStack

def make_env(env_name, seed, idx, **kwargs):
    def _init():
        env = gym.make(env_name, **kwargs)
        env.reset(seed=seed + idx)
        env = GrayscaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, OBSERVATION_SHAPE)

        # Apply custom frame stack
        env = CustomFrameStack(env, stack_size=STACK_SIZE)

        def transpose_observation(obs):
            if obs.ndim == 3 and obs.shape == (STACK_SIZE, *OBSERVATION_SHAPE):
                return obs  # Already in (C, H, W) format
            elif obs.ndim == 3:  # (H, W, C)
                return obs.transpose(2, 0, 1)  # Convert to (C, H, W)
            elif obs.ndim == 2:  # Grayscale single frame (H, W)
                return np.expand_dims(obs, axis=0)  # Add channel dimension
            else:
                raise ValueError(f"Unexpected observation shape: {obs.shape}")

        transposed_space = Box(
            low=0,
            high=255,
            shape=(STACK_SIZE, *OBSERVATION_SHAPE),  # (C, H, W)
            dtype=np.uint8
        )
        env = gym.wrappers.TransformObservation(env, lambda obs: transpose_observation(obs), transposed_space)

        # Debug: Check initial observation shape
        test_obs, _ = env.reset()
        print(f"[DEBUG] Environment {idx} initial observation shape: {test_obs.shape}")
        return env
    return _init


if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {'GPU' if torch.cuda.is_available() else 'CPU'}.")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # Environment setup with vectorized environments
    print("[DEBUG] Initializing environments...")
    env = AsyncVectorEnv([make_env(ENV_NAME, seed=42, idx=i) for i in range(NUM_ENVS)])
    print("[DEBUG] Environments initialized successfully.")

    # Model and buffer initialization
    policy_net = DQNNetwork(input_channels=STACK_SIZE, num_actions=env.single_action_space.n).to(device)
    target_net = copy.deepcopy(policy_net)
    target_net.eval()
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY, device)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler()  # Initialize gradient scaler for mixed precision training
    writer = SummaryWriter(TENSORBOARD_LOG_DIR)

    # Initialize tracking for maximum reward
    max_reward = float('-inf')
    max_reward_episode = 0
    start_episode = 0
    steps_done = 0

    # Try to load pretrained model
    try:
        policy_net.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
        print(f"Loaded pretrained model from {PRETRAINED_MODEL_PATH}")
    except FileNotFoundError:
        print(f"No pretrained model found at {PRETRAINED_MODEL_PATH}. Initializing a new model from scratch.")

    # Ensure the target network is a copy of the policy network
    target_net.load_state_dict(policy_net.state_dict())

    # Training loop
    print("[DEBUG] Starting training loop...")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs"),
        record_shapes=True,
        with_stack=True
    ) as prof:
        for episode in range(start_episode, start_episode + NUM_EPISODES):
            # Reset environment and initialize variables
            states, infos = env.reset()
            print(f"[DEBUG] States shape after reset: {states.shape}")  # Debug the shape of states
            dones = [False] * NUM_ENVS
            total_rewards = [0.0] * NUM_ENVS
            epsilon = max(EPSILON_START * (EPSILON_DECAY ** episode), EPSILON_MIN)

            while not all(dones):
                # Select actions for all environments
                actions = [
                    select_action(state, policy_net, epsilon, env.single_action_space, device)
                    for state in states
                ]

                # Perform actions in all environments
                next_states, rewards, terminations, truncations, infos = env.step(actions)

                # Update the dones list based on terminations and truncations
                dones = [done or term or trunc for done, term, trunc in zip(dones, terminations, truncations)]

                # Store transitions in the replay buffer
                replay_buffer.push_batch(states, actions, rewards, next_states, dones)

                # Update states
                states = next_states

                # Optimize the model
                with torch.amp.autocast(device_type="cuda"):  # Enable mixed precision
                    optimize_model(
                        policy_net,
                        target_net,
                        replay_buffer,
                        BATCH_SIZE,
                        device,
                        optimizer,
                        GAMMA,
                        writer,
                        steps_done,
                        scaler
                    )

                # Accumulate rewards
                total_rewards = [r + tr for r, tr in zip(total_rewards, rewards)]
                steps_done += NUM_ENVS

                # Update the target network
                if steps_done % TARGET_UPDATE_FREQUENCY == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                    print(f"[DEBUG] Target network updated at step {steps_done}.")

            # Log results
            mean_reward = sum(total_rewards) / NUM_ENVS
            writer.add_scalar("Reward/episode", mean_reward, episode)
            print(f"Episode {episode + 1}/{NUM_EPISODES} - Mean Reward: {mean_reward:.2f}, Epsilon: {epsilon:.4f}")

            # Save the model periodically
            if (episode + 1) % 100 == 0:
                save_path = MODEL_SAVE_PATH.format(episode=episode + 1)
                torch.save(policy_net.state_dict(), save_path)
                print(f"Model saved at {save_path}.")

    # Print profiler results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Close TensorBoard writer and environment
    writer.close()
    env.close()
