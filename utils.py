import torch
import random
import numpy as np
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

class DQNNetwork(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQNNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = x.squeeze(-1)
        x = x.to(torch.float32)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push_batch(self, states, actions, rewards, next_states, dones):
        # Add transitions from all environments
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.push(state, action, reward, next_state, done)

    def push(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        self.buffer.append((
            state,
            torch.tensor(action, dtype=torch.int64),
            torch.tensor(reward, dtype=torch.float32),
            next_state,
            torch.tensor(done, dtype=torch.float32),
        ))


    def sample(self, batch_size):
        # Sample a batch of transitions
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        # Stack tensors and move them to the correct device
        return (
            torch.stack(state).to(self.device),
            torch.stack(action).to(self.device),
            torch.stack(reward).to(self.device),
            torch.stack(next_state).to(self.device),
            torch.stack(done).to(self.device),
        )

    def __len__(self):
        return len(self.buffer)




class AddChannelDimensionWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_shape[0], old_shape[1], 1),
            dtype=np.uint8,
        )

    def observation(self, obs):
        return np.expand_dims(obs, axis=-1)


def select_action(state, policy_net, epsilon, action_space, device):
    if random.random() < epsilon:
        return action_space.sample()
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
        return q_values.max(1)[1].item()

def optimize_model(policy_net, target_net, replay_buffer, batch_size, device, optimizer, gamma=0.99, writer=None, steps_done=0, scaler=None):
    if len(replay_buffer) < batch_size:
        return

    # Sample a batch of transitions
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # Compute current Q-values
    with torch.cuda.amp.autocast():  # Enable mixed precision
        q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            max_next_q_values = target_net(next_states).max(1)[0]
            target_q_values = rewards + (gamma * max_next_q_values * (1 - dones))

        # Compute loss
        loss = torch.nn.functional.mse_loss(q_values, target_q_values)

    # Backpropagation
    optimizer.zero_grad()

    if scaler:
        scaler.scale(loss).backward()  # Scale loss for mixed precision
        scaler.step(optimizer)         # Step the optimizer
        scaler.update()                # Update the scaler
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
        optimizer.step()

    # Log loss
    if writer:
        writer.add_scalar("Loss/train", loss.item(), steps_done)

class CustomFrameStack(gym.Wrapper):
    def __init__(self, env, stack_size):
        super().__init__(env)
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)

        old_shape = self.observation_space.shape
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(stack_size, *old_shape),  # New shape: (stack_size, height, width, channels)
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.stack_size):
            self.frames.append(obs)
        return self._get_stacked_obs(), info

    def step(self, action):
        obs, reward, done, truncation, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_stacked_obs(), reward, done, truncation, info

    def _get_stacked_obs(self):
        # Stack frames along the new channel dimension (stack_size, height, width, channels)
        return np.stack(list(self.frames), axis=0)

