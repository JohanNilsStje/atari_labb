import torch
import random
import numpy as np
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym


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
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.int64),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(next_state, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.stack(state),
            torch.stack(action),
            torch.stack(reward),
            torch.stack(next_state),
            torch.stack(done),
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


def optimize_model(policy_net, target_net, replay_buffer, batch_size, device, optimizer, gamma=0.99, writer=None, steps_done=0):
    if len(replay_buffer) < batch_size:
        return

    # Sample a batch of transitions
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    # Move all tensors to the correct device
    state = state.to(device)
    action = action.to(device)
    reward = reward.to(device)
    next_state = next_state.to(device)
    done = done.to(device)

    # Compute current Q values
    q_values = policy_net(state).gather(1, action.unsqueeze(1)).squeeze(1)

    # Compute target Q values
    with torch.no_grad():
        max_next_q_values = target_net(next_state).max(1)[0]
        target_q_values = reward + (gamma * max_next_q_values * (1 - done))

    # Compute loss
    criterion = nn.MSELoss()
    loss = criterion(q_values, target_q_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()

    # Log to TensorBoard if writer is available
    if writer is not None:
        writer.add_scalar('Loss/train', loss.item(), steps_done)


