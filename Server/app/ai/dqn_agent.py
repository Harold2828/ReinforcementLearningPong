# app/ai/dqn_agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# Q-Network
# -----------------------------
class DQN(nn.Module):
    """
    Simple fully-connected Deep Q-Network.

    Input:  state_dim (e.g. 10)
    Output: num_actions (e.g. 3: up, down, stay)
    """
    def __init__(self, state_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Replay Buffer
# -----------------------------
@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Fixed-size replay memory that stores transitions (s, a, r, s', done).
    """
    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self.memory: List[Transition] = []
        self.position = 0

    def push(self, transition: Transition) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


# -----------------------------
# DQN Agent
# -----------------------------
class DQNAgent:
    """
    DQN agent for a discrete action space.

    Main methods:
    - select_action(state, explore=True) -> int
    - remember(state, action, reward, next_state, done)
    - train_step()
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int = 3,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 10_000,
        target_update_interval: int = 1_000,
        device: str | None = None,
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_interval = target_update_interval
        self.total_steps = 0

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Main and target networks
        self.policy_net = DQN(state_dim, num_actions).to(self.device)
        self.target_net = DQN(state_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer()

    # -------------------------
    # Exploration schedule
    # -------------------------
    def epsilon(self) -> float:
        """
        Epsilon for epsilon-greedy exploration.
        Decays exponentially from epsilon_start to epsilon_end.
        """
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1.0 * self.total_steps / self.epsilon_decay)

    # -------------------------
    # Action selection
    # -------------------------
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Given a state vector, return an action index (0..num_actions-1).

        If explore=True, use epsilon-greedy.
        """
        self.total_steps += 1

        if explore and random.random() < self.epsilon():
            return random.randrange(self.num_actions)

        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        action = int(q_values.argmax(dim=1).item())
        return action

    # -------------------------
    # Memory
    # -------------------------
    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store a single transition in the replay buffer.
        """
        self.replay_buffer.push(Transition(state, action, reward, next_state, done))

    # -------------------------
    # Training step
    # -------------------------
    def train_step(self) -> None:
        """
        Sample a batch from replay buffer and perform one gradient update.
        """
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples yet

        transitions = self.replay_buffer.sample(self.batch_size)

        # Convert batch to tensors
        state_batch = torch.tensor(
            np.stack([t.state for t in transitions]),
            dtype=torch.float32,
            device=self.device,
        )
        action_batch = torch.tensor(
            [t.action for t in transitions],
            dtype=torch.int64,
            device=self.device,
        ).unsqueeze(1)
        reward_batch = torch.tensor(
            [t.reward for t in transitions],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)
        next_state_batch = torch.tensor(
            np.stack([t.next_state for t in transitions]),
            dtype=torch.float32,
            device=self.device,
        )
        done_batch = torch.tensor(
            [t.done for t in transitions],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        # Q(s, a) for actions taken
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Target values
        with torch.no_grad():
            max_next_q_values = self.target_net(next_state_batch).max(dim=1, keepdim=True)[0]
            targets = reward_batch + (1.0 - done_batch) * self.gamma * max_next_q_values

        loss = self.criterion(q_values, targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network periodically
        if self.total_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
