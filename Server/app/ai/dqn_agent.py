from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .q_learning_agent import ALLOWED_ACTIONS, PongState


STATE_DIMENSION = 12


@dataclass(frozen=True)
class DQNConfiguration:
    replayCapacity: int = 100_000
    replayWarmup: int = 5_000
    batchSize: int = 64
    learningRate: float = 0.0001
    discountFactor: float = 0.99
    epsilonStart: float = 1.0
    epsilonMin: float = 0.05
    epsilonDecaySteps: int = 100_000
    targetUpdateInterval: int = 1_000
    gradientClip: float = 10.0

    def validate(self) -> None:
        if self.replayCapacity <= 0:
            raise ValueError("replayCapacity must be greater than zero.")
        if not 0 <= self.replayWarmup <= self.replayCapacity:
            raise ValueError("replayWarmup must be between zero and replayCapacity.")
        if self.batchSize <= 0 or self.batchSize > self.replayCapacity:
            raise ValueError("batchSize must be positive and no larger than replayCapacity.")
        if self.learningRate <= 0:
            raise ValueError("learningRate must be greater than zero.")
        if not 0 <= self.discountFactor <= 1:
            raise ValueError("discountFactor must be between zero and one.")
        if not 0 <= self.epsilonMin <= self.epsilonStart <= 1:
            raise ValueError("epsilon values must satisfy 0 <= epsilonMin <= epsilonStart <= 1.")
        if self.epsilonDecaySteps <= 0 or self.targetUpdateInterval <= 0:
            raise ValueError("decay and target update intervals must be greater than zero.")
        if self.gradientClip <= 0:
            raise ValueError("gradientClip must be greater than zero.")


class DuelingDQN(nn.Module):
    """Small dueling network for the structured Pong state."""

    def __init__(self, stateDimension: int = STATE_DIMENSION, numberOfActions: int = len(ALLOWED_ACTIONS)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(stateDimension, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.valueHead = nn.Linear(128, 1)
        self.advantageHead = nn.Linear(128, numberOfActions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.features(state)
        value = self.valueHead(features)
        advantage = self.advantageHead(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# Preserve the original public name for existing callers.
DQN = DuelingDQN


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000, randomGenerator: random.Random | None = None):
        self.capacity = capacity
        self.memory: list[Transition] = []
        self.position = 0
        self.randomGenerator = randomGenerator or random.Random()

    def push(self, transition: Transition) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batchSize: int) -> list[Transition]:
        return self.randomGenerator.sample(self.memory, batchSize)

    def __len__(self) -> int:
        return len(self.memory)


def pong_state_to_vector(state: PongState) -> np.ndarray:
    """Normalize a mirrored player-centric state into a neural input."""
    width = max(float(state.width), 1.0)
    height = max(float(state.height), 1.0)
    opponentPaddleY = state.opponentPaddleY if state.opponentPaddleY is not None else height / 2
    scoreDifference = max(-5.0, min(5.0, state.scoreAgent - state.scoreOpponent)) / 5.0
    velocityScale = max(width, height)
    return np.asarray(
        [
            state.ballX / width,
            state.ballY / height,
            state.ballVelocityX / velocityScale,
            state.ballVelocityY / velocityScale,
            state.paddleY / height,
            opponentPaddleY / height,
            (state.ballY - state.paddleY) / height,
            (state.ballY - opponentPaddleY) / height,
            scoreDifference,
            1.0 if state.ballVelocityX > 0 else -1.0,
            1.0 if state.ballVelocityY > 0 else -1.0,
            1.0 if state.done else 0.0,
        ],
        dtype=np.float32,
    )


class DQNAgent:
    """Dueling Double DQN agent with replay and resumable checkpoints."""

    def __init__(
        self,
        state_dim: int = STATE_DIMENSION,
        num_actions: int = len(ALLOWED_ACTIONS),
        gamma: float | None = None,
        lr: float | None = None,
        batch_size: int | None = None,
        epsilon_start: float | None = None,
        epsilon_end: float | None = None,
        epsilon_decay: int | None = None,
        target_update_interval: int | None = None,
        device: str | None = None,
        configuration: DQNConfiguration | None = None,
        randomGenerator: random.Random | None = None,
    ):
        base = configuration or DQNConfiguration()
        self.configuration = DQNConfiguration(
            replayCapacity=base.replayCapacity,
            replayWarmup=base.replayWarmup,
            batchSize=batch_size if batch_size is not None else base.batchSize,
            learningRate=lr if lr is not None else base.learningRate,
            discountFactor=gamma if gamma is not None else base.discountFactor,
            epsilonStart=epsilon_start if epsilon_start is not None else base.epsilonStart,
            epsilonMin=epsilon_end if epsilon_end is not None else base.epsilonMin,
            epsilonDecaySteps=epsilon_decay if epsilon_decay is not None else base.epsilonDecaySteps,
            targetUpdateInterval=target_update_interval or base.targetUpdateInterval,
            gradientClip=base.gradientClip,
        )
        self.configuration.validate()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.randomGenerator = randomGenerator or random.Random()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy_net = DuelingDQN(state_dim, num_actions).to(self.device)
        self.target_net = DuelingDQN(state_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.configuration.learningRate)
        self.criterion = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(self.configuration.replayCapacity, self.randomGenerator)
        self.total_steps = 0
        self.training_steps = 0
        self.latest_loss: float | None = None
        self.trainingEnabled = True

    def epsilon(self) -> float:
        progress = min(1.0, self.total_steps / self.configuration.epsilonDecaySteps)
        return self.configuration.epsilonStart + progress * (
            self.configuration.epsilonMin - self.configuration.epsilonStart
        )

    @property
    def epsilonValue(self) -> float:
        return self.epsilon()

    def select_action(
        self,
        state: np.ndarray,
        explore: bool = True,
        epsilonOverride: float | None = None,
    ) -> int:
        if state.shape != (self.state_dim,):
            raise ValueError(f"Expected state shape ({self.state_dim},), received {state.shape}.")
        if explore and self.trainingEnabled:
            self.total_steps += 1
            explorationRate = self.epsilon() if epsilonOverride is None else min(
                self.epsilon(), max(0.0, min(1.0, epsilonOverride))
            )
            if self.randomGenerator.random() < explorationRate:
                return self.randomGenerator.randrange(self.num_actions)
        stateTensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.policy_net(stateTensor).argmax(dim=1).item())

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.replay_buffer.push(Transition(state.copy(), int(action), float(reward), next_state.copy(), bool(done)))

    def train_step(self) -> float | None:
        minimumSize = max(self.configuration.batchSize, self.configuration.replayWarmup)
        if not self.trainingEnabled or len(self.replay_buffer) < minimumSize:
            return None
        transitions = self.replay_buffer.sample(self.configuration.batchSize)
        states = torch.as_tensor(np.stack([item.state for item in transitions]), device=self.device)
        actions = torch.as_tensor([item.action for item in transitions], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor([item.reward for item in transitions], dtype=torch.float32, device=self.device).unsqueeze(1)
        nextStates = torch.as_tensor(np.stack([item.next_state for item in transitions]), device=self.device)
        dones = torch.as_tensor([item.done for item in transitions], dtype=torch.float32, device=self.device).unsqueeze(1)
        predictedValues = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            nextActions = self.policy_net(nextStates).argmax(dim=1, keepdim=True)
            nextValues = self.target_net(nextStates).gather(1, nextActions)
            targets = rewards + (1.0 - dones) * self.configuration.discountFactor * nextValues
        loss = self.criterion(predictedValues, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.configuration.gradientClip)
        self.optimizer.step()
        self.training_steps += 1
        self.latest_loss = float(loss.item())
        if self.training_steps % self.configuration.targetUpdateInterval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return self.latest_loss

    def start_training(self) -> None:
        self.trainingEnabled = True
        self.policy_net.train()

    def stop_training(self) -> None:
        self.trainingEnabled = False
        self.policy_net.eval()

    def diagnostics(self) -> dict[str, float | int | None]:
        return {"trainingLoss": self.latest_loss, "replaySize": len(self.replay_buffer), "trainingSteps": self.training_steps}

    def save(self, path: str | Path) -> None:
        checkpointPath = Path(path).expanduser().resolve()
        checkpointPath.parent.mkdir(parents=True, exist_ok=True)
        temporaryPath = checkpointPath.with_suffix(checkpointPath.suffix + ".tmp")
        torch.save(
            {
                "version": 2,
                "configuration": asdict(self.configuration),
                "stateDimension": self.state_dim,
                "numberOfActions": self.num_actions,
                "policyState": self.policy_net.state_dict(),
                "targetState": self.target_net.state_dict(),
                "optimizerState": self.optimizer.state_dict(),
                "totalSteps": self.total_steps,
                "trainingSteps": self.training_steps,
            },
            temporaryPath,
        )
        temporaryPath.replace(checkpointPath)

    def load(self, path: str | Path) -> bool:
        checkpointPath = Path(path).expanduser().resolve()
        if not checkpointPath.exists():
            return False
        checkpoint: Any = torch.load(checkpointPath, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "policyState" in checkpoint:
            self.policy_net.load_state_dict(checkpoint["policyState"])
            self.target_net.load_state_dict(checkpoint.get("targetState", checkpoint["policyState"]))
            if "optimizerState" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizerState"])
            self.total_steps = int(checkpoint.get("totalSteps", 0))
            self.training_steps = int(checkpoint.get("trainingSteps", 0))
        else:
            self.policy_net.load_state_dict(checkpoint)
            self.target_net.load_state_dict(checkpoint)
        return True

    @staticmethod
    def action_name(actionIndex: int) -> str:
        return ALLOWED_ACTIONS[actionIndex] if 0 <= actionIndex < len(ALLOWED_ACTIONS) else "STAY"

    @staticmethod
    def action_index(actionName: str) -> int:
        return ALLOWED_ACTIONS.index(actionName) if actionName in ALLOWED_ACTIONS else ALLOWED_ACTIONS.index("STAY")
