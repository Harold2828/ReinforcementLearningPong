from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .dqn_agent import DQNAgent, pong_state_to_vector
from .multi_agent_training_service import (
    HUMAN_VS_AI,
    TRAINING_SELF_PLAY,
    MultiAgentGameState,
    MultiAgentMetrics,
    MultiAgentTrainingService,
    calculate_adversarial_rewards,
)


class NeuralTrainingService:
    """Socket-compatible self-play service backed by one mirrored shared DQN policy."""

    algorithmName = "dueling_double_dqn"

    def __init__(self, agent: DQNAgent, modelPath: Path):
        self.agent = agent
        self.agentPlayer = agent  # Compatibility with pretraining/status callers.
        self.opponentAgent = agent
        self.modelPath = modelPath
        self.trainingEnabled = True
        self.metrics = MultiAgentMetrics()
        self._previousStates: dict[str, np.ndarray | None] = {"agent": None, "opponent": None}
        self._previousActions: dict[str, int | None] = {"agent": None, "opponent": None}

    def load_models(self) -> None:
        self.agent.load(self.modelPath)

    def save_models(self) -> None:
        self.agent.save(self.modelPath)

    def process_state(self, state: MultiAgentGameState) -> dict[str, Any]:
        agentReward, opponentReward = calculate_adversarial_rewards(state)
        learningEnabled = state.gameMode == TRAINING_SELF_PLAY and self.trainingEnabled
        explorationEnabled = learningEnabled
        agentAction = "STAY"
        opponentAction = "STAY"

        if MultiAgentTrainingService._is_agent_ai_controlled(state.gameMode):
            agentAction = self._process_role(
                "agent",
                pong_state_to_vector(MultiAgentTrainingService._agent_state_from_game_state(state)),
                agentReward,
                state.done,
                learningEnabled,
                explorationEnabled,
            )
        if MultiAgentTrainingService._is_opponent_ai_controlled(state.gameMode):
            opponentAction = self._process_role(
                "opponent",
                pong_state_to_vector(MultiAgentTrainingService._opponent_state_from_game_state(state)),
                opponentReward,
                state.done,
                learningEnabled,
                explorationEnabled,
            )

        self.metrics.record_step(state, agentReward, opponentReward)
        if state.done:
            if learningEnabled:
                self.save_models()
            self.reset_episode()

        diagnostics = self.agent.diagnostics()
        return {
            "action": agentAction,
            "agentAction": agentAction,
            "opponentAction": opponentAction,
            "reward": agentReward,
            "agentReward": agentReward,
            "opponentReward": opponentReward,
            "mode": state.gameMode,
            "algorithm": self.algorithmName,
            "learningEnabled": learningEnabled,
            "epsilon": self.agent.epsilonValue,
            "opponentEpsilon": self.agent.epsilonValue,
            "episode": self.metrics.completedEpisodes + 1,
            "metrics": self.metrics.to_dict(),
            **diagnostics,
        }

    def _process_role(
        self,
        role: str,
        currentState: np.ndarray,
        reward: float,
        done: bool,
        learningEnabled: bool,
        explorationEnabled: bool,
    ) -> str:
        previousState = self._previousStates[role]
        previousAction = self._previousActions[role]
        if learningEnabled and previousState is not None and previousAction is not None:
            self.agent.remember(previousState, previousAction, reward, currentState, done)
            self.agent.train_step()

        actionIndex = self.agent.select_action(currentState, explore=explorationEnabled)
        self._previousStates[role] = currentState
        self._previousActions[role] = actionIndex
        return self.agent.action_name(actionIndex)

    def start_training(self) -> None:
        self.trainingEnabled = True
        self.agent.start_training()

    def stop_training(self) -> None:
        self.trainingEnabled = False
        self.agent.stop_training()

    def reset_episode(self) -> None:
        self._previousStates = {"agent": None, "opponent": None}
        self._previousActions = {"agent": None, "opponent": None}

    def training_status(self, gameMode: str = HUMAN_VS_AI) -> dict[str, Any]:
        return {
            "connected": True,
            "mode": gameMode,
            "algorithm": self.algorithmName,
            "learningEnabled": gameMode == TRAINING_SELF_PLAY and self.trainingEnabled,
            "epsilon": self.agent.epsilonValue,
            "opponentEpsilon": self.agent.epsilonValue,
            "episode": self.metrics.completedEpisodes + 1,
            "metrics": self.metrics.to_dict(),
            **self.agent.diagnostics(),
        }
