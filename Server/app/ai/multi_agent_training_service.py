from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
from typing import Any

from .q_learning_agent import ALLOWED_ACTIONS, PongState, QLearningAgent


HUMAN_VS_AI = "HUMAN_VS_AI"
AI_VS_HUMAN = "AI_VS_HUMAN"
AI_VS_AI = "AI_VS_AI"
TRAINING_SELF_PLAY = "TRAINING_SELF_PLAY"
EVALUATION = "EVALUATION"

SUPPORTED_GAME_MODES = (
    HUMAN_VS_AI,
    AI_VS_HUMAN,
    AI_VS_AI,
    TRAINING_SELF_PLAY,
    EVALUATION,
)


@dataclass(frozen=True)
class MultiAgentGameState:
    gameMode: str
    ballX: float
    ballY: float
    ballVelocityX: float
    ballVelocityY: float
    agentPaddleY: float
    opponentPaddleY: float
    agentScore: int
    opponentScore: int
    width: float = 800.0
    height: float = 600.0
    lastHitBy: str | None = None
    pointWinner: str | None = None
    episodeId: str | int | None = None
    previousAgentDistanceToBall: float | None = None
    previousOpponentDistanceToBall: float | None = None

    @property
    def done(self) -> bool:
        return self.pointWinner in ("agent", "opponent")


@dataclass
class MultiAgentMetrics:
    agentWins: int = 0
    opponentWins: int = 0
    agentTotalReward: float = 0.0
    opponentTotalReward: float = 0.0
    agentRewardHistory: list[float] = field(default_factory=list)
    opponentRewardHistory: list[float] = field(default_factory=list)
    agentHits: int = 0
    opponentHits: int = 0
    completedEpisodes: int = 0
    selfPlayEpisodes: int = 0

    def record_step(self, state: MultiAgentGameState, agentReward: float, opponentReward: float) -> None:
        self.agentTotalReward += agentReward
        self.opponentTotalReward += opponentReward

        if state.lastHitBy == "agent":
            self.agentHits += 1
        elif state.lastHitBy == "opponent":
            self.opponentHits += 1

        if not state.done:
            return

        self.completedEpisodes += 1
        if state.gameMode == TRAINING_SELF_PLAY:
            self.selfPlayEpisodes += 1

        if state.pointWinner == "agent":
            self.agentWins += 1
        elif state.pointWinner == "opponent":
            self.opponentWins += 1

        self.agentRewardHistory.append(self.agentTotalReward)
        self.opponentRewardHistory.append(self.opponentTotalReward)
        self.agentRewardHistory = self.agentRewardHistory[-100:]
        self.opponentRewardHistory = self.opponentRewardHistory[-100:]
        self.agentTotalReward = 0.0
        self.opponentTotalReward = 0.0

    def to_dict(self) -> dict[str, float | int]:
        totalWins = self.agentWins + self.opponentWins
        totalHits = self.agentHits + self.opponentHits
        return {
            "agentWinRate": self.agentWins / totalWins if totalWins else 0.0,
            "opponentWinRate": self.opponentWins / totalWins if totalWins else 0.0,
            "agentAverageReward": _average(self.agentRewardHistory),
            "opponentAverageReward": _average(self.opponentRewardHistory),
            "agentHitRate": self.agentHits / totalHits if totalHits else 0.0,
            "opponentHitRate": self.opponentHits / totalHits if totalHits else 0.0,
            "selfPlayEpisodes": self.selfPlayEpisodes,
        }


def validate_multi_agent_state(payload: dict[str, Any]) -> MultiAgentGameState:
    if not isinstance(payload, dict):
        raise ValueError("State payload must be an object.")

    gameMode = payload.get("gameMode", HUMAN_VS_AI)
    if gameMode not in SUPPORTED_GAME_MODES:
        raise ValueError(f"Invalid gameMode: {gameMode}")

    width = float(payload.get("width", 800.0))
    height = float(payload.get("height", 600.0))
    if width <= 0 or height <= 0:
        raise ValueError("State width and height must be greater than zero.")

    agentPaddleY = _require_bounded_number(payload, "agentPaddleY", 0.0, height)
    opponentPaddleY = _require_bounded_number(payload, "opponentPaddleY", 0.0, height)
    agentScore = _require_non_negative_integer(payload, "agentScore")
    opponentScore = _require_non_negative_integer(payload, "opponentScore")

    lastHitBy = payload.get("lastHitBy")
    if lastHitBy not in (None, "agent", "opponent"):
        raise ValueError("lastHitBy must be agent, opponent, or null.")

    pointWinner = payload.get("pointWinner")
    if pointWinner not in (None, "agent", "opponent"):
        raise ValueError("pointWinner must be agent, opponent, or null.")

    return MultiAgentGameState(
        gameMode=gameMode,
        ballX=_require_finite_number(payload, "ballX"),
        ballY=_require_finite_number(payload, "ballY"),
        ballVelocityX=_require_finite_number(payload, "ballVelocityX"),
        ballVelocityY=_require_finite_number(payload, "ballVelocityY"),
        agentPaddleY=agentPaddleY,
        opponentPaddleY=opponentPaddleY,
        agentScore=agentScore,
        opponentScore=opponentScore,
        width=width,
        height=height,
        lastHitBy=lastHitBy,
        pointWinner=pointWinner,
        episodeId=payload.get("episodeId"),
        previousAgentDistanceToBall=_optional_finite_number(payload, "previousAgentDistanceToBall"),
        previousOpponentDistanceToBall=_optional_finite_number(payload, "previousOpponentDistanceToBall"),
    )


def calculate_adversarial_rewards(state: MultiAgentGameState) -> tuple[float, float]:
    agentReward = 0.01
    opponentReward = 0.01

    if state.pointWinner == "agent":
        agentReward += 5.0
        opponentReward -= 5.0
    elif state.pointWinner == "opponent":
        agentReward -= 5.0
        opponentReward += 5.0

    if state.lastHitBy == "agent":
        agentReward += 1.0
        opponentReward -= 0.05
    elif state.lastHitBy == "opponent":
        agentReward -= 0.05
        opponentReward += 1.0

    if state.previousAgentDistanceToBall is not None:
        agentReward += _alignment_reward(abs(state.ballY - state.agentPaddleY), state.previousAgentDistanceToBall)
    if state.previousOpponentDistanceToBall is not None:
        opponentReward += _alignment_reward(abs(state.ballY - state.opponentPaddleY), state.previousOpponentDistanceToBall)

    return agentReward, opponentReward


class MultiAgentTrainingService:
    def __init__(
        self,
        agentPlayer: QLearningAgent,
        opponentAgent: QLearningAgent,
        agentModelPath: Path,
        opponentModelPath: Path,
    ):
        self.agentPlayer = agentPlayer
        self.opponentAgent = opponentAgent
        self.agentModelPath = agentModelPath
        self.opponentModelPath = opponentModelPath
        self.trainingEnabled = True
        self.metrics = MultiAgentMetrics()
        self._agentPreviousStateKey: str | None = None
        self._agentPreviousAction: str | None = None
        self._opponentPreviousStateKey: str | None = None
        self._opponentPreviousAction: str | None = None

    def load_models(self) -> None:
        self.agentPlayer.load(self.agentModelPath)
        self.opponentAgent.load(self.opponentModelPath)

    def save_models(self) -> None:
        self.agentPlayer.save(self.agentModelPath)
        self.opponentAgent.save(self.opponentModelPath)

    def process_state(self, state: MultiAgentGameState) -> dict[str, Any]:
        agentReward, opponentReward = calculate_adversarial_rewards(state)
        learningEnabled = state.gameMode == TRAINING_SELF_PLAY and self.trainingEnabled
        explorationEnabled = state.gameMode == TRAINING_SELF_PLAY and self.trainingEnabled

        agentAction = "STAY"
        opponentAction = "STAY"

        if self._is_agent_ai_controlled(state.gameMode):
            agentAction = self._process_agent(
                self.agentPlayer,
                self._agent_state_from_game_state(state),
                agentReward,
                learningEnabled,
                explorationEnabled,
                "agent",
            )

        if self._is_opponent_ai_controlled(state.gameMode):
            opponentAction = self._process_agent(
                self.opponentAgent,
                self._opponent_state_from_game_state(state),
                opponentReward,
                learningEnabled,
                explorationEnabled,
                "opponent",
            )

        if state.done:
            self.metrics.record_step(state, agentReward, opponentReward)
            if learningEnabled:
                self.save_models()
            self.reset_episode()
        else:
            self.metrics.record_step(state, agentReward, opponentReward)

        return {
            "action": agentAction,
            "agentAction": agentAction,
            "opponentAction": opponentAction,
            "reward": agentReward,
            "agentReward": agentReward,
            "opponentReward": opponentReward,
            "mode": state.gameMode,
            "learningEnabled": learningEnabled,
            "epsilon": self.agentPlayer.epsilonValue,
            "opponentEpsilon": self.opponentAgent.epsilonValue,
            "metrics": self.metrics.to_dict(),
        }

    def start_training(self) -> None:
        self.trainingEnabled = True
        self.agentPlayer.start_training()
        self.opponentAgent.start_training()

    def stop_training(self) -> None:
        self.trainingEnabled = False
        self.agentPlayer.stop_training()
        self.opponentAgent.stop_training()

    def reset_episode(self) -> None:
        self._agentPreviousStateKey = None
        self._agentPreviousAction = None
        self._opponentPreviousStateKey = None
        self._opponentPreviousAction = None

    def training_status(self, gameMode: str = HUMAN_VS_AI) -> dict[str, Any]:
        learningEnabled = gameMode == TRAINING_SELF_PLAY and self.trainingEnabled
        return {
            "connected": True,
            "mode": gameMode,
            "learningEnabled": learningEnabled,
            "epsilon": self.agentPlayer.epsilonValue,
            "opponentEpsilon": self.opponentAgent.epsilonValue,
            "metrics": self.metrics.to_dict(),
        }

    def _process_agent(
        self,
        qLearningAgent: QLearningAgent,
        state: PongState,
        rewardValue: float,
        learningEnabled: bool,
        explorationEnabled: bool,
        role: str,
    ) -> str:
        stateKey = qLearningAgent.discretize_state(state)
        qLearningAgent._ensure_state_actions(stateKey)

        previousStateKey = self._agentPreviousStateKey if role == "agent" else self._opponentPreviousStateKey
        previousAction = self._agentPreviousAction if role == "agent" else self._opponentPreviousAction

        if learningEnabled and previousStateKey and previousAction:
            qLearningAgent.update_q_value(previousStateKey, previousAction, rewardValue, stateKey)

        originalTrainingState = qLearningAgent.trainingEnabled
        qLearningAgent.trainingEnabled = explorationEnabled
        selectedAction = qLearningAgent.select_action(state)
        qLearningAgent.trainingEnabled = originalTrainingState

        if role == "agent":
            self._agentPreviousStateKey = stateKey
            self._agentPreviousAction = selectedAction
        else:
            self._opponentPreviousStateKey = stateKey
            self._opponentPreviousAction = selectedAction

        return selectedAction if selectedAction in ALLOWED_ACTIONS else "STAY"

    @staticmethod
    def _is_agent_ai_controlled(gameMode: str) -> bool:
        return gameMode in (HUMAN_VS_AI, AI_VS_AI, TRAINING_SELF_PLAY, EVALUATION)

    @staticmethod
    def _is_opponent_ai_controlled(gameMode: str) -> bool:
        return gameMode in (AI_VS_HUMAN, AI_VS_AI, TRAINING_SELF_PLAY, EVALUATION)

    @staticmethod
    def _agent_state_from_game_state(state: MultiAgentGameState) -> PongState:
        return PongState(
            ballX=state.ballX,
            ballY=state.ballY,
            ballVelocityX=state.ballVelocityX,
            ballVelocityY=state.ballVelocityY,
            paddleY=state.agentPaddleY,
            opponentPaddleY=state.opponentPaddleY,
            scoreAgent=state.agentScore,
            scoreOpponent=state.opponentScore,
            done=state.done,
            width=state.width,
            height=state.height,
        )

    @staticmethod
    def _opponent_state_from_game_state(state: MultiAgentGameState) -> PongState:
        return PongState(
            ballX=state.width - state.ballX,
            ballY=state.ballY,
            ballVelocityX=-state.ballVelocityX,
            ballVelocityY=state.ballVelocityY,
            paddleY=state.opponentPaddleY,
            opponentPaddleY=state.agentPaddleY,
            scoreAgent=state.opponentScore,
            scoreOpponent=state.agentScore,
            done=state.done,
            width=state.width,
            height=state.height,
        )


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _alignment_reward(currentDistance: float, previousDistance: float) -> float:
    if currentDistance < previousDistance:
        return 0.1
    if currentDistance > previousDistance:
        return -0.1
    return 0.0


def _require_finite_number(payload: dict[str, Any], fieldName: str) -> float:
    if fieldName not in payload:
        raise ValueError(f"Missing required state field: {fieldName}")
    value = payload[fieldName]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"State field {fieldName} must be a finite number.")
    return float(value)


def _optional_finite_number(payload: dict[str, Any], fieldName: str) -> float | None:
    if fieldName not in payload or payload[fieldName] is None:
        return None
    return _require_finite_number(payload, fieldName)


def _require_bounded_number(payload: dict[str, Any], fieldName: str, minimumValue: float, maximumValue: float) -> float:
    value = _require_finite_number(payload, fieldName)
    if value < minimumValue or value > maximumValue:
        raise ValueError(f"State field {fieldName} must be within the playable area.")
    return value


def _require_non_negative_integer(payload: dict[str, Any], fieldName: str) -> int:
    value = _require_finite_number(payload, fieldName)
    if value < 0 or int(value) != value:
        raise ValueError(f"State field {fieldName} must be a non-negative integer.")
    return int(value)
