from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import random
from typing import Any


LOGGER = logging.getLogger(__name__)

ALLOWED_ACTIONS = ("UP", "DOWN", "STAY")


@dataclass(frozen=True)
class QLearningConfiguration:
    learningRate: float = 0.2
    discountFactor: float = 0.95
    epsilonStart: float = 1.0
    epsilonMin: float = 0.05
    epsilonDecay: float = 0.995
    maxEpisodes: int | None = None
    maxStepsPerEpisode: int = 1000
    modelSavePath: str = "models/q_learning_model.json"

    def validate(self) -> None:
        if not 0 < self.learningRate <= 1:
            raise ValueError("learningRate must be greater than 0 and less than or equal to 1.")
        if not 0 <= self.discountFactor <= 1:
            raise ValueError("discountFactor must be greater than or equal to 0 and less than or equal to 1.")
        if not 0 <= self.epsilonStart <= 1:
            raise ValueError("epsilonStart must be between 0 and 1.")
        if not 0 <= self.epsilonMin <= self.epsilonStart:
            raise ValueError("epsilonMin must be between 0 and epsilonStart.")
        if not 0 < self.epsilonDecay <= 1:
            raise ValueError("epsilonDecay must be greater than 0 and less than or equal to 1.")
        if self.maxStepsPerEpisode <= 0:
            raise ValueError("maxStepsPerEpisode must be greater than 0.")


@dataclass(frozen=True)
class PongState:
    ballX: float
    ballY: float
    ballVelocityX: float
    ballVelocityY: float
    paddleY: float
    scoreAgent: float
    scoreOpponent: float
    done: bool
    opponentPaddleY: float | None = None
    agentHitBall: bool = False
    agentMissedBall: bool = False
    agentScored: bool = False
    previousDistanceToBall: float | None = None
    width: float = 800.0
    height: float = 600.0


@dataclass
class EpisodeMetrics:
    episodeIndex: int
    totalReward: float
    averageReward: float
    paddleHits: int
    missedBalls: int
    episodeLength: int
    epsilonValue: float
    createdAt: str
    winRate: float

    @property
    def paddleHitRate(self) -> float:
        interactions = self.paddleHits + self.missedBalls
        return self.paddleHits / interactions if interactions else 0.0

    @property
    def missRate(self) -> float:
        interactions = self.paddleHits + self.missedBalls
        return self.missedBalls / interactions if interactions else 0.0

    def to_dict(self) -> dict[str, float | int | str]:
        payload = asdict(self)
        payload["paddleHitRate"] = self.paddleHitRate
        payload["missRate"] = self.missRate
        payload["averageEpisodeLength"] = self.episodeLength
        return payload


@dataclass
class EpisodeTracker:
    episodeIndex: int = 1
    totalReward: float = 0.0
    paddleHits: int = 0
    missedBalls: int = 0
    episodeLength: int = 0
    wins: int = 0
    completedEpisodes: int = 0
    rewardHistory: list[float] = field(default_factory=list)
    latestMetrics: EpisodeMetrics | None = None


def _require_finite_number(payload: dict[str, Any], fieldName: str) -> float:
    if fieldName not in payload:
        raise ValueError(f"Missing required state field: {fieldName}")

    value = payload[fieldName]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"State field {fieldName} must be a finite number.")

    return float(value)


def validate_pong_state(payload: dict[str, Any]) -> PongState:
    if not isinstance(payload, dict):
        raise ValueError("State payload must be an object.")

    done = payload.get("done")
    if not isinstance(done, bool):
        raise ValueError("State field done must be a boolean.")

    opponentPaddleY = payload.get("opponentPaddleY")
    if opponentPaddleY is not None:
        if isinstance(opponentPaddleY, bool) or not isinstance(opponentPaddleY, (int, float)) or not math.isfinite(opponentPaddleY):
            raise ValueError("State field opponentPaddleY must be a finite number when provided.")
        opponentPaddleY = float(opponentPaddleY)

    width = float(payload.get("width", 800.0))
    height = float(payload.get("height", 600.0))
    if width <= 0 or height <= 0:
        raise ValueError("State width and height must be greater than zero.")

    previousDistanceToBall = payload.get("previousDistanceToBall")
    if previousDistanceToBall is not None:
        previousDistanceToBall = float(previousDistanceToBall)

    return PongState(
        ballX=_require_finite_number(payload, "ballX"),
        ballY=_require_finite_number(payload, "ballY"),
        ballVelocityX=_require_finite_number(payload, "ballVelocityX"),
        ballVelocityY=_require_finite_number(payload, "ballVelocityY"),
        paddleY=_require_finite_number(payload, "paddleY"),
        opponentPaddleY=opponentPaddleY,
        scoreAgent=_require_finite_number(payload, "scoreAgent"),
        scoreOpponent=_require_finite_number(payload, "scoreOpponent"),
        done=done,
        agentHitBall=bool(payload.get("agentHitBall", False)),
        agentMissedBall=bool(payload.get("agentMissedBall", False)),
        agentScored=bool(payload.get("agentScored", False)),
        previousDistanceToBall=previousDistanceToBall,
        width=width,
        height=height,
    )


def calculate_reward(currentState: PongState) -> float:
    rewardValue = 0.01

    if currentState.agentHitBall:
        rewardValue += 1.0
    if currentState.agentScored:
        rewardValue += 5.0
    if currentState.agentMissedBall:
        rewardValue -= 5.0

    if currentState.previousDistanceToBall is not None:
        currentDistanceToBall = abs(currentState.ballY - currentState.paddleY)
        if currentDistanceToBall < currentState.previousDistanceToBall:
            rewardValue += 0.1
        elif currentDistanceToBall > currentState.previousDistanceToBall:
            rewardValue -= 0.1

    return rewardValue


class QLearningAgent:
    def __init__(
        self,
        configuration: QLearningConfiguration | None = None,
        randomGenerator: random.Random | None = None,
    ):
        self.configuration = configuration or QLearningConfiguration()
        self.configuration.validate()
        self.randomGenerator = randomGenerator or random.Random()
        self.qTable: dict[str, dict[str, float]] = {}
        self.epsilonValue = self.configuration.epsilonStart
        self.trainingEnabled = True
        self.previousStateKey: str | None = None
        self.previousAction: str | None = None
        self.episodeTracker = EpisodeTracker()

    def discretize_state(self, currentState: PongState) -> str:
        ballXBucket = self._bucket(currentState.ballX, currentState.width, 12)
        ballYBucket = self._bucket(currentState.ballY, currentState.height, 12)
        paddleYBucket = self._bucket(currentState.paddleY, currentState.height, 12)
        ballVelocityXDirection = -1 if currentState.ballVelocityX < 0 else 1
        ballVelocityYDirection = -1 if currentState.ballVelocityY < 0 else 1
        scoreDifference = int(max(-3, min(3, currentState.scoreAgent - currentState.scoreOpponent)))

        return "|".join(
            str(part)
            for part in (
                ballXBucket,
                ballYBucket,
                paddleYBucket,
                ballVelocityXDirection,
                ballVelocityYDirection,
                scoreDifference,
            )
        )

    def select_action(self, currentState: PongState) -> str:
        currentStateKey = self.discretize_state(currentState)
        self._ensure_state_actions(currentStateKey)

        if self.trainingEnabled and self.randomGenerator.random() < self.epsilonValue:
            selectedAction = self.randomGenerator.choice(ALLOWED_ACTIONS)
        else:
            selectedAction = max(
                ALLOWED_ACTIONS,
                key=lambda actionName: self.qTable[currentStateKey].get(actionName, 0.0),
            )

        return selectedAction

    def learn_from_state(self, currentState: PongState) -> tuple[str, float, EpisodeMetrics | None]:
        currentStateKey = self.discretize_state(currentState)
        self._ensure_state_actions(currentStateKey)

        rewardValue = calculate_reward(currentState)

        if self.trainingEnabled and self.previousStateKey and self.previousAction:
            self.update_q_value(
                currentStateKey=self.previousStateKey,
                selectedAction=self.previousAction,
                rewardValue=rewardValue,
                nextStateKey=currentStateKey,
            )

        selectedAction = self.select_action(currentState)
        self.previousStateKey = currentStateKey
        self.previousAction = selectedAction
        completedMetrics = self._record_step(currentState, rewardValue)

        return selectedAction, rewardValue, completedMetrics

    def update_q_value(
        self,
        currentStateKey: str,
        selectedAction: str,
        rewardValue: float,
        nextStateKey: str,
    ) -> float:
        self._ensure_state_actions(currentStateKey)
        self._ensure_state_actions(nextStateKey)
        currentQValue = self.qTable[currentStateKey][selectedAction]
        nextBestQValue = max(self.qTable[nextStateKey].values())
        updatedQValue = currentQValue + self.configuration.learningRate * (
            rewardValue + self.configuration.discountFactor * nextBestQValue - currentQValue
        )
        self.qTable[currentStateKey][selectedAction] = updatedQValue
        return updatedQValue

    def start_training(self) -> None:
        self.trainingEnabled = True

    def stop_training(self) -> None:
        self.trainingEnabled = False

    def reset_episode(self) -> None:
        self.previousStateKey = None
        self.previousAction = None
        self.episodeTracker = EpisodeTracker(
            episodeIndex=self.episodeTracker.episodeIndex,
            wins=self.episodeTracker.wins,
            completedEpisodes=self.episodeTracker.completedEpisodes,
            rewardHistory=list(self.episodeTracker.rewardHistory),
            latestMetrics=self.episodeTracker.latestMetrics,
        )

    def save(self, modelPath: Path) -> None:
        safeModelPath = self._safe_model_path(modelPath)
        safeModelPath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "configuration": asdict(self.configuration),
            "epsilonValue": self.epsilonValue,
            "qTable": self.qTable,
            "episodeIndex": self.episodeTracker.episodeIndex,
            "metrics": self.episodeTracker.latestMetrics.to_dict() if self.episodeTracker.latestMetrics else None,
        }
        safeModelPath.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, modelPath: Path) -> bool:
        safeModelPath = self._safe_model_path(modelPath)
        if not safeModelPath.exists():
            LOGGER.warning("Q-learning model file is missing. Starting with a new Q-table: %s", safeModelPath)
            return False

        try:
            payload = json.loads(safeModelPath.read_text(encoding="utf-8"))
            qTable = payload.get("qTable", {})
            if not isinstance(qTable, dict):
                raise ValueError("qTable must be an object.")
            self.qTable = {
                str(stateKey): {actionName: float(value) for actionName, value in actions.items() if actionName in ALLOWED_ACTIONS}
                for stateKey, actions in qTable.items()
                if isinstance(actions, dict)
            }
            self.epsilonValue = float(payload.get("epsilonValue", self.configuration.epsilonStart))
            self.episodeTracker.episodeIndex = int(payload.get("episodeIndex", 1))
            return True
        except Exception as error:
            LOGGER.error("Q-learning model file is invalid. Starting with a new Q-table: %s", error)
            self.qTable = {}
            return False

    def _record_step(self, currentState: PongState, rewardValue: float) -> EpisodeMetrics | None:
        tracker = self.episodeTracker
        tracker.totalReward += rewardValue
        tracker.episodeLength += 1
        if currentState.agentHitBall:
            tracker.paddleHits += 1
        if currentState.agentMissedBall:
            tracker.missedBalls += 1
        if currentState.agentScored:
            tracker.wins += 1

        episodeEnded = currentState.done or tracker.episodeLength >= self.configuration.maxStepsPerEpisode
        if not episodeEnded:
            return None

        tracker.completedEpisodes += 1
        tracker.rewardHistory.append(tracker.totalReward)
        if len(tracker.rewardHistory) > 100:
            tracker.rewardHistory = tracker.rewardHistory[-100:]

        averageReward = sum(tracker.rewardHistory) / len(tracker.rewardHistory)
        winRate = tracker.wins / tracker.completedEpisodes if tracker.completedEpisodes else 0.0
        metrics = EpisodeMetrics(
            episodeIndex=tracker.episodeIndex,
            totalReward=tracker.totalReward,
            averageReward=averageReward,
            paddleHits=tracker.paddleHits,
            missedBalls=tracker.missedBalls,
            episodeLength=tracker.episodeLength,
            epsilonValue=self.epsilonValue,
            createdAt=datetime.now(timezone.utc).isoformat(),
            winRate=winRate,
        )

        LOGGER.info("Q-learning episode metrics: %s", metrics.to_dict())
        tracker.latestMetrics = metrics
        tracker.episodeIndex += 1
        self.epsilonValue = max(self.configuration.epsilonMin, self.epsilonValue * self.configuration.epsilonDecay)
        self.previousStateKey = None
        self.previousAction = None
        tracker.totalReward = 0.0
        tracker.paddleHits = 0
        tracker.missedBalls = 0
        tracker.episodeLength = 0

        return metrics

    def _ensure_state_actions(self, stateKey: str) -> None:
        if stateKey not in self.qTable:
            self.qTable[stateKey] = {actionName: 0.0 for actionName in ALLOWED_ACTIONS}

    @staticmethod
    def _bucket(value: float, maximumValue: float, numberOfBuckets: int) -> int:
        boundedValue = max(0.0, min(float(maximumValue), float(value)))
        bucketSize = float(maximumValue) / numberOfBuckets
        return min(numberOfBuckets - 1, int(boundedValue / bucketSize))

    @staticmethod
    def _safe_model_path(modelPath: Path) -> Path:
        resolvedPath = modelPath.expanduser().resolve()
        if ".." in modelPath.parts:
            raise ValueError("Model save path must not contain unsafe path traversal.")
        return resolvedPath
