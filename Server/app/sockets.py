from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from flask_socketio import emit

from .ai.q_learning_agent import QLearningAgent, validate_pong_state


LOGGER = logging.getLogger(__name__)


def register_sockets(socketio, qLearningAgent: QLearningAgent, modelSavePath: Path) -> None:
    @socketio.on("connect")
    def on_connect():
        emit(
            "training_status",
            {
                "connected": True,
                "mode": "Training" if qLearningAgent.trainingEnabled else "Evaluation",
                "episode": qLearningAgent.episodeTracker.episodeIndex,
                "epsilon": qLearningAgent.epsilonValue,
            },
        )

    @socketio.on("disconnect")
    def on_disconnect():
        LOGGER.info("Client disconnected from Q-learning socket.")

    @socketio.on("state_update")
    def on_state_update(payload: dict[str, Any]):
        try:
            currentState = validate_pong_state(payload)
            selectedAction, rewardValue, completedMetrics = qLearningAgent.learn_from_state(currentState)

            if completedMetrics:
                qLearningAgent.save(modelSavePath)

            responsePayload = {
                "action": selectedAction,
                "reward": rewardValue,
                "episode": qLearningAgent.episodeTracker.episodeIndex,
                "epsilon": qLearningAgent.epsilonValue,
                "mode": "Training" if qLearningAgent.trainingEnabled else "Evaluation",
                "metrics": completedMetrics.to_dict() if completedMetrics else None,
            }
            emit("ai_move", responsePayload)
            return responsePayload
        except ValueError as error:
            LOGGER.warning("Invalid state payload rejected: %s", error)
            errorPayload = {"message": str(error)}
            emit("state_error", errorPayload)
            return errorPayload

    @socketio.on("start_training")
    def on_start_training():
        qLearningAgent.start_training()
        statusPayload = _training_status(qLearningAgent)
        emit("training_status", statusPayload)
        return statusPayload

    @socketio.on("stop_training")
    def on_stop_training():
        qLearningAgent.stop_training()
        qLearningAgent.save(modelSavePath)
        statusPayload = _training_status(qLearningAgent)
        emit("training_status", statusPayload)
        return statusPayload

    @socketio.on("reset_episode")
    def on_reset_episode():
        qLearningAgent.reset_episode()
        statusPayload = _training_status(qLearningAgent)
        emit("training_status", statusPayload)
        return statusPayload

    @socketio.on("player_action")
    def on_legacy_player_action(payload: dict[str, Any]):
        LOGGER.warning("Received legacy player_action payload. Use state_update for Q-learning.")
        return on_state_update(_legacy_payload_to_state(payload))


def _training_status(qLearningAgent: QLearningAgent) -> dict[str, Any]:
    return {
        "connected": True,
        "mode": "Training" if qLearningAgent.trainingEnabled else "Evaluation",
        "episode": qLearningAgent.episodeTracker.episodeIndex,
        "epsilon": qLearningAgent.epsilonValue,
        "metrics": qLearningAgent.episodeTracker.latestMetrics.to_dict()
        if qLearningAgent.episodeTracker.latestMetrics
        else None,
    }


def _legacy_payload_to_state(payload: dict[str, Any]) -> dict[str, Any]:
    ball = payload.get("ball", {})
    myself = payload.get("myself", {})
    player = payload.get("player", {})
    ballPosition = ball.get("position", {})
    myselfPosition = myself.get("position", {})
    playerPosition = player.get("position", {})

    return {
        "ballX": ballPosition.get("x"),
        "ballY": ballPosition.get("y"),
        "ballVelocityX": ball.get("velocity", {}).get("x", 0),
        "ballVelocityY": ball.get("velocity", {}).get("y", 0),
        "paddleY": myselfPosition.get("y"),
        "opponentPaddleY": playerPosition.get("y"),
        "scoreAgent": myself.get("score"),
        "scoreOpponent": player.get("score"),
        "done": bool(payload.get("done", False)),
        "agentHitBall": bool(ball.get("hit", False)),
        "agentMissedBall": bool(myself.get("missed", False)),
        "agentScored": bool(myself.get("scored", False)),
        "width": payload.get("width", 800),
        "height": payload.get("height", 600),
    }
