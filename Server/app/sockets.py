from __future__ import annotations

import logging
from typing import Any

from flask_socketio import emit

from .ai.multi_agent_training_service import (
    HUMAN_VS_AI,
    MultiAgentTrainingService,
    validate_multi_agent_state,
)


LOGGER = logging.getLogger(__name__)


def register_sockets(socketio, trainingService: MultiAgentTrainingService) -> None:
    @socketio.on("connect")
    def on_connect():
        emit("training_status", trainingService.training_status())

    @socketio.on("disconnect")
    def on_disconnect():
        LOGGER.info("Client disconnected from multi-agent Q-learning socket.")

    @socketio.on("state_update")
    def on_state_update(payload: dict[str, Any]):
        try:
            currentState = validate_multi_agent_state(payload)
            responsePayload = trainingService.process_state(currentState)
            emit("ai_move", responsePayload)
            return responsePayload
        except ValueError as error:
            LOGGER.warning("Invalid multi-agent state payload rejected: %s", error)
            errorPayload = {"message": str(error)}
            emit("state_error", errorPayload)
            return errorPayload

    @socketio.on("start_training")
    def on_start_training():
        trainingService.start_training()
        statusPayload = trainingService.training_status()
        emit("training_status", statusPayload)
        return statusPayload

    @socketio.on("stop_training")
    def on_stop_training():
        trainingService.stop_training()
        trainingService.save_models()
        statusPayload = trainingService.training_status()
        emit("training_status", statusPayload)
        return statusPayload

    @socketio.on("reset_episode")
    def on_reset_episode():
        trainingService.reset_episode()
        statusPayload = trainingService.training_status()
        emit("training_status", statusPayload)
        return statusPayload

    @socketio.on("player_action")
    def on_legacy_player_action(payload: dict[str, Any]):
        LOGGER.warning("Received legacy player_action payload. Use state_update for multi-agent Q-learning.")
        return on_state_update(_legacy_payload_to_multi_agent_state(payload))


def _legacy_payload_to_multi_agent_state(payload: dict[str, Any]) -> dict[str, Any]:
    ball = payload.get("ball", {})
    myself = payload.get("myself", {})
    player = payload.get("player", {})
    ballPosition = ball.get("position", {})
    myselfPosition = myself.get("position", {})
    playerPosition = player.get("position", {})

    pointWinner = None
    if myself.get("scored"):
        pointWinner = "agent"
    elif player.get("scored"):
        pointWinner = "opponent"

    return {
        "gameMode": payload.get("gameMode", HUMAN_VS_AI),
        "ballX": ballPosition.get("x"),
        "ballY": ballPosition.get("y"),
        "ballVelocityX": ball.get("velocity", {}).get("x", 0),
        "ballVelocityY": ball.get("velocity", {}).get("y", 0),
        "agentPaddleY": myselfPosition.get("y"),
        "opponentPaddleY": playerPosition.get("y"),
        "agentScore": myself.get("score"),
        "opponentScore": player.get("score"),
        "lastHitBy": "agent" if ball.get("hit", False) else None,
        "pointWinner": pointWinner,
        "width": payload.get("width", 800),
        "height": payload.get("height", 600),
    }
