from __future__ import annotations

import logging
import threading
import time
from typing import Any

from flask_socketio import emit

from .ai.multi_agent_training_service import HUMAN_VS_AI, validate_multi_agent_state


LOGGER = logging.getLogger(__name__)


def register_sockets(socketio, trainingService) -> None:
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
        except (TypeError, ValueError) as error:
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


def register_evolution_sockets(socketio, evolutionTrainingService) -> None:
    """SPEC-06 evolution training streaming over Socket.IO.

    Streams LIVE snapshots/metrics as ``evolution_event`` payloads and exposes
    start/pause/resume/stop controls. Runs execute on a background thread and
    honor pause/stop at round boundaries.
    """
    evolutionTrainingService.on_event = lambda event: socketio.emit("evolution_event", event)

    @socketio.on("evolution_start_run")
    def on_start_run(payload: dict[str, Any] | None = None):
        payload = payload or {}
        runningThread = getattr(evolutionTrainingService, "trainingThread", None)
        if runningThread is not None and runningThread.is_alive():
            return {"status": "busy", "message": "an evolution run is already in progress"}
        runUuid = payload.get("runUuid") or f"run-{int(time.time() * 1000)}"
        seed = int(payload.get("seed") or 0)
        fitnessFormula = payload.get("fitnessFormula") or ""
        thread = threading.Thread(
            target=_run_evolution_thread,
            args=(evolutionTrainingService, runUuid, seed, fitnessFormula),
            daemon=True,
        )
        evolutionTrainingService.trainingThread = thread
        thread.start()
        return {"status": "started", "runUuid": runUuid}

    @socketio.on("evolution_pause_run")
    def on_pause_run(payload=None):
        evolutionTrainingService.controller.pause()
        status = "paused" if evolutionTrainingService.runId is not None else "idle"
        return {"status": status}

    @socketio.on("evolution_resume_run")
    def on_resume_run(payload=None):
        evolutionTrainingService.controller.resume()
        status = "running" if evolutionTrainingService.runId is not None else "idle"
        return {"status": status}

    @socketio.on("evolution_stop_run")
    def on_stop_run(payload=None):
        evolutionTrainingService.controller.request_stop()
        return {"status": "stopping"}


def _run_evolution_thread(service, runUuid: str, seed: int, fitnessFormula: str) -> None:
    try:
        summary = service.run_generation(runUuid=runUuid, seed=seed, fitnessFormula=fitnessFormula)
        if service.on_event is not None:
            service.on_event({**summary, "type": "run_finished", "source": "LIVE"})
    except BaseException as error:
        LOGGER.exception("Evolution run failed for %s", runUuid)
        if service.on_event is not None:
            service.on_event({"type": "run_error", "source": "LIVE", "message": str(error)})


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
