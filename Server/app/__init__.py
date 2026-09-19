from __future__ import annotations

import os
from pathlib import Path

from flask import Flask
from flask_socketio import SocketIO
from .ai.service_factory import create_training_service

# async_mode="threading": SPEC-06 evolution runs emit socket.io events from a
# dedicated native thread; eventlet/gevent sockets only accept emits from their
# own hub greenlets, so threaded mode is required for LIVE streaming to work.
socketio = SocketIO(cors_allowed_origins="*", async_mode="threading")

multi_agent_training_service = None

def create_app():
    global multi_agent_training_service

    app = Flask(__name__)
    serverRoot = Path(__file__).resolve().parents[1]
    multi_agent_training_service = create_training_service(serverRoot)

    socketio.init_app(app)

    from .sockets import register_sockets
    register_sockets(socketio, multi_agent_training_service)

    _enable_evolution_training(socketio, serverRoot)

    return app


def _enable_evolution_training(socketio, serverRoot: Path) -> None:
    enabled = os.getenv("EVOLUTION_TRAINING_ENABLED", "0").strip().lower() in ("1", "true", "yes")
    if not enabled:
        return
    from .ai.service_factory import create_evolution_training_service
    from .sockets import register_evolution_sockets

    evolutionTrainingService = create_evolution_training_service(serverRoot)
    register_evolution_sockets(socketio, evolutionTrainingService)
