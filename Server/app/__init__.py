from __future__ import annotations

from pathlib import Path

from flask import Flask
from flask_socketio import SocketIO
from .ai.service_factory import create_training_service

socketio = SocketIO(cors_allowed_origins="*")

multi_agent_training_service = None

def create_app():
    global multi_agent_training_service

    app = Flask(__name__)
    serverRoot = Path(__file__).resolve().parents[1]
    multi_agent_training_service = create_training_service(serverRoot)

    socketio.init_app(app)

    from .sockets import register_sockets
    register_sockets(socketio, multi_agent_training_service)

    return app
