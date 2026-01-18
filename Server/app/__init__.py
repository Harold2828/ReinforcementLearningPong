# app/__init__.py
from flask import Flask
from flask_socketio import SocketIO
from .ai.dqn_agent import DQNAgent

socketio = SocketIO(cors_allowed_origins="*")

dqn_agent: DQNAgent | None = None

def create_app():
    global dqn_agent

    app = Flask(__name__)

    dqn_agent = DQNAgent(state_dim=10, num_actions=3)

    dqn_agent.load("models/dqn_pong.pt")

    socketio.init_app(app)

    from .sockets import register_sockets
    register_sockets(socketio)

    return app

