# app/__init__.py
from flask import Flask
from flask_socketio import SocketIO
from .ai.dqn_agent import DQNAgent
from .ai.utils import json_to_state

dqn_agent: DQNAgent | None = None

prev_state = None
prev_action = None
prev_ball_y = None
prev_my_y = None
episode_done = False

socketio = SocketIO(cors_allowed_origins="*")


def create_app():
    global dqn_agent
    app = Flask(__name__)
    dqn_agent = DQNAgent(state_dim=10, num_actions=3)

    socketio.init_app(app)

    from .sockets import register_sockets
    register_sockets(socketio)

    return app
