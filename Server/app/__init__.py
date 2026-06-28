from __future__ import annotations

import os
from pathlib import Path

from flask import Flask
from flask_socketio import SocketIO
from .ai.q_learning_agent import QLearningAgent, QLearningConfiguration

socketio = SocketIO(cors_allowed_origins="*")

q_learning_agent: QLearningAgent | None = None

def create_app():
    global q_learning_agent

    app = Flask(__name__)
    serverRoot = Path(__file__).resolve().parents[1]
    modelSavePath = os.getenv("MODEL_SAVE_PATH", "models/q_learning_model.json")
    resolvedModelSavePath = serverRoot / modelSavePath

    configuration = QLearningConfiguration(
        learningRate=float(os.getenv("Q_LEARNING_RATE", "0.2")),
        discountFactor=float(os.getenv("Q_DISCOUNT_FACTOR", "0.95")),
        epsilonStart=float(os.getenv("Q_EPSILON_START", "1.0")),
        epsilonMin=float(os.getenv("Q_EPSILON_MIN", "0.05")),
        epsilonDecay=float(os.getenv("Q_EPSILON_DECAY", "0.995")),
        maxStepsPerEpisode=int(os.getenv("Q_MAX_STEPS_PER_EPISODE", "1000")),
        modelSavePath=modelSavePath,
    )
    q_learning_agent = QLearningAgent(configuration=configuration)
    q_learning_agent.load(resolvedModelSavePath)

    socketio.init_app(app)

    from .sockets import register_sockets
    register_sockets(socketio, q_learning_agent, resolvedModelSavePath)

    return app
