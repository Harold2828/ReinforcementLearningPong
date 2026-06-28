from __future__ import annotations

import os
from pathlib import Path

from flask import Flask
from flask_socketio import SocketIO
from .ai.q_learning_agent import QLearningAgent, QLearningConfiguration
from .ai.multi_agent_training_service import MultiAgentTrainingService

socketio = SocketIO(cors_allowed_origins="*")

multi_agent_training_service: MultiAgentTrainingService | None = None

def create_app():
    global multi_agent_training_service

    app = Flask(__name__)
    serverRoot = Path(__file__).resolve().parents[1]
    agentModelSavePath = os.getenv("AGENT_MODEL_SAVE_PATH", "models/agent_q_learning_model.json")
    opponentModelSavePath = os.getenv("OPPONENT_MODEL_SAVE_PATH", "models/opponent_q_learning_model.json")
    resolvedAgentModelSavePath = serverRoot / agentModelSavePath
    resolvedOpponentModelSavePath = serverRoot / opponentModelSavePath

    configuration = QLearningConfiguration(
        learningRate=float(os.getenv("Q_LEARNING_RATE", "0.2")),
        discountFactor=float(os.getenv("Q_DISCOUNT_FACTOR", "0.95")),
        epsilonStart=float(os.getenv("Q_EPSILON_START", "1.0")),
        epsilonMin=float(os.getenv("Q_EPSILON_MIN", "0.05")),
        epsilonDecay=float(os.getenv("Q_EPSILON_DECAY", "0.995")),
        maxStepsPerEpisode=int(os.getenv("Q_MAX_STEPS_PER_EPISODE", "1000")),
        modelSavePath=agentModelSavePath,
    )
    agentPlayer = QLearningAgent(configuration=configuration)
    opponentAgent = QLearningAgent(
        configuration=QLearningConfiguration(
            learningRate=configuration.learningRate,
            discountFactor=configuration.discountFactor,
            epsilonStart=configuration.epsilonStart,
            epsilonMin=configuration.epsilonMin,
            epsilonDecay=configuration.epsilonDecay,
            maxStepsPerEpisode=configuration.maxStepsPerEpisode,
            modelSavePath=opponentModelSavePath,
        )
    )
    multi_agent_training_service = MultiAgentTrainingService(
        agentPlayer=agentPlayer,
        opponentAgent=opponentAgent,
        agentModelPath=resolvedAgentModelSavePath,
        opponentModelPath=resolvedOpponentModelSavePath,
    )
    multi_agent_training_service.load_models()

    socketio.init_app(app)

    from .sockets import register_sockets
    register_sockets(socketio, multi_agent_training_service)

    return app
