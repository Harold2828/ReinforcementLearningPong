from __future__ import annotations

import os
from pathlib import Path
import random

from .dqn_agent import DQNAgent, DQNConfiguration
from .multi_agent_training_service import MultiAgentTrainingService
from .neural_training_service import NeuralTrainingService
from .q_learning_agent import QLearningAgent, QLearningConfiguration


def create_training_service(serverRoot: Path, randomSeed: int | None = None):
    algorithm = os.getenv("RL_ALGORITHM", "tabular").strip().lower()
    if algorithm == "dqn":
        configuration = DQNConfiguration(
            replayCapacity=int(os.getenv("DQN_REPLAY_CAPACITY", "100000")),
            replayWarmup=int(os.getenv("DQN_REPLAY_WARMUP", "5000")),
            batchSize=int(os.getenv("DQN_BATCH_SIZE", "64")),
            learningRate=float(os.getenv("DQN_LEARNING_RATE", "0.0001")),
            discountFactor=float(os.getenv("DQN_DISCOUNT_FACTOR", "0.99")),
            epsilonStart=float(os.getenv("DQN_EPSILON_START", "1.0")),
            epsilonMin=float(os.getenv("DQN_EPSILON_MIN", "0.05")),
            epsilonDecaySteps=int(os.getenv("DQN_EPSILON_DECAY_STEPS", "100000")),
            targetUpdateInterval=int(os.getenv("DQN_TARGET_UPDATE_INTERVAL", "1000")),
            gradientClip=float(os.getenv("DQN_GRADIENT_CLIP", "10.0")),
        )
        modelPath = _resolve_path(serverRoot, os.getenv("DQN_MODEL_SAVE_PATH", "models/dqn_self_play.pt"))
        service = NeuralTrainingService(
            DQNAgent(configuration=configuration, randomGenerator=random.Random(randomSeed)),
            modelPath,
            humanTrainingEpsilon=float(os.getenv("HUMAN_TRAINING_EPSILON", "0.1")),
        )
        service.load_models()
        return service
    if algorithm != "tabular":
        raise ValueError("RL_ALGORITHM must be either 'tabular' or 'dqn'.")

    agentPath = _resolve_path(serverRoot, os.getenv("AGENT_MODEL_SAVE_PATH", "models/agent_q_learning_model.json"))
    opponentPath = _resolve_path(serverRoot, os.getenv("OPPONENT_MODEL_SAVE_PATH", "models/opponent_q_learning_model.json"))
    configuration = QLearningConfiguration(
        learningRate=float(os.getenv("Q_LEARNING_RATE", "0.2")),
        discountFactor=float(os.getenv("Q_DISCOUNT_FACTOR", "0.95")),
        epsilonStart=float(os.getenv("Q_EPSILON_START", "1.0")),
        epsilonMin=float(os.getenv("Q_EPSILON_MIN", "0.05")),
        epsilonDecay=float(os.getenv("Q_EPSILON_DECAY", "0.995")),
        maxStepsPerEpisode=int(os.getenv("Q_MAX_STEPS_PER_EPISODE", "1000")),
        modelSavePath=str(agentPath),
    )
    service = MultiAgentTrainingService(
        agentPlayer=QLearningAgent(configuration, random.Random(randomSeed)),
        opponentAgent=QLearningAgent(configuration, random.Random(None if randomSeed is None else randomSeed + 1)),
        agentModelPath=agentPath,
        opponentModelPath=opponentPath,
        humanTrainingEpsilon=float(os.getenv("HUMAN_TRAINING_EPSILON", "0.1")),
    )
    service.load_models()
    return service


def _resolve_path(serverRoot: Path, configuredPath: str) -> Path:
    path = Path(configuredPath)
    return path if path.is_absolute() else serverRoot / path
