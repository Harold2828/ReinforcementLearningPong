from __future__ import annotations

import os
from pathlib import Path
import random

import torch

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


def create_evolution_training_service(serverRoot: Path):
    """SPEC-06 evolutionary training service read from the environment.

    Enabled via EVOLUTION_TRAINING_ENABLED=1. Persists to
    <serverRoot>/data/evolution.db with checkpoints under <serverRoot>/data/checkpoints
    unless overridden.
    """
    torch.set_num_threads(int(os.getenv("EVOLUTION_TORCH_THREADS", "1")))
    torch.set_num_interop_threads(int(os.getenv("EVOLUTION_TORCH_INTEROP_THREADS", "1")))

    from ..evolution.genetic import GeneticConfiguration
    from ..evolution.evaluation import EvaluationConfiguration, FitnessConfiguration
    from ..evolution.training_orchestrator import (
        EvolutionTrainingConfiguration,
        EvolutionTrainingService,
    )
    from ..persistence.store import EvolutionStore

    dataDir = _resolve_path(serverRoot, os.getenv("EVOLUTION_DATA_PATH", "data"))
    dbPath = Path(os.getenv("EVOLUTION_DB_PATH", str(dataDir / "evolution.db")))
    checkpointRoot = Path(os.getenv("EVOLUTION_CHECKPOINT_ROOT", str(dataDir / "checkpoints")))
    store = EvolutionStore(dbPath, checkpointRoot)
    courtCount = int(os.getenv("EVOLUTION_COURT_COUNT", "6"))
    configuration = EvolutionTrainingConfiguration(
        courtCount=courtCount,
        stepsPerAgentPerGeneration=int(os.getenv("EVOLUTION_STEPS_PER_AGENT", "100000")),
        roundTicks=int(os.getenv("EVOLUTION_ROUND_TICKS", "1000")),
        maxGenerations=int(os.getenv("EVOLUTION_MAX_GENERATIONS", "1")),
        snapshotInterval=int(os.getenv("EVOLUTION_SNAPSHOT_INTERVAL", "10")),
        optimizerInterval=int(os.getenv("EVOLUTION_OPTIMIZER_INTERVAL", "1")),
        winScore=int(os.getenv("EVOLUTION_WIN_SCORE", "7")),
        device=os.getenv("EVOLUTION_DEVICE", "cpu").strip().lower(),
    )
    geneticConfiguration = GeneticConfiguration(
        populationSize=int(os.getenv("EVOLUTION_POPULATION_SIZE", "12")),
        parentPoolSize=int(os.getenv("EVOLUTION_PARENT_COUNT", "6")),
        elitismCount=int(os.getenv("EVOLUTION_ELITE_COUNT", "2")),
        offspringCount=int(os.getenv("EVOLUTION_OFFSPRING_COUNT", "10")),
        seed=int(os.getenv("EVOLUTION_SEED", "42")),
    )
    dqnConfiguration = DQNConfiguration(
        replayCapacity=int(os.getenv("EVOLUTION_REPLAY_CAPACITY", "100000")),
        replayWarmup=int(os.getenv("EVOLUTION_REPLAY_WARMUP", "5000")),
        batchSize=int(os.getenv("EVOLUTION_BATCH_SIZE", "64")),
        epsilonDecaySteps=int(os.getenv("EVOLUTION_EPSILON_DECAY_STEPS", "100000")),
    )
    evaluationConfiguration = EvaluationConfiguration(
        seeds=tuple(
            int(value.strip())
            for value in os.getenv("EVOLUTION_EVALUATION_SEEDS", "101,211,307").split(",")
            if value.strip()
        ),
        maxStepsPerMatch=int(os.getenv("EVOLUTION_EVALUATION_MAX_STEPS", "2000")),
        winScore=int(os.getenv("EVOLUTION_EVALUATION_WIN_SCORE", "7")),
        maxConcurrentMatches=int(
            os.getenv("EVOLUTION_EVALUATION_CONCURRENCY", str(courtCount))
        ),
        benchmarkVersion=os.getenv(
            "EVOLUTION_BENCHMARK", "spec-07-round-robin-v1"
        ),
    )
    fitnessConfiguration = FitnessConfiguration(
        winRateWeight=float(os.getenv("EVOLUTION_FITNESS_WIN_WEIGHT", "0.65")),
        pointDifferentialWeight=float(os.getenv("EVOLUTION_FITNESS_POINT_WEIGHT", "0.25")),
        comboPerformanceWeight=float(os.getenv("EVOLUTION_FITNESS_COMBO_WEIGHT", "0.10")),
        comboCap=int(os.getenv("EVOLUTION_COMBO_CAP", "10")),
    )
    return EvolutionTrainingService(
        store,
        configuration=configuration,
        geneticConfiguration=geneticConfiguration,
        dqnConfiguration=dqnConfiguration,
        codeRevision=os.getenv("EVOLUTION_CODE_REVISION", ""),
        benchmarkDefinition=os.getenv("EVOLUTION_BENCHMARK", "spec-07-round-robin-v1"),
        evaluationConfiguration=evaluationConfiguration,
        fitnessConfiguration=fitnessConfiguration,
    )


def _resolve_path(serverRoot: Path, configuredPath: str) -> Path:
    path = Path(configuredPath)
    return path if path.is_absolute() else serverRoot / path
