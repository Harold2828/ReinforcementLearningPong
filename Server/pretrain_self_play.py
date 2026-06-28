from __future__ import annotations

import argparse
import os
from pathlib import Path
import random

from app.ai.multi_agent_training_service import MultiAgentTrainingService
from app.ai.pong_training_env import PongTrainingEnv, PongTrainingEnvConfig
from app.ai.q_learning_agent import QLearningAgent, QLearningConfiguration


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain Pong Q-learning agents in a headless gym-like environment.")
    parser.add_argument("--episodes", type=int, default=int(os.getenv("PRETRAIN_EPISODES", "500")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("PRETRAIN_SEED", "7")))
    parser.add_argument("--max-steps", type=int, default=int(os.getenv("PRETRAIN_MAX_STEPS", "1000")))
    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be greater than zero.")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be greater than zero.")

    serverRoot = Path(__file__).resolve().parent
    agentModelPath = _resolve_model_path(serverRoot, os.getenv("AGENT_MODEL_SAVE_PATH", "models/agent_q_learning_model.json"))
    opponentModelPath = _resolve_model_path(serverRoot, os.getenv("OPPONENT_MODEL_SAVE_PATH", "models/opponent_q_learning_model.json"))

    configuration = _configuration()
    service = MultiAgentTrainingService(
        agentPlayer=QLearningAgent(configuration=configuration, randomGenerator=random.Random(args.seed)),
        opponentAgent=QLearningAgent(configuration=configuration, randomGenerator=random.Random(args.seed + 1)),
        agentModelPath=agentModelPath,
        opponentModelPath=opponentModelPath,
    )
    service.load_models()

    env = PongTrainingEnv(
        PongTrainingEnvConfig(maxStepsPerEpisode=args.max_steps),
        randomGenerator=random.Random(args.seed + 2),
    )

    for episodeIndex in range(1, args.episodes + 1):
        state = env.reset()
        done = False
        steps = 0

        while not done:
            response = service.process_state(state)
            snapshot = env.step(response["agentAction"], response["opponentAction"])
            state = snapshot.state
            done = snapshot.done
            steps += 1

            if done:
                service.process_state(state)

        if episodeIndex == 1 or episodeIndex % max(1, args.episodes // 10) == 0:
            metrics = service.metrics.to_dict()
            print(
                "episode={episode} steps={steps} agentWinRate={agentWinRate:.3f} "
                "opponentWinRate={opponentWinRate:.3f} epsilon={epsilon:.3f} opponentEpsilon={opponentEpsilon:.3f}".format(
                    episode=episodeIndex,
                    steps=steps,
                    agentWinRate=metrics["agentWinRate"],
                    opponentWinRate=metrics["opponentWinRate"],
                    epsilon=service.agentPlayer.epsilonValue,
                    opponentEpsilon=service.opponentAgent.epsilonValue,
                )
            )

    service.save_models()
    print(f"saved agent model: {agentModelPath}")
    print(f"saved opponent model: {opponentModelPath}")


def _configuration() -> QLearningConfiguration:
    return QLearningConfiguration(
        learningRate=float(os.getenv("Q_LEARNING_RATE", "0.2")),
        discountFactor=float(os.getenv("Q_DISCOUNT_FACTOR", "0.95")),
        epsilonStart=float(os.getenv("Q_EPSILON_START", "1.0")),
        epsilonMin=float(os.getenv("Q_EPSILON_MIN", "0.05")),
        epsilonDecay=float(os.getenv("Q_EPSILON_DECAY", "0.995")),
        maxStepsPerEpisode=int(os.getenv("Q_MAX_STEPS_PER_EPISODE", "1000")),
    )


def _resolve_model_path(serverRoot: Path, modelPath: str) -> Path:
    path = Path(modelPath)
    return path if path.is_absolute() else serverRoot / path


if __name__ == "__main__":
    main()
