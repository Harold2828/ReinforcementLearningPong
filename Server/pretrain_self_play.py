from __future__ import annotations

import argparse
import os
from pathlib import Path
import random

from app.ai.pong_training_env import PongTrainingEnv, PongTrainingEnvConfig
from app.ai.service_factory import create_training_service


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
    service = create_training_service(serverRoot, randomSeed=args.seed)

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
    print(f"saved {service.algorithmName} model checkpoint(s)")


if __name__ == "__main__":
    main()
