from pathlib import Path
import random

from app.ai.dqn_agent import DQNAgent, DQNConfiguration
from app.ai.multi_agent_training_service import TRAINING_SELF_PLAY, MultiAgentGameState
from app.ai.neural_training_service import NeuralTrainingService


def make_state(**overrides):
    values = dict(
        gameMode=TRAINING_SELF_PLAY,
        ballX=400,
        ballY=300,
        ballVelocityX=120,
        ballVelocityY=20,
        agentPaddleY=300,
        opponentPaddleY=300,
        agentScore=0,
        opponentScore=0,
    )
    values.update(overrides)
    return MultiAgentGameState(**values)


def test_neural_service_collects_both_player_perspectives(tmp_path: Path):
    agent = DQNAgent(
        configuration=DQNConfiguration(replayCapacity=20, replayWarmup=20, batchSize=2),
        device="cpu",
        randomGenerator=random.Random(5),
    )
    service = NeuralTrainingService(agent, tmp_path / "dqn.pt")

    first = service.process_state(make_state())
    second = service.process_state(make_state(ballY=310))

    assert first["algorithm"] == "dueling_double_dqn"
    assert first["agentAction"] in {"UP", "DOWN", "STAY"}
    assert first["opponentAction"] in {"UP", "DOWN", "STAY"}
    assert second["replaySize"] == 2


def test_neural_service_saves_terminal_checkpoint_and_reports_episode(tmp_path: Path):
    checkpoint = tmp_path / "dqn.pt"
    agent = DQNAgent(
        configuration=DQNConfiguration(replayCapacity=20, replayWarmup=20, batchSize=2),
        device="cpu",
        randomGenerator=random.Random(9),
    )
    service = NeuralTrainingService(agent, checkpoint)
    service.process_state(make_state())

    response = service.process_state(make_state(pointWinner="agent", agentScore=1))

    assert checkpoint.exists()
    assert response["episode"] == 2
    assert response["metrics"]["selfPlayEpisodes"] == 1
