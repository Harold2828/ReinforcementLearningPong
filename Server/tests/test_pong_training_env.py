import random

from app.ai.multi_agent_training_service import TRAINING_SELF_PLAY
from app.ai.pong_training_env import PongTrainingEnv, PongTrainingEnvConfig


def test_training_env_reset_returns_self_play_state():
    env = PongTrainingEnv(randomGenerator=random.Random(1))

    state = env.reset()

    assert state.gameMode == TRAINING_SELF_PLAY
    assert state.agentPaddleY == 300
    assert state.opponentPaddleY == 300
    assert state.comboSmash == 0


def test_training_env_records_agent_hit_and_combo():
    env = PongTrainingEnv(
        PongTrainingEnvConfig(ballSpeedX=12, ballSpeedY=0),
        randomGenerator=random.Random(1),
    )
    env.ballX = env.config.agentPaddleX - env.config.ballSpeedX
    env.ballY = env.agentPaddleY
    env.ballVelocityX = env.config.ballSpeedX
    env.ballVelocityY = 0

    snapshot = env.step("STAY", "STAY")

    assert snapshot.state.lastHitBy == "agent"
    assert snapshot.state.comboSmash == 1
    assert snapshot.done is False


def test_training_env_forces_terminal_state_at_max_steps():
    env = PongTrainingEnv(
        PongTrainingEnvConfig(maxStepsPerEpisode=1),
        randomGenerator=random.Random(2),
    )

    snapshot = env.step("STAY", "STAY")

    assert snapshot.done is True
    assert snapshot.state.pointWinner in {"agent", "opponent"}
