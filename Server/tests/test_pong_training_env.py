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
    contact = env.config.agentPaddleX - (env.config.paddleWidth + env.config.ballWidth) / 2
    env.ballX = contact - env.config.ballSpeedX * env.config.stepSeconds + 0.01
    env.ballY = env.agentPaddleY
    env.ballVelocityX = env.config.ballSpeedX
    env.ballVelocityY = 0

    snapshot = env.step("STAY", "STAY")

    assert snapshot.state.lastHitBy == "agent"
    assert snapshot.state.comboSmash == 1
    assert snapshot.done is False


def test_original_phaser_trajectory_and_collision_rules(monkeypatch):
    env = PongTrainingEnv(randomGenerator=random.Random(1))
    assert env.config.ballWidth == 48.1
    assert env.config.ballHeight == 51.9
    first = env.step("UP", "DOWN").state

    assert first.ballX == 400 + 200 / 60
    assert first.ballY == 300 + 200 / 60
    assert first.agentPaddleY == 300 - 500 / 60
    assert first.opponentPaddleY == 300 + 500 / 60

    contact = env.config.agentPaddleX - (env.config.paddleWidth + env.config.ballWidth) / 2
    env.ballX = contact - env.ballVelocityX * env.config.stepSeconds + 0.01
    env.ballY = env.agentPaddleY + 10
    env.ballVelocityX = 200
    env.ballVelocityY = 0
    factors = iter((3.5, 1.5))
    monkeypatch.setattr(env, "_random_normal", lambda *_: next(factors))

    collision = env.step("STAY", "STAY").state
    assert collision.ballX == contact
    assert collision.ballVelocityX == -300
    assert collision.ballVelocityY == 35
    assert collision.lastHitBy == "agent"


def test_original_phaser_wall_bounds_and_collision_order():
    env = PongTrainingEnv(randomGenerator=random.Random(1))
    halfHeight = env.config.ballHeight / 2
    env.ballY = halfHeight + 1
    env.ballVelocityY = -200
    env.step("STAY", "STAY")
    assert env.ballY == halfHeight
    assert env.ballVelocityY == 200

    env.ballX = 640
    env.ballY = env.agentPaddleY
    env.ballVelocityX = 10_000
    scored = env.step("STAY", "STAY").state
    assert scored.pointWinner == "opponent"
    assert scored.lastHitBy is None


def test_training_env_forces_terminal_state_at_max_steps():
    env = PongTrainingEnv(
        PongTrainingEnvConfig(maxStepsPerEpisode=1),
        randomGenerator=random.Random(2),
    )

    snapshot = env.step("STAY", "STAY")

    assert snapshot.done is True
    assert snapshot.state.pointWinner in {"agent", "opponent"}
