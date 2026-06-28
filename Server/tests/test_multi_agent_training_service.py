from pathlib import Path

from app.ai.multi_agent_training_service import (
    EVALUATION,
    TRAINING_SELF_PLAY,
    MultiAgentGameState,
    MultiAgentTrainingService,
    calculate_adversarial_rewards,
)
from app.ai.q_learning_agent import QLearningAgent, QLearningConfiguration


def make_state(**overrides):
    values = {
        "gameMode": TRAINING_SELF_PLAY,
        "ballX": 400,
        "ballY": 300,
        "ballVelocityX": 120,
        "ballVelocityY": -30,
        "agentPaddleY": 300,
        "opponentPaddleY": 300,
        "agentScore": 0,
        "opponentScore": 0,
        "width": 800,
        "height": 600,
    }
    values.update(overrides)
    return MultiAgentGameState(**values)


def make_service(tmp_path: Path):
    configuration = QLearningConfiguration(epsilonStart=0.0, epsilonMin=0.0)
    return MultiAgentTrainingService(
        agentPlayer=QLearningAgent(configuration),
        opponentAgent=QLearningAgent(configuration),
        agentModelPath=tmp_path / "agent_q_learning_model.json",
        opponentModelPath=tmp_path / "opponent_q_learning_model.json",
    )


def test_rewards_are_opposite_when_agent_scores():
    agentReward, opponentReward = calculate_adversarial_rewards(make_state(pointWinner="agent"))

    assert agentReward > 0
    assert opponentReward < 0


def test_rewards_are_opposite_when_opponent_scores():
    agentReward, opponentReward = calculate_adversarial_rewards(make_state(pointWinner="opponent"))

    assert agentReward < 0
    assert opponentReward > 0


def test_paddle_hit_rewards_correct_agent():
    agentReward, opponentReward = calculate_adversarial_rewards(make_state(lastHitBy="opponent"))

    assert opponentReward > agentReward


def test_evaluation_mode_does_not_update_q_tables(tmp_path):
    service = make_service(tmp_path)
    state = make_state(gameMode=EVALUATION)

    service.process_state(state)
    service.process_state(make_state(gameMode=EVALUATION, ballY=320))

    assert all(value == 0 for actions in service.agentPlayer.qTable.values() for value in actions.values())
    assert all(value == 0 for actions in service.opponentAgent.qTable.values() for value in actions.values())
