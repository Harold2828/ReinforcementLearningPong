from pathlib import Path

from app.ai.multi_agent_training_service import (
    EVALUATION,
    HUMAN_VS_AI_TRAINING,
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
        "comboSmash": 0,
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


def test_combo_smash_increases_hit_reward_with_cap():
    baseAgentReward, _ = calculate_adversarial_rewards(make_state(lastHitBy="agent", comboSmash=0))
    comboAgentReward, _ = calculate_adversarial_rewards(make_state(lastHitBy="agent", comboSmash=10))
    cappedAgentReward, _ = calculate_adversarial_rewards(make_state(lastHitBy="agent", comboSmash=999))

    assert comboAgentReward > baseAgentReward
    assert cappedAgentReward == baseAgentReward + 0.5


def test_alignment_reward_only_applies_when_ball_moves_toward_paddle():
    towardAgentReward, _ = calculate_adversarial_rewards(
        make_state(ballVelocityX=120, ballY=260, agentPaddleY=260, previousAgentDistanceToBall=80)
    )
    awayAgentReward, _ = calculate_adversarial_rewards(
        make_state(ballVelocityX=-120, ballY=260, agentPaddleY=260, previousAgentDistanceToBall=80)
    )

    assert towardAgentReward > awayAgentReward


def test_training_self_play_decays_epsilon_after_terminal_state(tmp_path):
    configuration = QLearningConfiguration(epsilonStart=1.0, epsilonMin=0.05, epsilonDecay=0.5)
    service = MultiAgentTrainingService(
        agentPlayer=QLearningAgent(configuration),
        opponentAgent=QLearningAgent(configuration),
        agentModelPath=tmp_path / "agent_q_learning_model.json",
        opponentModelPath=tmp_path / "opponent_q_learning_model.json",
    )

    service.process_state(make_state(pointWinner="agent"))

    assert service.agentPlayer.epsilonValue == 0.5
    assert service.opponentAgent.epsilonValue == 0.5


def test_evaluation_mode_does_not_update_q_tables(tmp_path):
    service = make_service(tmp_path)
    state = make_state(gameMode=EVALUATION)

    service.process_state(state)
    service.process_state(make_state(gameMode=EVALUATION, ballY=320))

    assert all(value == 0 for actions in service.agentPlayer.qTable.values() for value in actions.values())
    assert all(value == 0 for actions in service.opponentAgent.qTable.values() for value in actions.values())


def test_human_training_updates_only_ai_and_uses_reduced_exploration(tmp_path):
    configuration = QLearningConfiguration(epsilonStart=1.0, epsilonMin=0.0)
    service = MultiAgentTrainingService(
        agentPlayer=QLearningAgent(configuration),
        opponentAgent=QLearningAgent(configuration),
        agentModelPath=tmp_path / "agent.json",
        opponentModelPath=tmp_path / "opponent.json",
        humanTrainingEpsilon=0.1,
    )

    service.process_state(make_state(gameMode=HUMAN_VS_AI_TRAINING))
    response = service.process_state(
        make_state(gameMode=HUMAN_VS_AI_TRAINING, ballY=320, pointWinner="agent")
    )

    assert response["learningEnabled"] is True
    assert response["opponentAction"] == "STAY"
    assert response["epsilon"] <= 0.1
    assert response["opponentEpsilon"] == 0.0
    assert response["metrics"]["humanTrainingEpisodes"] == 1
    assert service.agentPlayer.qTable
    assert service.opponentAgent.qTable == {}
