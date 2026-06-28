from pathlib import Path
import random

from app.ai.q_learning_agent import (
    ALLOWED_ACTIONS,
    PongState,
    QLearningAgent,
    QLearningConfiguration,
    calculate_reward,
    validate_pong_state,
)


class ChooseDownOnTieRandom:
    def random(self):
        return 1.0

    def choice(self, actions):
        assert set(actions) == set(ALLOWED_ACTIONS)
        return "DOWN"


def make_state(**overrides):
    values = {
        "ballX": 400,
        "ballY": 300,
        "ballVelocityX": -120,
        "ballVelocityY": 30,
        "paddleY": 295,
        "opponentPaddleY": 310,
        "scoreAgent": 0,
        "scoreOpponent": 0,
        "done": False,
        "width": 800,
        "height": 600,
    }
    values.update(overrides)
    return PongState(**values)


def test_validate_pong_state_requires_required_fields():
    payload = {
        "ballX": 400,
        "ballY": 300,
        "ballVelocityX": -120,
        "ballVelocityY": 30,
        "paddleY": 295,
        "scoreAgent": 0,
        "scoreOpponent": 0,
        "done": False,
    }

    state = validate_pong_state(payload)

    assert state.ballX == 400
    assert state.done is False


def test_validate_pong_state_rejects_invalid_payload():
    payload = {
        "ballX": float("nan"),
        "ballY": 300,
        "ballVelocityX": -120,
        "ballVelocityY": 30,
        "paddleY": 295,
        "scoreAgent": 0,
        "scoreOpponent": 0,
        "done": False,
    }

    try:
        validate_pong_state(payload)
    except ValueError as error:
        assert "ballX" in str(error)
    else:
        raise AssertionError("Invalid state should raise ValueError.")


def test_reward_calculation_includes_hit_score_miss_and_survival():
    hitReward = calculate_reward(make_state(agentHitBall=True))
    missReward = calculate_reward(make_state(agentMissedBall=True))
    scoreReward = calculate_reward(make_state(agentScored=True))
    closerReward = calculate_reward(make_state(previousDistanceToBall=20, paddleY=295, ballY=300))

    assert hitReward > 1
    assert missReward < 0
    assert scoreReward > 5
    assert closerReward > 0.01


def test_q_value_update_uses_q_learning_formula():
    agent = QLearningAgent(
        QLearningConfiguration(learningRate=0.5, discountFactor=0.9),
        randomGenerator=random.Random(1),
    )
    agent.qTable["current"] = {"UP": 1.0, "DOWN": 0.0, "STAY": 0.0}
    agent.qTable["next"] = {"UP": 2.0, "DOWN": 0.5, "STAY": 0.0}

    updatedQValue = agent.update_q_value("current", "UP", 3.0, "next")

    assert updatedQValue == 2.9


def test_terminal_q_value_update_does_not_bootstrap_future_value():
    agent = QLearningAgent(
        QLearningConfiguration(learningRate=0.5, discountFactor=0.9),
        randomGenerator=random.Random(1),
    )
    agent.qTable["current"] = {"UP": 1.0, "DOWN": 0.0, "STAY": 0.0}
    agent.qTable["terminal"] = {"UP": 20.0, "DOWN": 10.0, "STAY": 0.0}

    updatedQValue = agent.update_q_value("current", "UP", 3.0, "terminal", terminal=True)

    assert updatedQValue == 2.0


def test_epsilon_greedy_selects_valid_actions_and_exploits_best_known_action():
    state = make_state()
    agent = QLearningAgent(
        QLearningConfiguration(epsilonStart=0.0, epsilonMin=0.0),
        randomGenerator=random.Random(2),
    )
    stateKey = agent.discretize_state(state)
    agent.qTable[stateKey] = {"UP": -1.0, "DOWN": 4.0, "STAY": 0.0}

    selectedAction = agent.select_action(state)

    assert selectedAction == "DOWN"
    assert selectedAction in ALLOWED_ACTIONS


def test_exploitation_breaks_ties_without_up_bias():
    state = make_state()
    agent = QLearningAgent(
        QLearningConfiguration(epsilonStart=0.0, epsilonMin=0.0),
        randomGenerator=ChooseDownOnTieRandom(),
    )
    stateKey = agent.discretize_state(state)
    agent.qTable[stateKey] = {"UP": 0.0, "DOWN": 0.0, "STAY": 0.0}

    selectedAction = agent.select_action(state)

    assert selectedAction == "DOWN"


def test_model_save_and_load_round_trip(tmp_path: Path):
    modelPath = tmp_path / "q_learning_model.json"
    agent = QLearningAgent(QLearningConfiguration(epsilonStart=0.4, epsilonMin=0.1))
    agent.qTable["state"] = {"UP": 1.25, "DOWN": -0.5, "STAY": 0.0}
    agent.epsilonValue = 0.25

    agent.save(modelPath)
    restoredAgent = QLearningAgent(QLearningConfiguration(epsilonStart=0.4, epsilonMin=0.1))
    loaded = restoredAgent.load(modelPath)

    assert loaded is True
    assert restoredAgent.qTable["state"]["UP"] == 1.25
    assert restoredAgent.epsilonValue == 0.25


def test_missing_or_corrupt_model_does_not_crash(tmp_path: Path):
    agent = QLearningAgent()

    assert agent.load(tmp_path / "missing.json") is False

    corruptPath = tmp_path / "corrupt.json"
    corruptPath.write_text("{not valid json", encoding="utf-8")

    assert agent.load(corruptPath) is False
    assert agent.qTable == {}
