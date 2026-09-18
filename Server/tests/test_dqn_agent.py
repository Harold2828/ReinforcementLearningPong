from pathlib import Path
import random

import numpy as np
import torch

from app.ai.dqn_agent import (
    STATE_DIMENSION,
    DQNAgent,
    DQNConfiguration,
    DuelingDQN,
    pong_state_to_vector,
)
from app.ai.q_learning_agent import PongState


def configuration() -> DQNConfiguration:
    return DQNConfiguration(
        replayCapacity=20,
        replayWarmup=2,
        batchSize=2,
        epsilonStart=0.0,
        epsilonMin=0.0,
        epsilonDecaySteps=10,
        targetUpdateInterval=1,
    )


def state() -> PongState:
    return PongState(
        ballX=400,
        ballY=250,
        ballVelocityX=120,
        ballVelocityY=-30,
        paddleY=300,
        opponentPaddleY=280,
        scoreAgent=1,
        scoreOpponent=0,
        done=False,
    )


def test_dueling_network_outputs_one_value_per_action():
    network = DuelingDQN()
    output = network(torch.zeros((4, STATE_DIMENSION)))

    assert output.shape == (4, 3)


def test_state_vector_is_normalized_and_has_stable_shape():
    vector = pong_state_to_vector(state())

    assert vector.shape == (STATE_DIMENSION,)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()


def test_agent_trains_with_double_dqn_replay():
    agent = DQNAgent(configuration=configuration(), device="cpu", randomGenerator=random.Random(7))
    current = pong_state_to_vector(state())
    following = current.copy()
    following[1] += 0.01
    agent.remember(current, 0, 0.5, following, False)
    agent.remember(following, 1, 5.0, current, True)

    loss = agent.train_step()

    assert loss is not None
    assert loss >= 0
    assert agent.training_steps == 1


def test_checkpoint_round_trip_restores_training_progress(tmp_path: Path):
    checkpoint = tmp_path / "dqn.pt"
    source = DQNAgent(configuration=configuration(), device="cpu", randomGenerator=random.Random(3))
    source.total_steps = 17
    source.training_steps = 4
    source.save(checkpoint)
    restored = DQNAgent(configuration=configuration(), device="cpu", randomGenerator=random.Random(3))

    assert restored.load(checkpoint) is True
    assert restored.total_steps == 17
    assert restored.training_steps == 4
    for sourceParameter, restoredParameter in zip(source.policy_net.parameters(), restored.policy_net.parameters()):
        assert torch.equal(sourceParameter, restoredParameter)
