import random
from pathlib import Path

import numpy as np
import pytest
import torch

from app.ai.dqn_agent import (
    DEFAULT_HIDDEN_WIDTHS,
    STATE_DIMENSION,
    ArchitectureMismatchError,
    DQNAgent,
    DQNConfiguration,
    DuelingDQN,
)
from app.evolution.genome import Genome
from app.ai.q_learning_agent import ALLOWED_ACTIONS


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


def state_vector() -> np.ndarray:
    return np.zeros(STATE_DIMENSION, dtype=np.float32)


def snapshot(agent) -> dict:
    return {
        "hiddenWidths": agent.hiddenWidths,
        "totalSteps": agent.total_steps,
        "policyState": {key: tensor.clone() for key, tensor in agent.policy_net.state_dict().items()},
    }


def assert_unchanged(snapshot_before, agent) -> None:
    for key, tensor in snapshot_before["policyState"].items():
        assert torch.equal(tensor, agent.policy_net.state_dict()[key])
    assert agent.total_steps == snapshot_before["totalSteps"]


def test_valid_architectures_initialize_and_output_three_q_values_for_each_depth():
    architectures = [(32,), (64, 128), (32, 64, 128), (64, 128, 64, 32)]
    for hiddenWidths in architectures:
        network = DuelingDQN(hiddenWidths=hiddenWidths)
        output = network(torch.zeros((4, STATE_DIMENSION)))
        assert network.hiddenWidths == hiddenWidths
        assert output.shape == (4, len(ALLOWED_ACTIONS))


@pytest.mark.parametrize(
    "invalid",
    [
        (),
        (32, 64, 128, 256, 32),
        (16,),
        (128, 0),
        (True, 128),
    ],
)
def test_invalid_architectures_are_rejected(invalid):
    with pytest.raises(ValueError):
        DuelingDQN(hiddenWidths=invalid)
    with pytest.raises(ValueError):
        DQNAgent(configuration=configuration(), hiddenWidths=invalid)


def test_dueling_heads_output_three_action_values_for_different_depths():
    for hiddenWidths in [(32,), (64, 128), (32, 64, 128), (64, 128, 64, 32)]:
        output = DuelingDQN(hiddenWidths=hiddenWidths)(torch.zeros((1, STATE_DIMENSION)))
        assert output.shape == (1, len(ALLOWED_ACTIONS))


def test_online_and_target_networks_share_the_agent_architecture():
    agent = DQNAgent(configuration=configuration(), hiddenWidths=(64, 128, 64, 32), device="cpu")
    assert agent.policy_net.hiddenWidths == (64, 128, 64, 32)
    assert agent.target_net.hiddenWidths == agent.policy_net.hiddenWidths
    assert set(agent.policy_net.state_dict()) == set(agent.target_net.state_dict())


def test_agent_accepts_a_spec02_genome_architecture():
    agent = DQNAgent(configuration=configuration(), hiddenWidths=Genome((32, 128)))
    assert agent.hiddenWidths == (32, 128)
    assert agent.policy_net.hiddenWidths == (32, 128)


def test_default_architecture_is_preserved_as_128_128():
    agent = DQNAgent(configuration=configuration())
    assert agent.hiddenWidths == DEFAULT_HIDDEN_WIDTHS
    assert agent.hiddenWidths == (128, 128)
    assert agent.policy_net(torch.zeros((1, STATE_DIMENSION))).shape == (1, len(ALLOWED_ACTIONS))


def test_checkpoint_round_trip_restores_weights_counters_and_metadata(tmp_path: Path):
    checkpoint = tmp_path / "arch.pt"
    source = DQNAgent(configuration=configuration(), hiddenWidths=(32, 64, 128), device="cpu", randomGenerator=random.Random(3))
    source.total_steps = 21
    source.training_steps = 7
    source.save(checkpoint)
    restored = DQNAgent(configuration=configuration(), hiddenWidths=(32, 64, 128), device="cpu", randomGenerator=random.Random(3))

    assert restored.load(checkpoint) is True
    assert restored.hiddenWidths == (32, 64, 128)
    assert restored.total_steps == 21
    assert restored.training_steps == 7
    for sourceParameter, restoredParameter in zip(source.policy_net.parameters(), restored.policy_net.parameters()):
        assert torch.equal(sourceParameter, restoredParameter)


def test_incompatible_architecture_checkpoint_fails_safely_without_corrupting_models(tmp_path: Path):
    checkpoint = tmp_path / "incompatible.pt"
    source = DQNAgent(configuration=configuration(), hiddenWidths=(32, 64), device="cpu")
    source.save(checkpoint)
    target = DQNAgent(configuration=configuration(), hiddenWidths=(256,), device="cpu", randomGenerator=random.Random(1))
    before = snapshot(target)

    with pytest.raises(ArchitectureMismatchError):
        target.load(checkpoint)

    assert_unchanged(before, target)
    assert target.policy_net(torch.zeros((1, STATE_DIMENSION))).shape == (1, len(ALLOWED_ACTIONS))


def test_legacy_v2_checkpoint_loads_only_into_matching_128_128_agent(tmp_path: Path):
    legacy = tmp_path / "legacy.pt"
    source = DQNAgent(configuration=configuration(), device="cpu")
    torch.save(
        {
            "version": 2,
            "configuration": {},
            "stateDimension": STATE_DIMENSION,
            "numberOfActions": len(ALLOWED_ACTIONS),
            "policyState": source.policy_net.state_dict(),
            "targetState": source.target_net.state_dict(),
            "totalSteps": 9,
        },
        legacy,
    )
    matching = DQNAgent(configuration=configuration(), device="cpu")
    assert matching.load(legacy) is True
    assert matching.total_steps == 9

    nonmatching = DQNAgent(configuration=configuration(), hiddenWidths=(32,), device="cpu")
    before = snapshot(nonmatching)
    with pytest.raises(ArchitectureMismatchError):
        nonmatching.load(legacy)
    assert_unchanged(before, nonmatching)


def test_raw_legacy_checkpoint_requires_default_architecture(tmp_path: Path):
    raw = tmp_path / "raw.pt"
    torch.save(DQNAgent(configuration=configuration(), device="cpu").policy_net.state_dict(), raw)
    nonmatching = DQNAgent(configuration=configuration(), hiddenWidths=(64, 128), device="cpu")
    before = snapshot(nonmatching)
    with pytest.raises(ArchitectureMismatchError):
        nonmatching.load(raw)
    assert_unchanged(before, nonmatching)

    matching = DQNAgent(configuration=configuration(), device="cpu")
    assert matching.load(raw) is True


def test_dimension_mismatch_fails_safely(tmp_path: Path):
    checkpoint = tmp_path / "dims.pt"
    source = DQNAgent(configuration=configuration(), hiddenWidths=(64, 128), device="cpu")
    source.save(checkpoint)
    raw = torch.load(checkpoint, weights_only=False)
    raw["architecture"]["numberOfActions"] = 4
    torch.save(raw, checkpoint)
    target = DQNAgent(configuration=configuration(), hiddenWidths=(64, 128), device="cpu")
    before = snapshot(target)

    with pytest.raises(ArchitectureMismatchError):
        target.load(checkpoint)
    assert_unchanged(before, target)


def test_agents_are_isolated_after_training():
    first = DQNAgent(configuration=configuration(), hiddenWidths=(64, 128), device="cpu")
    second = DQNAgent(configuration=configuration(), hiddenWidths=(64, 128), device="cpu")
    second_before = snapshot(second)

    first.remember(state_vector(), 0, 0.5, state_vector(), False)
    first.remember(state_vector(), 1, 5.0, state_vector(), True)
    assert first.train_step() is not None

    assert_unchanged(second_before, second)
    assert first.policy_net is not second.policy_net
    assert first.target_net is not second.target_net