import json
import threading
import time

import numpy as np
import pytest

from app.ai.dqn_agent import DQNConfiguration
from app.evolution.genetic import GeneticConfiguration
from app.evolution.simulation import TERMINAL, MatchConfig
from app.evolution.training_orchestrator import (
    EvolutionTrainingConfiguration,
    EvolutionTrainingService,
    identity_rewards,
    pong_state_from_envelope,
    round_robin_pairs,
)
from app.persistence.store import EvolutionStore


@pytest.fixture
def store(tmp_path):
    with EvolutionStore(tmp_path / "evolution.db", tmp_path / "checkpoints") as store:
        yield store


def open_store(base_path):
    return EvolutionStore(base_path / "evolution.db", base_path / "checkpoints")


def tiny_dqn_configuration() -> DQNConfiguration:
    return DQNConfiguration(
        replayCapacity=512,
        replayWarmup=8,
        batchSize=16,
        epsilonStart=0.5,
        epsilonMin=0.05,
        epsilonDecaySteps=200,
        targetUpdateInterval=64,
    )


def make_service(store, configuration=None, seed=42, genomes=None):
    return EvolutionTrainingService(
        store,
        configuration=configuration,
        geneticConfiguration=GeneticConfiguration(seed=seed),
        dqnConfiguration=tiny_dqn_configuration(),
        genomes=genomes,
    )


def test_round_robin_pairs_cycle_covers_every_pair_once():
    population = 10
    cycle = population - 1
    seen = {}
    for round_index in range(cycle):
        pairs = round_robin_pairs(population, round_index)
        assert len(pairs) == 5
        indices = [index for pair in pairs for index in pair[:2]]
        assert sorted(indices) == list(range(population))
        for a, b, reversedSides in pairs:
            assert reversedSides is (round_index % 2 == 1)
            assert frozenset((a, b)) not in seen
            seen[frozenset((a, b))] = (round_index, a, b)
    assert len(seen) == population * (population - 1) // 2


def test_round_robin_pairs_rejects_odd_population():
    with pytest.raises(ValueError):
        round_robin_pairs(5, 0)


def test_identity_rewards_are_differential():
    agent = identity_rewards({"pointWinner": "agent"})
    opponent = identity_rewards({"pointWinner": "opponent"})
    neutral = identity_rewards({"pointWinner": None})
    assert agent == (5.01, -4.99)
    assert opponent == (-4.99, 5.01)
    assert neutral == (0.01, 0.01)


def test_pong_state_from_envelope_mirrors_side_b():
    envelope = {
        "ball": {"x": 0.7, "y": 0.5, "vx": 1.0, "vy": 0.2},
        "agentA": {"paddleY": 0.4, "score": 2},
        "agentB": {"paddleY": 0.6, "score": 3},
        "pointWinner": None,
    }
    config = MatchConfig()
    stateA = pong_state_from_envelope(envelope, "A", config)
    stateB = pong_state_from_envelope(envelope, "B", config)

    assert stateA.ballX == pytest.approx(560.0)
    assert stateA.ballVelocityX == pytest.approx(100.0)
    assert stateA.paddleY == pytest.approx(240.0)
    assert stateA.opponentPaddleY == pytest.approx(360.0)
    assert stateA.scoreAgent == 2
    assert stateA.scoreOpponent == 3

    assert stateB.ballX == pytest.approx(240.0)
    assert stateB.ballVelocityX == pytest.approx(-100.0)
    assert stateB.paddleY == pytest.approx(360.0)
    assert stateB.opponentPaddleY == pytest.approx(240.0)
    assert stateB.scoreAgent == 3
    assert stateB.scoreOpponent == 2
    assert stateB.done == stateA.done


def test_run_generation_persists_agents_matches_and_v3_checkpoints(store, tmp_path):
    events = []
    configuration = EvolutionTrainingConfiguration(
        stepsPerAgentPerGeneration=40,
        roundTicks=20,
        snapshotInterval=5,
        metricsInterval=1,
    )
    service = make_service(store, configuration=configuration, seed=7)
    service.on_event = events.append

    summary = service.run_generation(runUuid="run-persist", seed=7)

    assert summary["status"] == "completed"
    assert summary["rounds"] == 2
    run = store.get_run(summary["runId"])
    assert run["status"] == "completed"

    generation = store.get_generation(summary["generationId"])
    assert generation["status"] == "completed"

    agents = store.list_agents(summary["generationId"])
    assert len(agents) == 10
    assert all(agent["role"] == "initial" for agent in agents)
    for agent in agents:
        architecture = json.loads(agent["architecture_json"])
        assert architecture["schemaVersion"] == "genome-v1"
        checkpoints = store.list_checkpoints(agent["id"])
        assert len(checkpoints) == 1
        assert checkpoints[0]["model_spec_version"] == 3
        assert checkpoints[0]["checkpoint_type"] == "full"
        assert store.checkpoints.verify(checkpoints[0]["relative_path"], checkpoints[0]["sha256"])

    assert all(agent.dqn.total_steps == 40 for agent in service.agents)

    matchCount = store.connection.execute(
        "SELECT COUNT(*) FROM matches WHERE run_id = ?", (summary["runId"],)
    ).fetchone()[0]
    assert matchCount == summary["rounds"] * 5

    eventTypes = [event["type"] for event in events]
    assert "population" in eventTypes
    assert "training_metrics" in eventTypes
    assert "match_snapshot" in eventTypes
    assert all(event.get("source") == "LIVE" for event in events)


def test_run_generation_is_deterministic_across_stores(tmp_path):
    configuration = EvolutionTrainingConfiguration(
        stepsPerAgentPerGeneration=40,
        roundTicks=20,
        metricsInterval=1,
    )
    with open_store(tmp_path / "first") as firstStore, open_store(tmp_path / "second") as secondStore:
        first = make_service(firstStore, configuration=configuration, seed=11)
        second = make_service(secondStore, configuration=configuration, seed=11)
        firstSummary = first.run_generation(runUuid="run-deterministic", seed=11)
        secondSummary = second.run_generation(runUuid="run-deterministic", seed=11)

        def signatures(service, summary, evolutionStore):
            agents = sorted(service.agents, key=lambda agent: agent.index)
            agentSigs = [
                (agent.wins, agent.losses, agent.hits, agent.dqn.total_steps) for agent in agents
            ]
            rows = evolutionStore.connection.execute(
                "SELECT arena, agent_a_id, agent_b_id, score_a, score_b, duration_steps "
                "FROM matches WHERE run_id = ? ORDER BY id",
                (summary["runId"],),
            ).fetchall()
            return agentSigs, tuple(tuple(row) for row in rows)

        assert signatures(first, firstSummary, firstStore) == signatures(
            second, secondSummary, secondStore
        )


def test_terminal_transitions_never_bootstrap(store):
    captured = []
    configuration = EvolutionTrainingConfiguration(
        stepsPerAgentPerGeneration=100_000,
        roundTicks=600,
        metricsInterval=1,
        winScore=7,
    )
    service = make_service(store, configuration=configuration, seed=3)
    service.on_event = captured.append

    service.run_generation(runUuid="run-no-bootstrap", seed=3, maxRounds=1)

    terminalEnvelopes = [
        event
        for event in captured
        if event.get("type") == "match_snapshot" and event.get("status") == TERMINAL
    ]
    assert terminalEnvelopes, "expected at least one terminal match in the round"

    sawDone = False
    for agent in service.agents:
        memory = agent.dqn.replay_buffer.memory
        for index, current in enumerate(memory):
            if not current.done:
                continue
            sawDone = True
            if index + 1 < len(memory):
                following = memory[index + 1]
                assert not following.done, "terminal transitions must not bootstrap"
                assert not np.array_equal(current.next_state, following.state)
    assert sawDone, "expected a stored terminal transition"


def test_identity_rewards_accumulate_per_agent_without_cross_contamination(store):
    configuration = EvolutionTrainingConfiguration(
        stepsPerAgentPerGeneration=100_000,
        roundTicks=150,
        metricsInterval=1,
    )
    service = make_service(store, configuration=configuration, seed=5)
    summary = service.run_generation(runUuid="run-identity", seed=5, maxRounds=3)

    rounds = summary["rounds"]
    assert rounds == 3
    totalWins = 0
    totalLosses = 0
    for agent in service.agents:
        # One transition is lost per round (first frame has no preceding state)
        # and one per point boundary (the frame after a done envelope starts a
        # fresh episode and must not bootstrap from the terminal state).
        doneTransitions = sum(1 for transition in agent.dqn.replay_buffer.memory if transition.done)
        transitions = agent.dqn.total_steps - rounds - doneTransitions
        expected = 0.01 * transitions + 5.0 * agent.wins - 5.0 * agent.losses
        assert agent.cumulativeReward == pytest.approx(expected, abs=1e-6)
        totalWins += agent.wins
        totalLosses += agent.losses
    assert totalWins == totalLosses


def test_rejects_population_outside_spec_five_arenas(store):
    from app.evolution.genome import Genome

    configuration = EvolutionTrainingConfiguration(stepsPerAgentPerGeneration=20, roundTicks=10)
    with pytest.raises(ValueError):
        make_service(
            store,
            configuration=configuration,
            genomes=[Genome((32,)), Genome((64,))],
        )


def test_stop_leaves_run_and_generation_statuses_consistent(tmp_path):
    configuration = EvolutionTrainingConfiguration(
        stepsPerAgentPerGeneration=100_000,
        roundTicks=20,
        metricsInterval=1,
    )
    errors = []
    holder = {"service": None}

    def worker():
        try:
            with open_store(tmp_path / "stop") as store:
                service = make_service(store, configuration=configuration, seed=9)
                holder["service"] = service
                service.run_generation(runUuid="run-stop", seed=9)
        except BaseException as exc:  # pragma: no cover - failure surfacing
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    deadline = time.monotonic() + 30
    while holder["service"] is None or holder["service"]._roundIndex < 1:
        assert time.monotonic() < deadline, "run did not start in time"
        time.sleep(0.05)
    holder["service"].controller.request_stop()
    thread.join(30)

    assert not errors
    assert not thread.is_alive()
    with open_store(tmp_path / "stop") as store:
        run = store.get_run(holder["service"].runId)
        assert run["status"] == "cancelled"
        generation = store.get_generation(holder["service"].generationId)
        assert generation["status"] == "aborted"


def test_point_boundary_terminal_transitions_without_match_end(store):
    configuration = EvolutionTrainingConfiguration(
        stepsPerAgentPerGeneration=100_000,
        roundTicks=200,
        metricsInterval=1,
        winScore=40,
    )
    service = make_service(store, configuration=configuration, seed=5)
    summary = service.run_generation(runUuid="run-point-boundary", seed=5, maxRounds=1)

    rows = store.connection.execute(
        "SELECT agent_a_id, agent_b_id, score_a, score_b, result_json FROM matches WHERE run_id = ?",
        (summary["runId"],),
    ).fetchall()
    assert rows, "expected matches for the round"

    scored = [row for row in rows if row["score_a"] + row["score_b"] > 0]
    assert scored, "expected at least one match that scored points"
    assert all(json.loads(row["result_json"]).get("status") != TERMINAL for row in scored), (
        "winScore far above the round's rally count keeps every scored match live; "
        "point-boundary DQN transitions are the acceptance target"
    )

    storeIdToIndex = {agent.storeId: agent.index for agent in service.agents}
    pointsPerAgent = [0] * len(service.agents)
    for row in rows:
        points = row["score_a"] + row["score_b"]
        pointsPerAgent[storeIdToIndex[row["agent_a_id"]]] += points
        pointsPerAgent[storeIdToIndex[row["agent_b_id"]]] += points

    for index, agent in enumerate(service.agents):
        doneTransitions = sum(1 for transition in agent.dqn.replay_buffer.memory if transition.done)
        assert doneTransitions == pointsPerAgent[index], (
            "every scored point must train a done transition at the point boundary "
            "while the match is still live"
        )