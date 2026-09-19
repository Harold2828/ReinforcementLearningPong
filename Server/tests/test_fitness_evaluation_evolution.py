import json

import pytest
import torch

from app.ai.dqn_agent import DQNAgent, DQNConfiguration
from app.evolution.evaluation import (
    EvaluationConfiguration,
    FitnessConfiguration,
    RoundRobinEvaluator,
    calculate_fitness,
)
from app.evolution.genetic import (
    EliteRecord,
    GenerationPlan,
    GeneticConfiguration,
    OffspringRecord,
)
from app.evolution.genome import Genome
from app.evolution.training_orchestrator import (
    AgentRecord,
    EvolutionTrainingConfiguration,
    EvolutionTrainingService,
)
from app.persistence.store import EvolutionStore


def test_fitness_is_bounded_and_handles_draws_zero_matches_and_combo_cap():
    empty = calculate_fitness(0, 0, 0, 0, 0, 0)
    assert empty == {
        "winRate": 0.0,
        "pointDifferential": 0.0,
        "comboPerformance": 0.0,
        "fitness": 0.0,
    }

    result = calculate_fitness(0, 2, 0, 0, 0, 10_000)
    assert result["winRate"] == 0.5
    assert result["pointDifferential"] == 0.5
    assert result["comboPerformance"] == 1.0
    assert result["fitness"] == pytest.approx(0.55)
    assert all(0.0 <= result[key] <= 1.0 for key in result)


def test_evaluation_is_inference_only_and_reverses_sides():
    agents = [
        DQNAgent(
            configuration=DQNConfiguration(replayCapacity=8, replayWarmup=8, batchSize=2),
            hiddenWidths=(32,),
            device="cpu",
        )
        for _ in range(12)
    ]
    evaluator = RoundRobinEvaluator(
        EvaluationConfiguration(
            seeds=(3,), maxStepsPerMatch=2, winScore=1, maxConcurrentMatches=6
        )
    )
    before = [
        (agent.total_steps, agent.training_steps, len(agent.replay_buffer))
        for agent in agents
    ]
    results = evaluator.evaluate(agents)

    assert len(evaluator.matchResults) == 132
    assert len({
        frozenset((match["agentAIndex"], match["agentBIndex"]))
        for match in evaluator.matchResults
    }) == 66
    orientations = {}
    for match in evaluator.matchResults:
        key = frozenset((match["agentAIndex"], match["agentBIndex"]))
        orientations.setdefault(key, set()).add(match["reversedSides"])
    assert all(sides == {False, True} for sides in orientations.values())
    assert evaluator.maxConcurrentObserved == 6
    assert all(result["sampleCount"] == 22 for result in results)
    assert all(result["trainingStateUnchanged"] for result in results)
    assert all(result["weightsUnchanged"] for result in results)
    assert [
        (agent.total_steps, agent.training_steps, len(agent.replay_buffer))
        for agent in agents
    ] == before
    assert all(agent.trainingEnabled for agent in agents)


def test_elite_and_offspring_weight_inheritance_is_safe():
    config = DQNConfiguration(replayCapacity=8, replayWarmup=8, batchSize=2)
    parent = DQNAgent(configuration=config, hiddenWidths=(32,), device="cpu")
    with torch.no_grad():
        for tensor in parent.policy_net.parameters():
            tensor.fill_(7.0)
        for tensor in parent.target_net.parameters():
            tensor.fill_(9.0)
    parentRecord = AgentRecord(0, 1, parent)

    eliteChild = DQNAgent(configuration=config, hiddenWidths=(32,), device="cpu")
    elitePlan = GenerationPlan(
        genomes=(Genome((32,)),),
        eliteIndices=(0,),
        parentPoolIndices=(0,),
        elites=(EliteRecord(0, 0, Genome((32,))),),
        offspring=(),
        generationIndex=1,
        generationSeed=1,
    )
    eliteAudit = EvolutionTrainingService._inherit_weights(
        eliteChild, 0, elitePlan, [parentRecord]
    )
    assert eliteAudit["policy"] == "elite_exact_copy"
    for key, tensor in parent.policy_net.state_dict().items():
        assert torch.equal(tensor, eliteChild.policy_net.state_dict()[key])
    for key, tensor in parent.target_net.state_dict().items():
        assert torch.equal(tensor, eliteChild.target_net.state_dict()[key])

    offspringChild = DQNAgent(configuration=config, hiddenWidths=(64,), device="cpu")
    before = {key: tensor.clone() for key, tensor in offspringChild.policy_net.state_dict().items()}
    offspringPlan = GenerationPlan(
        genomes=(Genome((64,)),),
        eliteIndices=(),
        parentPoolIndices=(0,),
        elites=(),
        offspring=(OffspringRecord(0, 0, 0, 0, None, Genome((64,))),),
        generationIndex=1,
        generationSeed=1,
    )
    offspringAudit = EvolutionTrainingService._inherit_weights(
        offspringChild, 0, offspringPlan, [parentRecord]
    )
    assert offspringAudit["policy"] == "compatible_tensors_only"
    assert offspringAudit["skippedKeys"]
    for key in offspringAudit["copiedKeys"]:
        assert torch.equal(
            parent.policy_net.state_dict()[key], offspringChild.policy_net.state_dict()[key]
        )
    for key in offspringAudit["skippedKeys"]:
        if key in before and before[key].shape == offspringChild.policy_net.state_dict()[key].shape:
            assert torch.equal(before[key], offspringChild.policy_net.state_dict()[key])
    for key, tensor in offspringChild.policy_net.state_dict().items():
        assert torch.equal(tensor, offspringChild.target_net.state_dict()[key])
    assert len(offspringChild.replay_buffer) == 0
    assert offspringChild.total_steps == offspringChild.training_steps == 0


def test_complete_three_generation_run(tmp_path):
    events = []
    training = EvolutionTrainingConfiguration(
        stepsPerAgentPerGeneration=2,
        roundTicks=2,
        maxGenerations=3,
        snapshotInterval=1,
        metricsInterval=10,
        winScore=1,
        device="cpu",
    )
    genetic = GeneticConfiguration(seed=17, mutationProbability=0.0)
    dqn = DQNConfiguration(
        replayCapacity=16,
        replayWarmup=16,
        batchSize=2,
        epsilonStart=0.1,
        epsilonMin=0.0,
        epsilonDecaySteps=10,
    )
    evaluation = EvaluationConfiguration(
        seeds=(23,), maxStepsPerMatch=2, winScore=1, maxConcurrentMatches=6
    )
    genomes = [Genome((32,)) for _ in range(12)]

    with EvolutionStore(tmp_path / "evolution.db", tmp_path / "checkpoints") as store:
        service = EvolutionTrainingService(
            store,
            configuration=training,
            geneticConfiguration=genetic,
            dqnConfiguration=dqn,
            genomes=genomes,
            onEvent=events.append,
        )
        summary = service.run_evolution(
            runUuid="spec-07-three-generation",
            seed=29,
            evaluationConfiguration=evaluation,
            fitnessConfiguration=FitnessConfiguration(comboCap=4),
        )

        assert summary["status"] == "completed"
        assert len(summary["generations"]) == 3
        assert len({id(agent.dqn) for agent in service.agents}) == 12
        assert len({id(agent.dqn.policy_net) for agent in service.agents}) == 12
        assert len({id(agent.dqn.target_net) for agent in service.agents}) == 12
        assert len({id(agent.dqn.optimizer) for agent in service.agents}) == 12
        assert len({id(agent.dqn.replay_buffer) for agent in service.agents}) == 12
        assert len({id(agent.dqn.randomGenerator) for agent in service.agents}) == 12
        generations = store.list_generations(summary["runId"])
        assert [row["generation_index"] for row in generations] == [0, 1, 2]
        assert all(row["status"] == "completed" for row in generations)

        for index, generation in enumerate(summary["generations"]):
            agents = store.list_agents(generation["generationId"])
            assert generation["populationSize"] == len(agents) == 12
            assert len(generation["evaluation"]) == 12
            assert generation["evaluationMatchCount"] == 132
            assert generation["evaluationMaxConcurrency"] == 6
            assert all(item["sampleCount"] == 22 for item in generation["evaluation"])
            assert all(item["trainingStateUnchanged"] for item in generation["evaluation"])
            if index > 0:
                assert generation["eliteInheritanceVerified"] is True
            persisted = store.connection.execute(
                "SELECT fitness, metric_components_json FROM evaluations "
                "WHERE run_id = ? AND agent_id IN "
                "(SELECT id FROM agents WHERE generation_id = ?) ORDER BY agent_id",
                (summary["runId"], generation["generationId"]),
            ).fetchall()
            assert len(persisted) == 12
            assert [row["fitness"] for row in persisted] == pytest.approx(
                [item["fitness"] for item in generation["evaluation"]]
            )
            assert all(
                all(0.0 <= value <= 1.0 for value in json.loads(row["metric_components_json"]).values())
                for row in persisted
            )
            if index < 2:
                assert len(generation["selectedParentIndices"]) == 6
                assert len(generation["eliteParentIndices"]) == 2
                assert generation["offspringCount"] == 10
                selected = set(generation["selectedParentIndices"])
                scores = [item["fitness"] for item in generation["evaluation"]]
                assert min(scores[i] for i in selected) >= max(
                    scores[i] for i in range(12) if i not in selected
                )
            else:
                assert generation["offspringCount"] == 0

        for generation in generations[1:]:
            agents = store.list_agents(generation["id"])
            assert [agent["role"] for agent in agents[:2]] == ["elite", "elite"]
            assert all(agent["role"] == "offspring" for agent in agents[2:])
            for agent in agents:
                architecture = json.loads(agent["architecture_json"])
                Genome(tuple(architecture["hiddenWidths"])).validate(1, 4, {32, 64, 128, 256})
                parentCount = store.connection.execute(
                    "SELECT COUNT(*) FROM parentage WHERE child_agent_id = ?", (agent["id"],)
                ).fetchone()[0]
                assert parentCount >= 1

        assert store.connection.execute(
            "SELECT COUNT(*) FROM evaluations WHERE run_id = ?", (summary["runId"],)
        ).fetchone()[0] == 36

        evaluationMatches = store.connection.execute(
            "SELECT agent_a_id, agent_b_id, result_json FROM matches "
            "WHERE run_id = ? AND mode = 'evaluation_round_robin'",
            (summary["runId"],),
        ).fetchall()
        assert len(evaluationMatches) == 396
        assert all(
            len([
                row for row in evaluationMatches
                if row["agent_a_id"] == agent["id"] or row["agent_b_id"] == agent["id"]
            ]) == 22
            for generation in generations
            for agent in store.list_agents(generation["id"])
        )

        firstSnapshots = {}
        pairings = {}
        for event in events:
            if event.get("type") != "match_snapshot":
                continue
            generation = event["generationId"]
            firstSnapshots.setdefault((generation, event["arenaId"]), event)
            pairings.setdefault(generation, set()).add(
                (event["agentA"]["id"], event["agentB"]["id"])
            )
        assert len(firstSnapshots) == 18
        assert all(
            snapshot["agentA"]["score"] == snapshot["agentB"]["score"] == 0
            for snapshot in firstSnapshots.values()
        )
        assert len({frozenset(items) for items in pairings.values()}) == 3
