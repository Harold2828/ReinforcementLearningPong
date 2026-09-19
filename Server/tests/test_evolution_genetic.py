import json

import pytest

from app.evolution.genetic import (
    GeneticAlgorithm,
    GeneticConfiguration,
    lineage_records,
)
from app.evolution.genome import Genome
from app.persistence.store import EvolutionStore


def seeded(seed: int = 42, **overrides) -> GeneticAlgorithm:
    return GeneticAlgorithm(GeneticConfiguration(seed=seed, **overrides))


def test_configuration_rejects_invalid_population_splits():
    with pytest.raises(ValueError):
        GeneticConfiguration(populationSize=10, elitismCount=3, offspringCount=8).validate()
    with pytest.raises(ValueError):
        GeneticConfiguration(parentPoolSize=0).validate()
    with pytest.raises(ValueError):
        GeneticConfiguration(mutationProbability=-0.1).validate()


def test_genome_rejects_out_of_bounds_depth_and_widths():
    with pytest.raises(ValueError):
        Genome(())
    with pytest.raises(ValueError):
        Genome((0, 64))
    with pytest.raises(ValueError):
        Genome((64, 64, 64, 64, 64)).validate(1, 4, {32, 64, 128, 256})  # depth 5
    with pytest.raises(ValueError):
        Genome((32, 16)).validate(1, 4, {32, 64, 128, 256})  # unsupported width
    Genome((64, 128, 64)).validate(1, 4, {32, 64, 128, 256})  # valid


def test_initial_population_is_distinct_and_valid():
    ga = seeded()
    population = ga.initial_population()
    assert len(population) == 10
    assert len({genome for genome in population}) == 10
    allowed = set(ga.configuration.allowedWidths)
    for genome in population:
        genome.validate(
            ga.configuration.minHiddenLayers,
            ga.configuration.maxHiddenLayers,
            allowed,
        )


def test_one_hundred_seeded_transitions_keep_critical_invariants():
    for seed in range(1, 101):
        ga = seeded(seed=seed)
        population = list(ga.initial_population())
        fitness = [seed + index / 10.0 for index in range(10)]
        plan = ga.next_generation(population, fitness)

        assert plan.generationIndex == 1  # one transition per GA instance
        assert len(plan.genomes) == 10
        assert len(plan.elites) == 2
        assert len(plan.offspring) == 8
        assert len(plan.parentPoolIndices) == 4
        assert len({offspring.childIndex for offspring in plan.offspring} | set(plan.eliteIndices)) == 10
        for genome in plan.genomes:
            assert 1 <= genome.depth <= 4
            assert all(width in ga.configuration.allowedWidths for width in genome.hiddenWidths)
        plan.validate(ga.configuration)


def test_elites_preserved_unchanged():
    ga = seeded(seed=7)
    population = list(ga.initial_population())
    fitness = [float(index % 3) for index in range(10)]  # heavy ties exercise tie-breaks
    plan = ga.next_generation(population, fitness)

    assert len(plan.parentPoolIndices) == 4
    assert set(plan.parentPoolIndices).issubset(range(10))
    for elite in plan.elites:
        assert elite.genome == population[elite.parentIndex]
    elite_positions = sorted(elite.childIndex for elite in plan.elites)
    assert elite_positions == [0, 1]


def test_same_seed_is_reproducible():
    first = seeded(seed=123)
    second = seeded(seed=123)
    fitness = [float(index) for index in range(10)]
    population_a = first.initial_population()
    population_b = second.initial_population()
    plan_a = first.next_generation(population_a, fitness)
    plan_b = second.next_generation(population_b, fitness)

    assert population_a == population_b
    assert plan_a.genomes == plan_b.genomes
    assert plan_a.parentPoolIndices == plan_b.parentPoolIndices
    assert plan_a.generationSeed == plan_b.generationSeed
    assert [off.mutation for off in plan_a.offspring] == [off.mutation for off in plan_b.offspring]
    assert [off.crossoverPoint for off in plan_a.offspring] == [off.crossoverPoint for off in plan_b.offspring]


def test_forced_mutation_applies_real_operators_within_bounds():
    ga = seeded(seed=99, mutationProbability=1.0)
    population = list(ga.initial_population())
    plan = ga.next_generation(population, [float(index) for index in range(10)])

    allowed = set(ga.configuration.allowedWidths)
    for offspring in plan.offspring:
        assert offspring.mutation is not None
        assert offspring.mutation["operator"] in {"add_layer", "remove_layer", "change_width"}
        assert 1 <= offspring.genome.depth <= 4
        assert all(width in allowed for width in offspring.genome.hiddenWidths)


def test_lineage_records_integrate_with_spec01_store(tmp_path):
    store = EvolutionStore(db_path=tmp_path / "evolution.sqlite3", checkpoint_root=tmp_path / "checkpoints")
    run_id = store.create_run(
        run_uuid="run-ga-1",
        config={"population": 10},
        seed=42,
        code_revision="spec02",
        fitness_formula="0.65*W + 0.25*D + 0.10*C",
        benchmark_definition="bench-v1",
    )
    generation = store.start_generation(run_id, 0)

    ga = seeded(seed=5)
    previous = ga.initial_population()
    previous_ids = [
        store.register_agent(
            f"agent-{generation}-{index}",
            generation,
            "initial",
            json.dumps(genome.to_json(), sort_keys=True),
        )
        for index, genome in enumerate(previous)
    ]

    plan = ga.next_generation(previous, [float(index) for index in range(10)])
    next_generation_id = store.start_generation(run_id, 1)
    new_ids = [
        store.register_agent(
            f"agent-{next_generation_id}-{index}",
            next_generation_id,
            "offspring",
            json.dumps(genome.to_json(), sort_keys=True),
        )
        for index, genome in enumerate(plan.genomes)
    ]

    rows = lineage_records(plan, previous_ids, new_ids)
    assert len(rows) == 2 + 2 * 8  # elite clones + two parent rows per offspring
    for row in rows:
        store.record_parentage(child_agent_id=row["child_agent_id"], parent_agent_id=row["parent_agent_id"], mutation_json=row["mutation_json"])

    # UNIQUE(child_agent_id, parent_agent_id) collapses self-crossover duplicates.
    distinct_pairs = len({(row["child_agent_id"], row["parent_agent_id"]) for row in rows})
    stored = store.connection.execute("SELECT COUNT(*) AS c FROM parentage").fetchone()["c"]
    assert stored == distinct_pairs

    evidence = json.loads(rows[2]["mutation_json"])  # first offspring parent-A row
    assert "crossoverPoint" in evidence
    assert "mutation" in evidence or evidence["mutation"] is None
    assert "generationSeed" in evidence
    assert evidence["childGenome"]["hiddenWidths"] == list(plan.offspring[0].genome.hiddenWidths)
    store.close()