from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import math
import random
from typing import Sequence

from .genome import Genome

_SEED_MIX_PRIME = 1_006_109


def _generation_seed(configuration_seed: int, call_index: int) -> int:
    return configuration_seed * _SEED_MIX_PRIME + call_index


@dataclass(frozen=True)
class GeneticConfiguration:
    populationSize: int = 10
    parentPoolSize: int = 4
    elitismCount: int = 2
    offspringCount: int = 8
    mutationProbability: float = 0.20
    minHiddenLayers: int = 1
    maxHiddenLayers: int = 4
    allowedWidths: tuple[int, ...] = (32, 64, 128, 256)
    seed: int = 42
    weightTransfer: str = "compatible_tensors_only"

    def validate(self) -> None:
        if self.populationSize != self.elitismCount + self.offspringCount:
            raise ValueError("populationSize must equal elitismCount plus offspringCount")
        if self.populationSize <= 0:
            raise ValueError("populationSize must be greater than zero")
        if not 1 <= self.parentPoolSize <= self.populationSize:
            raise ValueError("parentPoolSize must be between one and populationSize")
        if not 1 <= self.elitismCount <= self.parentPoolSize:
            raise ValueError("elitismCount must be between one and parentPoolSize")
        if self.offspringCount <= 0:
            raise ValueError("offspringCount must be greater than zero")
        if not 0.0 <= self.mutationProbability <= 1.0:
            raise ValueError("mutationProbability must be between zero and one")
        if not 1 <= self.minHiddenLayers <= self.maxHiddenLayers:
            raise ValueError("hidden layer bounds must satisfy 1 <= min <= max")
        if not self.allowedWidths or any(width <= 0 for width in self.allowedWidths):
            raise ValueError("allowedWidths must be a non-empty tuple of positive integers")
        if self.weightTransfer != "compatible_tensors_only":
            raise ValueError("weightTransfer must be 'compatible_tensors_only'")


@dataclass(frozen=True)
class EliteRecord:
    childIndex: int
    parentIndex: int
    genome: Genome


@dataclass(frozen=True)
class OffspringRecord:
    childIndex: int
    parentAIndex: int
    parentBIndex: int
    crossoverPoint: int
    mutation: dict | None
    genome: Genome


@dataclass(frozen=True)
class GenerationPlan:
    genomes: tuple[Genome, ...]
    eliteIndices: tuple[int, ...]
    parentPoolIndices: tuple[int, ...]
    elites: tuple[EliteRecord, ...]
    offspring: tuple[OffspringRecord, ...]
    generationIndex: int
    generationSeed: int

    def validate(self, configuration: GeneticConfiguration) -> None:
        if len(self.genomes) != configuration.populationSize:
            raise ValueError("plan genome count does not match population size")
        if len(self.elites) != configuration.elitismCount:
            raise ValueError("plan elite count does not match configuration")
        if len(self.offspring) != configuration.offspringCount:
            raise ValueError("plan offspring count does not match configuration")
        child_indices = [record.childIndex for record in (*self.elites, *self.offspring)]
        if sorted(child_indices) != list(range(configuration.populationSize)):
            raise ValueError("plan child indices must partition the next population exactly")
        allowed = set(configuration.allowedWidths)
        for genome in self.genomes:
            genome.validate(configuration.minHiddenLayers, configuration.maxHiddenLayers, allowed)


class GeneticAlgorithm:
    """Stateless reproduction engine. Each call derives a deterministic RNG from
    configuration.seed and an internal call counter, so any generation is
    reproducible from the recorded generationSeed."""

    def __init__(self, configuration: GeneticConfiguration | None = None):
        self.configuration = configuration or GeneticConfiguration()
        self.configuration.validate()
        self._call_counter = 0
        self._generation_count = 0

    def _rng_at(self) -> tuple[random.Random, int]:
        generation_seed = _generation_seed(self.configuration.seed, self._call_counter)
        return random.Random(generation_seed), generation_seed

    def _advance(self) -> tuple[random.Random, int]:
        rng, generation_seed = self._rng_at()
        self._call_counter += 1
        return rng, generation_seed

    def _rank(self, fitness: Sequence[float], rng: random.Random) -> tuple[int, ...]:
        tiebreakers = [rng.random() for _ in range(len(fitness))]
        return tuple(
            sorted(
                range(len(fitness)),
                key=lambda index: (fitness[index], tiebreakers[index]),
                reverse=True,
            )
        )

    def initial_population(self) -> tuple[Genome, ...]:
        allowed = self.configuration.allowedWidths
        pool = [
            Genome(widths)
            for depth in range(self.configuration.minHiddenLayers, self.configuration.maxHiddenLayers + 1)
            for widths in itertools.product(allowed, repeat=depth)
        ]
        rng, _ = self._advance()
        sample_size = min(self.configuration.populationSize, len(pool))
        genomes = list(rng.sample(pool, sample_size))
        while len(genomes) < self.configuration.populationSize:
            genomes.append(rng.choice(pool))
        return tuple(genomes)

    def rank_indices(self, fitness: Sequence[float]) -> tuple[int, ...]:
        self._validate_fitness(fitness)
        rng, _ = self._advance()
        return self._rank(fitness, rng)

    def crossover(
        self,
        parent_a: Genome,
        parent_b: Genome,
        rng: random.Random,
    ) -> tuple[Genome, int]:
        max_point = min(parent_a.depth, parent_b.depth)
        crossover_point = rng.randint(0, max_point)
        child = Genome(parent_a.hiddenWidths[:crossover_point] + parent_b.hiddenWidths[crossover_point:])
        return child, crossover_point

    def mutate(self, genome: Genome, rng: random.Random) -> tuple[Genome, dict]:
        configuration = self.configuration
        widths = list(genome.hiddenWidths)
        operators: list[str] = []
        if len(widths) < configuration.maxHiddenLayers:
            operators.append("add_layer")
        if len(widths) > configuration.minHiddenLayers:
            operators.append("remove_layer")
        operators.append("change_width")
        operator = rng.choice(operators)
        mutation: dict = {"operator": operator}
        if operator == "add_layer":
            index = rng.randint(0, len(widths))
            width = rng.choice(configuration.allowedWidths)
            mutation.update({"index": index, "from": None, "to": width})
            widths.insert(index, width)
        elif operator == "remove_layer":
            index = rng.randint(0, len(widths) - 1)
            mutation.update({"index": index, "from": widths[index], "to": None})
            widths.pop(index)
        else:
            index = rng.randint(0, len(widths) - 1)
            alternatives = [w for w in configuration.allowedWidths if w != widths[index]]
            if alternatives:
                width = rng.choice(alternatives)
                mutation.update({"index": index, "from": widths[index], "to": width})
                widths[index] = width
        return Genome(tuple(widths)), mutation

    def next_generation(self, previous_genomes: Sequence[Genome], fitness: Sequence[float]) -> GenerationPlan:
        if len(previous_genomes) != self.configuration.populationSize:
            raise ValueError("previous_genomes must match the population size")
        self._validate_fitness(fitness)
        rng, generation_seed = self._rng_at()
        ranked = self._rank(fitness, rng)
        parent_pool = ranked[: self.configuration.parentPoolSize]
        elites = ranked[: self.configuration.elitismCount]

        genomes: list[Genome] = []
        elite_records: list[EliteRecord] = []
        for position, parent_index in enumerate(elites):
            genome = previous_genomes[parent_index]
            genomes.append(genome)
            elite_records.append(EliteRecord(childIndex=position, parentIndex=parent_index, genome=genome))

        offspring_records: list[OffspringRecord] = []
        for position in range(self.configuration.elitismCount, self.configuration.populationSize):
            parent_a_index = rng.choice(parent_pool)
            parent_b_index = rng.choice(parent_pool)
            child, crossover_point = self.crossover(
                previous_genomes[parent_a_index], previous_genomes[parent_b_index], rng
            )
            mutation: dict | None = None
            if rng.random() < self.configuration.mutationProbability:
                child, mutation = self.mutate(child, rng)
            genomes.append(child)
            offspring_records.append(
                OffspringRecord(
                    childIndex=position,
                    parentAIndex=parent_a_index,
                    parentBIndex=parent_b_index,
                    crossoverPoint=crossover_point,
                    mutation=mutation,
                    genome=child,
                )
            )

        self._call_counter += 1
        self._generation_count += 1
        plan = GenerationPlan(
            genomes=tuple(genomes),
            eliteIndices=tuple(range(self.configuration.elitismCount)),
            parentPoolIndices=tuple(parent_pool),
            elites=tuple(elite_records),
            offspring=tuple(offspring_records),
            generationIndex=self._generation_count,
            generationSeed=generation_seed,
        )
        plan.validate(self.configuration)
        return plan

    def _validate_fitness(self, fitness: Sequence[float]) -> None:
        if len(fitness) != self.configuration.populationSize:
            raise ValueError(
                f"fitness length {len(fitness)} does not match population size {self.configuration.populationSize}"
            )
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in fitness):
            raise ValueError("fitness values must be finite numbers")


def lineage_records(
    plan: GenerationPlan,
    previous_agent_ids: Sequence[int],
    new_agent_ids: Sequence[int],
) -> list[dict]:
    """Parentage rows for SPEC-01 store.record_parentage; includes both parents,
    crossover point, mutation, and the recorded generation seed."""
    if len(previous_agent_ids) != len(plan.genomes) or len(new_agent_ids) != len(plan.genomes):
        raise ValueError("agent id sequences must match the population size")
    rows: list[dict] = []
    for elite in plan.elites:
        rows.append(
            {
                "child_agent_id": new_agent_ids[elite.childIndex],
                "parent_agent_id": previous_agent_ids[elite.parentIndex],
                "mutation_json": json.dumps(
                    {
                        "role": "elite_clone",
                        "childGenome": elite.genome.to_json(),
                        "generationSeed": plan.generationSeed,
                    },
                    sort_keys=True,
                ),
            }
        )
    for offspring in plan.offspring:
        evidence = json.dumps(
            {
                "role": "offspring",
                "childGenome": offspring.genome.to_json(),
                "parentA": offspring.parentAIndex,
                "parentB": offspring.parentBIndex,
                "crossoverPoint": offspring.crossoverPoint,
                "mutation": offspring.mutation,
                "generationSeed": plan.generationSeed,
            },
            sort_keys=True,
        )
        rows.append(
            {
                "child_agent_id": new_agent_ids[offspring.childIndex],
                "parent_agent_id": previous_agent_ids[offspring.parentAIndex],
                "mutation_json": evidence,
            }
        )
        rows.append(
            {
                "child_agent_id": new_agent_ids[offspring.childIndex],
                "parent_agent_id": previous_agent_ids[offspring.parentBIndex],
                "mutation_json": evidence,
            }
        )
    return rows


def genome_architecture_json(genome: Genome) -> str:
    return json.dumps(genome.to_json(), sort_keys=True)
