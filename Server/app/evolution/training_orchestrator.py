"""SPEC-06 evolutionary training orchestration.

Runs the SPEC-02 population as isolated SPEC-04 DQN agents competing on the
SPEC-05 headless engine with round-robin opponent/side rotation, persisting
matches and v3 checkpoints to the SPEC-01 store and streaming LIVE snapshots
and metrics to SPEC-03 over Socket.IO.

Execution is modelled as concurrent matches but executed sequentially (CPU);
each round every agent plays exactly one match, so step budgets stay balanced
within one round of overshoot.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import json
from pathlib import Path
import random
import tempfile
import threading
import time
from typing import Callable, Sequence

import numpy as np
import torch

from ..ai.dqn_agent import DQNConfiguration, DQNAgent, pong_state_to_vector
from ..ai.multi_agent_training_service import TRAINING_SELF_PLAY
from ..ai.q_learning_agent import PongState
from ..persistence.store import EvolutionStore
from .genetic import GeneticAlgorithm, GeneticConfiguration, GenerationPlan, lineage_records
from .genome import Genome
from .simulation import (
    ARENA_IDS,
    TERMINAL,
    VELOCITY_SCALE,
    MatchAssignment,
    MatchConfig,
    MatchParticipant,
    SimulationEngine,
)

AGENT_ROLE_INITIAL = "initial"
POINT_REWARD = 5.0
BASE_REWARD = 0.01
MODEL_SPEC_VERSION = 3
CHECKPOINT_TYPE_FULL = "full"
EVENT_SOURCE_LIVE = "LIVE"

ENVELOPE_AGENT_IDENTITY = "agent"
ENVELOPE_OPPONENT_IDENTITY = "opponent"


def _mix(*values: int) -> int:
    acc = 0x9E3779B9
    for value in values:
        acc = ((acc ^ int(value)) * 1_000_003) & 0xFFFFFFFF
    return acc & 0x7FFFFFFF


def round_robin_pairs(populationSize: int, roundIndex: int) -> list[tuple[int, int, bool]]:
    """Circle-method pairing for one round.

    Returns (agentIndex, opponentIndex, reversedSides). Every index is paired
    exactly once per round, and over a full cycle of (populationSize - 1)
    rounds every agent meets every other agent exactly once with both sides
    exercised.
    """
    if populationSize < 2:
        raise ValueError("round-robin needs a population of at least two")
    if populationSize % 2 != 0:
        raise ValueError(f"round-robin needs an even population, received {populationSize}")
    fixed = 0
    rotating = list(range(1, populationSize))
    offset = roundIndex % len(rotating)
    rotated = rotating[offset:] + rotating[:offset]
    reversedFlag = roundIndex % 2 == 1
    pairs = [(fixed, rotated[0], reversedFlag)]
    width = len(rotated) // 2
    for i in range(1, width + 1):
        pairs.append((rotated[i], rotated[len(rotated) - i], reversedFlag))
    return pairs


def identity_rewards(envelope: dict) -> tuple[float, float]:
    """Differential identity rewards: (agentA, agentB) for the given envelope.

    Agent labels are identity labels (the SPEC-05 envelope maps pointWinner to
    A/B regardless of physical paddle side), so rewards never cross identities.
    """
    rewardA = BASE_REWARD
    rewardB = BASE_REWARD
    if envelope["pointWinner"] == ENVELOPE_AGENT_IDENTITY:
        rewardA += POINT_REWARD
        rewardB -= POINT_REWARD
    elif envelope["pointWinner"] == ENVELOPE_OPPONENT_IDENTITY:
        rewardA -= POINT_REWARD
        rewardB += POINT_REWARD
    return rewardA, rewardB


def pong_state_from_envelope(envelope: dict, side: str, config: MatchConfig) -> PongState:
    """Player-centric PongState for an identity side built from the envelope.

    The envelope is already in agentA's identity frame; side B mirrors it back.
    """
    width = config.width
    height = config.height
    ball = envelope["ball"]
    agentA = envelope["agentA"]
    agentB = envelope["agentB"]
    ballVelocityScale = VELOCITY_SCALE
    ballX = ball["x"] * width
    ballY = ball["y"] * height
    ballVX = ball["vx"] * ballVelocityScale
    ballVY = ball["vy"] * ballVelocityScale
    done = envelope["pointWinner"] is not None

    if side == "A":
        return PongState(
            ballX=ballX,
            ballY=ballY,
            ballVelocityX=ballVX,
            ballVelocityY=ballVY,
            paddleY=agentA["paddleY"] * height,
            opponentPaddleY=agentB["paddleY"] * height,
            scoreAgent=agentA["score"],
            scoreOpponent=agentB["score"],
            done=done,
            width=width,
            height=height,
        )
    if side == "B":
        return PongState(
            ballX=width - ballX,
            ballY=ballY,
            ballVelocityX=-ballVX,
            ballVelocityY=ballVY,
            paddleY=agentB["paddleY"] * height,
            opponentPaddleY=agentA["paddleY"] * height,
            scoreAgent=agentB["score"],
            scoreOpponent=agentA["score"],
            done=done,
            width=width,
            height=height,
        )
    raise ValueError(f"unknown identity side {side!r}")


@dataclass(frozen=True)
class EvolutionTrainingConfiguration:
    stepsPerAgentPerGeneration: int = 100_000
    roundTicks: int = 1_000
    maxGenerations: int = 1
    snapshotInterval: int = 10
    metricsInterval: int = 50
    winScore: int = 7
    device: str = "cpu"

    def validate(self) -> None:
        if self.stepsPerAgentPerGeneration <= 0:
            raise ValueError("stepsPerAgentPerGeneration must be greater than zero")
        if self.roundTicks <= 0:
            raise ValueError("roundTicks must be greater than zero")
        if self.maxGenerations <= 0:
            raise ValueError("maxGenerations must be greater than zero")
        if self.snapshotInterval <= 0:
            raise ValueError("snapshotInterval must be greater than zero")
        if self.metricsInterval <= 0:
            raise ValueError("metricsInterval must be greater than zero")
        if self.winScore <= 0:
            raise ValueError("winScore must be greater than zero")


@dataclass
class AgentRecord:
    index: int
    storeId: int
    dqn: DQNAgent
    generation: int = 0
    role: str = AGENT_ROLE_INITIAL
    inheritance: dict | None = None
    previousState: np.ndarray | None = None
    previousAction: int | None = None
    hits: int = 0
    wins: int = 0
    losses: int = 0
    cumulativeReward: float = 0.0


class RunController:
    """Pause/resume/stop semantics honored between rounds."""

    def __init__(self) -> None:
        self._stopRequested = False
        self._paused = threading.Event()
        self._resumed = threading.Event()

    @property
    def stopRequested(self) -> bool:
        return self._stopRequested

    def request_stop(self) -> None:
        self._stopRequested = True
        self._resumed.set()

    def reset(self) -> None:
        self._stopRequested = False
        self._paused.clear()
        self._resumed.clear()

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()
        self._resumed.set()
        self._resumed.clear()

    def wait_if_paused(self, checkEverySeconds: float = 0.05) -> bool:
        while self._paused.is_set():
            if self._stopRequested:
                return False
            time.sleep(checkEverySeconds)
        return not self._stopRequested


class EvolutionTrainingService:
    algorithmName = "evolutionary_dqn"

    def __init__(
        self,
        store: EvolutionStore,
        configuration: EvolutionTrainingConfiguration | None = None,
        geneticConfiguration: GeneticConfiguration | None = None,
        dqnConfiguration: DQNConfiguration | None = None,
        codeRevision: str = "",
        benchmarkDefinition: str = "spec-06-isolated-round-robin",
        genomes: Sequence[Genome] | None = None,
        onEvent: Callable[[dict], None] | None = None,
        evaluationConfiguration=None,
        fitnessConfiguration=None,
    ):
        from .evaluation import EvaluationConfiguration, FitnessConfiguration

        self.store = store
        self.configuration = configuration or EvolutionTrainingConfiguration()
        self.configuration.validate()
        self.geneticConfiguration = geneticConfiguration or GeneticConfiguration()
        self.geneticConfiguration.validate()
        self.dqnConfiguration = dqnConfiguration or DQNConfiguration()
        self.dqnConfiguration.validate()
        self.matchConfig = MatchConfig(winScore=self.configuration.winScore)
        self.codeRevision = codeRevision
        self.benchmarkDefinition = benchmarkDefinition
        self.on_event = onEvent
        self.evaluationConfiguration = evaluationConfiguration or EvaluationConfiguration()
        self.fitnessConfiguration = fitnessConfiguration or FitnessConfiguration()
        self.controller = RunController()
        self.geneticAlgorithm = GeneticAlgorithm(self.geneticConfiguration)
        provided = list(genomes) if genomes is not None else None
        self.genomes: tuple[Genome, ...] = (
            tuple(provided)
            if provided is not None
            else self.geneticAlgorithm.initial_population()
        )
        targetPopulation = 2 * len(ARENA_IDS)
        if len(self.genomes) != targetPopulation:
            raise ValueError(
                f"evolution population must be {targetPopulation} (2 per SPEC-05 arena), "
                f"got {len(self.genomes)}"
            )
        self.agents: list[AgentRecord] = []
        self.runId: int | None = None
        self.generationId: int | None = None
        self.runUuid: str | None = None
        self.generationIndex = 0
        self._roundIndex = 0
        self._sideLastReturns: dict[tuple[str, str], int] = {}
        self._lastSequences: dict[str, int] = {}
        self._eliteInheritanceVerified = True

    def _emit(self, event: dict) -> None:
        if self.on_event is not None:
            self.on_event(event)

    def run_generation(
        self,
        runUuid: str = "spec-06-run",
        seed: int = 0,
        fitnessFormula: str = "",
        maxRounds: int | None = None,
    ) -> dict:
        self._roundIndex = 0
        self.generationIndex = 0
        self.runUuid = runUuid
        self.agents = []
        self.runId = self.store.create_run(
            runUuid,
            asdict(self.configuration),
            seed,
            self.codeRevision,
            fitnessFormula,
            self.benchmarkDefinition,
        )
        self.generationId = self.store.start_generation(self.runId, 0)
        self._build_agents(seed)
        self._emit_population()

        budget = self.configuration.stepsPerAgentPerGeneration
        cycle = len(self.agents) - 1
        if maxRounds is not None and maxRounds <= 0:
            raise ValueError("maxRounds must be greater than zero")
        roundCap: int = maxRounds or (
            int(np.ceil(budget / self.configuration.roundTicks)) + cycle
        )
        try:
            while self._roundIndex < roundCap and self._any_below_budget(budget):
                if not self.controller.wait_if_paused():
                    break
                self._roundIndex += 1
                self._run_round(seed, self._roundIndex)
                if self._roundIndex % self.configuration.metricsInterval == 0:
                    self._emit_metrics()
                if self.controller.stopRequested:
                    break
        except BaseException:
            self.store.set_run_status(self.runId, "cancelled")
            self.store.abort_generation(self.generationId)
            raise

        self._save_checkpoints()
        completed = not self.controller.stopRequested and not self._any_below_budget(budget)
        if completed:
            self.store.complete_generation(self.generationId)
            self.store.set_run_status(self.runId, "completed")
        else:
            self.store.set_run_status(self.runId, "cancelled")
            self.store.abort_generation(self.generationId)
        self._emit_metrics()
        return {
            "runId": self.runId,
            "runUuid": runUuid,
            "generationId": self.generationId,
            "rounds": self._roundIndex,
            "status": "completed" if completed else "cancelled",
            "agents": self._agent_summaries(),
        }

    def run_evolution(
        self,
        runUuid: str = "spec-07-run",
        seed: int = 0,
        fitnessFormula: str = "0.65*winRate + 0.25*pointDifferential + 0.10*comboPerformance",
        evaluationConfiguration=None,
        fitnessConfiguration=None,
    ) -> dict:
        """Train, evaluate, and reproduce a complete configurable run."""
        from .evaluation import EvaluationConfiguration, FitnessConfiguration, FixedOpponentEvaluator

        self.controller.reset()
        evaluationConfiguration = evaluationConfiguration or self.evaluationConfiguration
        fitnessConfiguration = fitnessConfiguration or self.fitnessConfiguration
        evaluator = FixedOpponentEvaluator(evaluationConfiguration, fitnessConfiguration, self.matchConfig)
        self.runUuid = runUuid
        self.runId = self.store.create_run(
            runUuid,
            {
                **asdict(self.configuration),
                "evaluation": asdict(evaluationConfiguration),
                "fitness": asdict(fitnessConfiguration),
            },
            seed,
            self.codeRevision,
            fitnessFormula,
            evaluationConfiguration.benchmarkVersion,
        )
        generationSummaries = []
        pendingPlan: GenerationPlan | None = None
        previousAgents: list[AgentRecord] = []
        try:
            for generationIndex in range(self.configuration.maxGenerations):
                self.generationIndex = generationIndex
                self._roundIndex = 0
                self.generationId = self.store.start_generation(self.runId, generationIndex)
                self.agents = []
                self._eliteInheritanceVerified = True
                self._build_agents(
                    _mix(seed, generationIndex),
                    generationIndex=generationIndex,
                    plan=pendingPlan,
                    previousAgents=previousAgents,
                )
                if pendingPlan is not None:
                    for row in lineage_records(
                        pendingPlan,
                        [agent.storeId for agent in previousAgents],
                        [agent.storeId for agent in self.agents],
                    ):
                        self.store.record_parentage(**row)
                self._emit_population()
                self._train_to_budget(seed)
                if self.controller.stopRequested:
                    self.store.abort_generation(self.generationId)
                    self.store.set_run_status(self.runId, "cancelled")
                    break

                evaluation = evaluator.evaluate(
                    [agent.dqn for agent in self.agents], generationIndex
                )
                for agent, result in zip(self.agents, evaluation):
                    self.store.record_evaluation(
                        agent.storeId,
                        self.runId,
                        evaluationConfiguration.benchmarkVersion,
                        {
                            key: result[key]
                            for key in ("winRate", "pointDifferential", "comboPerformance")
                        },
                        result["sampleCount"],
                        {"stepCapTerminations": result["stepCapTerminations"]},
                        result["fitness"],
                    )
                self._save_checkpoints()
                self.store.complete_generation(self.generationId)

                nextPlan = None
                if generationIndex + 1 < self.configuration.maxGenerations:
                    nextPlan = self.geneticAlgorithm.next_generation(
                        self.genomes, [item["fitness"] for item in evaluation]
                    )
                generationSummaries.append(
                    {
                        "generation": generationIndex,
                        "generationId": self.generationId,
                        "populationSize": len(self.agents),
                        "rounds": self._roundIndex,
                        "evaluation": evaluation,
                        "selectedParentIndices": (
                            list(nextPlan.parentPoolIndices) if nextPlan is not None else []
                        ),
                        "eliteParentIndices": (
                            [elite.parentIndex for elite in nextPlan.elites] if nextPlan is not None else []
                        ),
                        "offspringCount": len(nextPlan.offspring) if nextPlan is not None else 0,
                        "eliteInheritanceVerified": self._eliteInheritanceVerified,
                    }
                )
                self._emit(
                    {
                        "type": "evaluation",
                        "source": EVENT_SOURCE_LIVE,
                        "runId": self.runId,
                        "generationId": self.generationId,
                        "benchmark": evaluationConfiguration.benchmarkVersion,
                        "results": [
                            {"agentId": agent.index, "fitness": result}
                            for agent, result in zip(self.agents, evaluation)
                        ],
                    }
                )
                previousAgents = self.agents
                pendingPlan = nextPlan
                if nextPlan is not None:
                    self.genomes = nextPlan.genomes
            completed = len(generationSummaries) == self.configuration.maxGenerations
            self.store.set_run_status(self.runId, "completed" if completed else "cancelled")
            return {
                "runId": self.runId,
                "runUuid": runUuid,
                "status": "completed" if completed else "cancelled",
                "generations": generationSummaries,
            }
        except BaseException:
            if self.generationId is not None:
                self.store.abort_generation(self.generationId)
            self.store.set_run_status(self.runId, "cancelled")
            raise

    def _train_to_budget(self, seed: int) -> None:
        budget = self.configuration.stepsPerAgentPerGeneration
        cycle = len(self.agents) - 1
        roundCap = int(np.ceil(budget / self.configuration.roundTicks)) + cycle
        while self._roundIndex < roundCap and self._any_below_budget(budget):
            if not self.controller.wait_if_paused():
                return
            self._roundIndex += 1
            self._run_round(seed, self._roundIndex)
            if self._roundIndex % self.configuration.metricsInterval == 0:
                self._emit_metrics()
            if self.controller.stopRequested:
                return

    def _build_agents(
        self,
        seed: int,
        generationIndex: int = 0,
        plan: GenerationPlan | None = None,
        previousAgents: Sequence[AgentRecord] = (),
    ) -> None:
        for index, genome in enumerate(self.genomes):
            rng = random.Random(_mix(seed, index))
            torch.manual_seed(_mix(seed, index, generationIndex))
            dqn = DQNAgent(
                configuration=self.dqnConfiguration,
                hiddenWidths=genome,
                device=self.configuration.device,
                randomGenerator=rng,
            )
            role = AGENT_ROLE_INITIAL
            inheritance = None
            if plan is not None:
                role = "elite" if index < self.geneticConfiguration.elitismCount else "offspring"
                inheritance = self._inherit_weights(dqn, index, plan, previousAgents)
            storeId = self.store.register_agent(
                f"run-{self.runUuid}-gen-{generationIndex}-agent-{index}",
                self.generationId,
                role,
                json.dumps(genome.to_json()),
            )
            self.agents.append(
                AgentRecord(
                    index=index,
                    storeId=storeId,
                    dqn=dqn,
                    generation=generationIndex,
                    role=role,
                    inheritance=inheritance,
                )
            )

    @staticmethod
    def _inherit_weights(
        child: DQNAgent,
        childIndex: int,
        plan: GenerationPlan,
        previousAgents: Sequence[AgentRecord],
    ) -> dict:
        if childIndex < len(plan.elites):
            parentIndex = plan.elites[childIndex].parentIndex
            parent = previousAgents[parentIndex].dqn
            child.policy_net.load_state_dict(copy.deepcopy(parent.policy_net.state_dict()))
            child.target_net.load_state_dict(copy.deepcopy(parent.target_net.state_dict()))
            for childNetwork, parentNetwork in (
                (child.policy_net, parent.policy_net),
                (child.target_net, parent.target_net),
            ):
                if any(
                    not np.array_equal(
                        childNetwork.state_dict()[key].detach().cpu().numpy(),
                        tensor.detach().cpu().numpy(),
                    )
                    for key, tensor in parentNetwork.state_dict().items()
                ):
                    raise RuntimeError("elite tensor inheritance changed parameters")
            return {
                "policy": "elite_exact_copy",
                "parentIndex": parentIndex,
                "copiedKeys": sorted(parent.policy_net.state_dict()),
                "skippedKeys": [],
                "targetSynchronized": False,
            }
        offspring = next(item for item in plan.offspring if item.childIndex == childIndex)
        parent = previousAgents[offspring.parentAIndex].dqn
        childState = child.policy_net.state_dict()
        parentState = parent.policy_net.state_dict()
        compatible = {
            key: tensor.detach().clone()
            for key, tensor in parentState.items()
            if key in childState and childState[key].shape == tensor.shape
        }
        skipped = sorted(
            key
            for key in set(childState) | set(parentState)
            if key not in compatible
        )
        childState.update(compatible)
        child.policy_net.load_state_dict(childState)
        child.target_net.load_state_dict(copy.deepcopy(child.policy_net.state_dict()))
        return {
            "policy": "compatible_tensors_only",
            "parentIndex": offspring.parentAIndex,
            "copiedKeys": sorted(compatible),
            "skippedKeys": skipped,
            "targetSynchronized": True,
        }

    def _emit_population(self) -> None:
        self._emit(
            {
                "type": "population",
                "source": EVENT_SOURCE_LIVE,
                "runId": f"run-{self.runId}",
                "generationId": f"generation-{self.generationId}",
                "agents": self._agent_summaries(),
            }
        )

    def _any_below_budget(self, budget: int) -> bool:
        return any(agent.dqn.total_steps < budget for agent in self.agents)

    def _run_round(self, seed: int, roundIndex: int) -> list[dict]:
        pairs = round_robin_pairs(len(self.agents), roundIndex)
        assignments = []
        matchSeeds: list[int] = []
        for arenaIndex in range(len(self.agents) // 2):
            aIndex, bIndex, reversedSides = pairs[arenaIndex]
            matchSeed = _mix(seed, self.runId, roundIndex, arenaIndex)
            agentA = self.agents[aIndex]
            agentB = self.agents[bIndex]
            assignments.append(
                MatchAssignment(
                    arenaId=ARENA_IDS[arenaIndex],
                    runId=f"run-{self.runId}",
                    generationId=f"generation-{self.generationId}",
                    agentA=MatchParticipant(
                        agentId=f"agent-{aIndex}",
                        generation=self.generationIndex,
                        epsilon=float(agentA.dqn.epsilonValue),
                    ),
                    agentB=MatchParticipant(
                        agentId=f"agent-{bIndex}",
                        generation=self.generationIndex,
                        epsilon=float(agentB.dqn.epsilonValue),
                    ),
                    reversedSides=reversedSides,
                    seed=matchSeed,
                )
            )
            matchSeeds.append(matchSeed)

        engine = SimulationEngine(assignments, self.matchConfig)
        self._sideLastReturns = {}
        self._lastSequences = {}
        self._clear_previous_states()
        pendingActions = {arenaId: ("STAY", "STAY") for arenaId in ARENA_IDS}
        lastEnvelopes: list[dict] | None = None

        for tick in range(self.configuration.roundTicks):
            envelopes = engine.step(pendingActions)
            lastEnvelopes = envelopes
            for arenaIndex, envelope in enumerate(envelopes):
                aIndex, bIndex, _ = pairs[arenaIndex]
                arenaId = envelope["arenaId"]
                if envelope["sequence"] <= self._lastSequences.get(arenaId, 0):
                    continue
                self._lastSequences[arenaId] = envelope["sequence"]
                nextActions = self._advance_envelope(envelope, aIndex, bIndex)
                if envelope["status"] != TERMINAL:
                    pendingActions[arenaId] = nextActions
            if (tick + 1) % self.configuration.snapshotInterval == 0:
                for envelope in envelopes:
                    self._emit({**envelope, "source": EVENT_SOURCE_LIVE})

        self._record_matches(roundIndex, pairs, matchSeeds, lastEnvelopes or [])
        return lastEnvelopes

    def _advance_envelope(self, envelope: dict, aIndex: int, bIndex: int) -> tuple[str, str]:
        agentA = self.agents[aIndex]
        agentB = self.agents[bIndex]
        obsA = pong_state_to_vector(pong_state_from_envelope(envelope, "A", self.matchConfig))
        obsB = pong_state_to_vector(pong_state_from_envelope(envelope, "B", self.matchConfig))
        rewardA, rewardB = identity_rewards(envelope)
        done = envelope["pointWinner"] is not None

        self._apply_transition(agentA, obsA, rewardA, done)
        self._apply_transition(agentB, obsB, rewardB, done)
        self._track_hits(envelope, aIndex, bIndex)

        if envelope["pointWinner"] == ENVELOPE_AGENT_IDENTITY:
            agentA.wins += 1
            agentB.losses += 1
        elif envelope["pointWinner"] == ENVELOPE_OPPONENT_IDENTITY:
            agentB.wins += 1
            agentA.losses += 1

        actions = (
            DQNAgent.action_name(self._next_action(agentA, obsA)),
            DQNAgent.action_name(self._next_action(agentB, obsB)),
        )
        if done:
            # A point boundary ends the DQN episode. The action selected at the
            # terminal frame must not open a transition into the next serve, or
            # the buffer would learn a bootstrap across two different episodes.
            agentA.previousState = None
            agentA.previousAction = None
            agentB.previousState = None
            agentB.previousAction = None
        return actions

    def _apply_transition(self, agent: AgentRecord, obs: np.ndarray, reward: float, done: bool) -> None:
        if not self._should_learn(agent):
            return
        if agent.previousState is not None and agent.previousAction is not None:
            agent.dqn.remember(agent.previousState, agent.previousAction, reward, obs, done)
            agent.cumulativeReward += reward
            agent.dqn.train_step()

    def _next_action(self, agent: AgentRecord, obs: np.ndarray) -> int:
        actionIndex = agent.dqn.select_action(obs, explore=self._should_learn(agent))
        agent.previousState = obs
        agent.previousAction = actionIndex
        return actionIndex

    def _should_learn(self, agent: AgentRecord) -> bool:
        return agent.dqn.total_steps < self.configuration.stepsPerAgentPerGeneration

    def _clear_previous_states(self) -> None:
        for agent in self.agents:
            agent.previousState = None
            agent.previousAction = None

    def _track_hits(self, envelope: dict, aIndex: int, bIndex: int) -> None:
        for identity, envelopeKey, agentIndex in (
            ("A", "agentA", aIndex),
            ("B", "agentB", bIndex),
        ):
            returns = envelope[envelopeKey]["returns"]
            previous = self._sideLastReturns.get((envelope["arenaId"], identity))
            if previous is not None and returns > previous:
                self.agents[agentIndex].hits += returns - previous
            self._sideLastReturns[(envelope["arenaId"], identity)] = returns

    def _record_matches(
        self,
        roundIndex: int,
        pairs: list[tuple[int, int, bool]],
        matchSeeds: list[int],
        envelopes: list[dict],
    ) -> None:
        for arenaIndex, envelope in enumerate(envelopes):
            aIndex, bIndex, reversedSides = pairs[arenaIndex]
            pointWinner = envelope["pointWinner"]
            combo = envelope["combo"] if envelope["status"] == TERMINAL else 0
            self.store.record_match(
                match_uuid=f"match-{self.runUuid}-gen-{self.generationIndex}-round-{roundIndex}-arena-{arenaIndex}",
                run_id=self.runId,
                arena=envelope["arenaId"],
                agent_a_id=self.agents[aIndex].storeId,
                agent_b_id=self.agents[bIndex].storeId,
                game_seed=matchSeeds[arenaIndex],
                mode=TRAINING_SELF_PLAY,
                score_a=envelope["agentA"]["score"],
                score_b=envelope["agentB"]["score"],
                combos_a=combo if pointWinner == ENVELOPE_AGENT_IDENTITY else 0,
                combos_b=combo if pointWinner == ENVELOPE_OPPONENT_IDENTITY else 0,
                duration_steps=envelope["step"],
                result={
                    "winner": pointWinner,
                    "reversedSides": reversedSides,
                    "round": roundIndex,
                    "mode": TRAINING_SELF_PLAY,
                    "status": envelope["status"],
                },
            )

    def _save_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tempDir:
            for agent in self.agents:
                temporaryPath = Path(tempDir) / f"agent-{agent.index}.pt"
                agent.dqn.save(temporaryPath)
                relativePath = (
                    f"run-{self.runUuid}/generation-{self.generationIndex}/agent-{agent.index}.pt"
                )
                relative, sha256 = self.store.checkpoints.write_bytes(
                    relativePath, temporaryPath.read_bytes()
                )
                self.store.record_checkpoint(
                    agent.storeId,
                    relative,
                    sha256,
                    MODEL_SPEC_VERSION,
                    CHECKPOINT_TYPE_FULL,
                )

    def _agent_summaries(self) -> list[dict]:
        return [
            {
                "agentId": agent.index,
                "storeId": agent.storeId,
                "generation": agent.generation,
                "architecture": self.genomes[agent.index].to_json(),
                "fitness": None,
                "role": agent.role,
                "weightInheritance": agent.inheritance,
                "status": "available",
                "games": agent.wins + agent.losses,
                "wins": agent.wins,
                "losses": agent.losses,
                "hits": agent.hits,
                "totalSteps": agent.dqn.total_steps,
                "replaySize": len(agent.dqn.replay_buffer),
                "trainingLoss": agent.dqn.latest_loss,
                "epsilon": round(agent.dqn.epsilonValue, 6),
                "cumulativeReward": round(agent.cumulativeReward, 6),
            }
            for agent in self.agents
        ]

    def _emit_metrics(self) -> None:
        self._emit(
            {
                "type": "training_metrics",
                "source": EVENT_SOURCE_LIVE,
                "runId": self.runId,
                "generationId": self.generationId,
                "round": self._roundIndex,
                "agents": self._agent_summaries(),
            }
        )
