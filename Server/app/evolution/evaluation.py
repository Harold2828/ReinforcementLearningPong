from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Sequence

import torch

from ..ai.dqn_agent import DQNAgent, pong_state_to_vector
from .simulation import MatchAssignment, MatchConfig, MatchParticipant, MatchSession, TERMINAL
from .training_orchestrator import pong_state_from_envelope, round_robin_pairs


@dataclass(frozen=True)
class FitnessConfiguration:
    winRateWeight: float = 0.65
    pointDifferentialWeight: float = 0.25
    comboPerformanceWeight: float = 0.10
    comboCap: int = 10

    def validate(self) -> None:
        weights = (self.winRateWeight, self.pointDifferentialWeight, self.comboPerformanceWeight)
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("fitness weights must be finite and non-negative")
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ValueError("fitness weights must sum to one")
        if self.comboCap <= 0:
            raise ValueError("comboCap must be greater than zero")


@dataclass(frozen=True)
class EvaluationConfiguration:
    seeds: tuple[int, ...] = (101, 211, 307)
    maxStepsPerMatch: int = 2_000
    winScore: int = 7
    maxConcurrentMatches: int = 6
    benchmarkVersion: str = "spec-07-round-robin-v1"

    def validate(self) -> None:
        if not self.seeds:
            raise ValueError("evaluation requires at least one seed")
        if self.maxStepsPerMatch <= 0 or self.winScore <= 0:
            raise ValueError("evaluation step and score budgets must be positive")
        if self.maxConcurrentMatches <= 0:
            raise ValueError("evaluation concurrency must be positive")


def calculate_fitness(
    wins: int,
    draws: int,
    losses: int,
    pointsFor: int,
    pointsAgainst: int,
    comboTotal: float,
    configuration: FitnessConfiguration | None = None,
) -> dict[str, float]:
    config = configuration or FitnessConfiguration()
    config.validate()
    matches = wins + draws + losses
    if matches <= 0:
        metrics = {"winRate": 0.0, "pointDifferential": 0.0, "comboPerformance": 0.0}
    else:
        winRate = (wins + 0.5 * draws) / matches
        pointScale = max(1, pointsFor + pointsAgainst)
        pointDifferential = 0.5 + 0.5 * (pointsFor - pointsAgainst) / pointScale
        metrics = {
            "winRate": _bound(winRate),
            "pointDifferential": _bound(pointDifferential),
            "comboPerformance": _bound(comboTotal / (matches * config.comboCap)),
        }
    fitness = (
        config.winRateWeight * metrics["winRate"]
        + config.pointDifferentialWeight * metrics["pointDifferential"]
        + config.comboPerformanceWeight * metrics["comboPerformance"]
    )
    return {**metrics, "fitness": _bound(fitness)}


class RoundRobinEvaluator:
    """Deterministic, inference-only tournament with both paddle sides."""

    def __init__(
        self,
        evaluation: EvaluationConfiguration | None = None,
        fitness: FitnessConfiguration | None = None,
        matchConfig: MatchConfig | None = None,
    ) -> None:
        self.configuration = evaluation or EvaluationConfiguration()
        self.fitnessConfiguration = fitness or FitnessConfiguration()
        self.configuration.validate()
        self.fitnessConfiguration.validate()
        base = matchConfig or MatchConfig()
        self.matchConfig = MatchConfig(
            width=base.width,
            height=base.height,
            agentPaddleX=base.agentPaddleX,
            opponentPaddleX=base.opponentPaddleX,
            paddleWidth=base.paddleWidth,
            paddleHeight=base.paddleHeight,
            ballWidth=base.ballWidth,
            ballHeight=base.ballHeight,
            paddleSpeed=base.paddleSpeed,
            ballSpeedX=base.ballSpeedX,
            ballSpeedY=base.ballSpeedY,
            stepSeconds=base.stepSeconds,
            maxStepsPerRally=min(base.maxStepsPerRally, self.configuration.maxStepsPerMatch),
            maxTotalSteps=self.configuration.maxStepsPerMatch,
            winScore=self.configuration.winScore,
        )
        self.matchResults: list[dict] = []
        self.maxConcurrentObserved = 0

    def evaluate(self, agents: Sequence[DQNAgent], generationIndex: int = 0) -> list[dict]:
        if len(agents) < 2 or len(agents) % 2:
            raise ValueError("round-robin evaluation requires an even population of at least two")
        counters = [_empty_counters() for _ in agents]
        trainingStates = [agent.trainingEnabled for agent in agents]
        trainingCounters = [
            (len(agent.replay_buffer), agent.total_steps, agent.training_steps)
            for agent in agents
        ]
        networkStates = [_network_state(agent) for agent in agents]
        self.matchResults = []
        self.maxConcurrentObserved = 0
        for agent in agents:
            agent.stop_training()
        try:
            for roundIndex in range(len(agents) - 1):
                pairs = [(a, b) for a, b, _ in round_robin_pairs(len(agents), roundIndex)]
                seed = self.configuration.seeds[roundIndex % len(self.configuration.seeds)]
                for reversedSides in (False, True):
                    for start in range(0, len(pairs), self.configuration.maxConcurrentMatches):
                        batch = pairs[start:start + self.configuration.maxConcurrentMatches]
                        self.maxConcurrentObserved = max(self.maxConcurrentObserved, len(batch))
                        self._play_batch(
                            agents, batch, generationIndex, roundIndex, seed, reversedSides, counters
                        )
        finally:
            for agent, enabled in zip(agents, trainingStates):
                if enabled:
                    agent.start_training()
        for index, agent in enumerate(agents):
            current = (len(agent.replay_buffer), agent.total_steps, agent.training_steps)
            if current != trainingCounters[index] or not _same_network_state(agent, networkStates[index]):
                raise RuntimeError("evaluation mutated agent training state or model weights")
        results = []
        for item in counters:
            metrics = calculate_fitness(
                item["wins"], item["draws"], item["losses"],
                item["pointsFor"], item["pointsAgainst"], item["comboTotal"],
                self.fitnessConfiguration,
            )
            results.append({
                **item,
                **metrics,
                "sampleCount": item["wins"] + item["draws"] + item["losses"],
                "trainingStateUnchanged": True,
                "weightsUnchanged": True,
            })
        return results

    def _play_batch(
        self, agents, pairs, generationIndex, roundIndex, seed, reversedSides, counters
    ) -> None:
        sessions = []
        envelopes = []
        combos = []
        for batchIndex, (aIndex, bIndex) in enumerate(pairs):
            assignment = MatchAssignment(
                arenaId=f"evaluation-{batchIndex}",
                runId="evaluation",
                generationId=f"generation-{generationIndex}",
                agentA=MatchParticipant(f"agent-{aIndex}", generationIndex, 0.0),
                agentB=MatchParticipant(f"agent-{bIndex}", generationIndex, 0.0),
                reversedSides=reversedSides,
                seed=seed,
                matchKey=f"round-{roundIndex}-side-{int(reversedSides)}",
            )
            session = MatchSession(assignment, self.matchConfig)
            sessions.append((session, aIndex, bIndex))
            envelopes.append(session.snapshot())
            combos.append([0, 0])

        for _ in range(self.configuration.maxStepsPerMatch):
            active = False
            for index, (session, aIndex, bIndex) in enumerate(sessions):
                envelope = envelopes[index]
                if envelope["status"] == TERMINAL:
                    continue
                active = True
                stateA = pong_state_to_vector(
                    pong_state_from_envelope(envelope, "A", self.matchConfig)
                )
                stateB = pong_state_to_vector(
                    pong_state_from_envelope(envelope, "B", self.matchConfig)
                )
                actionA = DQNAgent.action_name(agents[aIndex].select_action(stateA, explore=False))
                actionB = DQNAgent.action_name(agents[bIndex].select_action(stateB, explore=False))
                envelope = session.step(actionA, actionB)
                envelopes[index] = envelope
                if envelope["pointWinner"] == "agent":
                    combos[index][0] = max(combos[index][0], envelope["agentA"]["returns"])
                elif envelope["pointWinner"] == "opponent":
                    combos[index][1] = max(combos[index][1], envelope["agentB"]["returns"])
            if not active:
                break

        for (_, aIndex, bIndex), envelope, bestCombos in zip(sessions, envelopes, combos):
            self._record_match(
                envelope, aIndex, bIndex, generationIndex, roundIndex,
                seed, reversedSides, bestCombos, counters,
            )

    def _record_match(
        self, envelope, aIndex, bIndex, generationIndex, roundIndex,
        seed, reversedSides, bestCombos, counters,
    ) -> None:
        scoreA = envelope["agentA"]["score"]
        scoreB = envelope["agentB"]["score"]
        stepCap = envelope["status"] != TERMINAL or max(scoreA, scoreB) < self.configuration.winScore
        for own, ownScore, otherScore, combo in (
            (aIndex, scoreA, scoreB, bestCombos[0]),
            (bIndex, scoreB, scoreA, bestCombos[1]),
        ):
            item = counters[own]
            item["pointsFor"] += ownScore
            item["pointsAgainst"] += otherScore
            item["comboTotal"] += min(combo, self.fitnessConfiguration.comboCap)
            if ownScore > otherScore:
                item["wins"] += 1
            elif ownScore < otherScore:
                item["losses"] += 1
            else:
                item["draws"] += 1
            if stepCap:
                item["stepCapTerminations"] += 1
        self.matchResults.append({
            "generation": generationIndex,
            "round": roundIndex,
            "agentAIndex": aIndex,
            "agentBIndex": bIndex,
            "reversedSides": reversedSides,
            "seed": seed,
            "scoreA": scoreA,
            "scoreB": scoreB,
            "bestComboA": min(bestCombos[0], self.fitnessConfiguration.comboCap),
            "bestComboB": min(bestCombos[1], self.fitnessConfiguration.comboCap),
            "steps": envelope["step"],
            "status": envelope["status"],
            "stepCapTermination": stepCap,
        })


def _empty_counters() -> dict:
    return {
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "pointsFor": 0,
        "pointsAgainst": 0,
        "comboTotal": 0.0,
        "stepCapTerminations": 0,
    }


def _network_state(agent: DQNAgent) -> tuple[dict, dict]:
    return (
        copy.deepcopy(agent.policy_net.state_dict()),
        copy.deepcopy(agent.target_net.state_dict()),
    )


def _same_network_state(agent: DQNAgent, before: tuple[dict, dict]) -> bool:
    return all(
        torch.equal(current[key], saved[key])
        for current, saved in zip(
            (agent.policy_net.state_dict(), agent.target_net.state_dict()), before
        )
        for key in saved
    )


def _bound(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
