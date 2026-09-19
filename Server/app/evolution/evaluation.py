from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from ..ai.dqn_agent import DQNAgent, pong_state_to_vector
from .simulation import MatchAssignment, MatchConfig, MatchParticipant, MatchSession, TERMINAL
from .training_orchestrator import pong_state_from_envelope


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
    benchmarkVersion: str = "spec-07-fixed-tracker-v1"

    def validate(self) -> None:
        if not self.seeds:
            raise ValueError("evaluation requires at least one seed")
        if self.maxStepsPerMatch <= 0 or self.winScore <= 0:
            raise ValueError("evaluation step and score budgets must be positive")


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


class FixedOpponentEvaluator:
    """Deterministic, inference-only benchmark with equal seeds and both sides."""

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

    @property
    def sampleCount(self) -> int:
        return len(self.configuration.seeds) * 2

    def evaluate(self, agents: Sequence[DQNAgent], generationIndex: int = 0) -> list[dict]:
        return [self._evaluate_agent(agent, index, generationIndex) for index, agent in enumerate(agents)]

    def _evaluate_agent(self, agent: DQNAgent, index: int, generationIndex: int) -> dict:
        counters = {
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "pointsFor": 0,
            "pointsAgainst": 0,
            "comboTotal": 0.0,
            "stepCapTerminations": 0,
        }
        trainingWasEnabled = agent.trainingEnabled
        replaySize = len(agent.replay_buffer)
        totalSteps = agent.total_steps
        trainingSteps = agent.training_steps
        agent.stop_training()
        try:
            for seed in self.configuration.seeds:
                for reversedSides in (False, True):
                    self._play_match(agent, index, generationIndex, seed, reversedSides, counters)
        finally:
            if trainingWasEnabled:
                agent.start_training()
        if (len(agent.replay_buffer), agent.total_steps, agent.training_steps) != (
            replaySize,
            totalSteps,
            trainingSteps,
        ):
            raise RuntimeError("evaluation mutated agent training state")
        metrics = calculate_fitness(
            counters["wins"],
            counters["draws"],
            counters["losses"],
            counters["pointsFor"],
            counters["pointsAgainst"],
            counters["comboTotal"],
            self.fitnessConfiguration,
        )
        return {
            **counters,
            **metrics,
            "sampleCount": self.sampleCount,
            "trainingStateUnchanged": True,
        }

    def _play_match(self, agent, index, generationIndex, seed, reversedSides, counters) -> None:
        assignment = MatchAssignment(
            arenaId="evaluation",
            runId="evaluation",
            generationId=f"generation-{generationIndex}",
            agentA=MatchParticipant(f"agent-{index}", generationIndex, 0.0),
            agentB=MatchParticipant("fixed-tracker", generationIndex, 0.0),
            reversedSides=reversedSides,
            seed=seed,
        )
        session = MatchSession(assignment, self.matchConfig)
        envelope = session.snapshot()
        bestCombo = 0
        for _ in range(self.configuration.maxStepsPerMatch):
            stateA = pong_state_to_vector(pong_state_from_envelope(envelope, "A", self.matchConfig))
            actionA = DQNAgent.action_name(agent.select_action(stateA, explore=False))
            actionB = _tracking_action(envelope, "agentB")
            envelope = session.step(actionA, actionB)
            if envelope["pointWinner"] == "agent":
                bestCombo = max(bestCombo, envelope["agentA"]["returns"])
            if envelope["status"] == TERMINAL:
                break
        scoreA, scoreB = envelope["agentA"]["score"], envelope["agentB"]["score"]
        counters["pointsFor"] += scoreA
        counters["pointsAgainst"] += scoreB
        counters["comboTotal"] += min(bestCombo, self.fitnessConfiguration.comboCap)
        if scoreA > scoreB:
            counters["wins"] += 1
        elif scoreA < scoreB:
            counters["losses"] += 1
        else:
            counters["draws"] += 1
        if (
            envelope["status"] != TERMINAL
            or (max(scoreA, scoreB) < self.configuration.winScore and envelope["step"] >= self.configuration.maxStepsPerMatch)
        ):
            counters["stepCapTerminations"] += 1


def _tracking_action(envelope: dict, participant: str) -> str:
    paddle = envelope[participant]["paddleY"]
    ball = envelope["ball"]["y"]
    tolerance = 0.02
    if ball < paddle - tolerance:
        return "UP"
    if ball > paddle + tolerance:
        return "DOWN"
    return "STAY"


def _bound(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
