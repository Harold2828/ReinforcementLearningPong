"""SPEC-05 game simulation engine.

Authoritative headless physics (Phaser is a viewer). Each match owns a
PongTrainingEnv and its own random.Random so no state or RNG is shared
between arenas. Snapshots conform to the SPEC-03 frontend contract.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, replace

from ..ai.multi_agent_training_service import (
    MultiAgentGameState,
    TRAINING_SELF_PLAY,
    calculate_adversarial_rewards,
)
from ..ai.pong_training_env import PongTrainingEnv, PongTrainingEnvConfig

RUNNING = "running"
TERMINAL = "terminal"

ARENA_IDS = tuple(f"arena-{index}" for index in range(5))
VELOCITY_SCALE = 100.0


@dataclass(frozen=True)
class MatchConfig:
    width: float = 800.0
    height: float = 600.0
    agentPaddleX: float = 700.0
    opponentPaddleX: float = 100.0
    paddleHeight: float = 96.0
    paddleSpeed: float = 18.0
    ballSpeedX: float = 12.0
    ballSpeedY: float = 7.0
    maxStepsPerRally: int = 1000
    maxTotalSteps: int = 5000
    winScore: int = 7


@dataclass(frozen=True)
class MatchParticipant:
    agentId: str
    generation: int = 0
    epsilon: float = 0.0


@dataclass(frozen=True)
class MatchAssignment:
    """One arena pairing. reverseSides puts agentA on the left paddle."""

    arenaId: str
    runId: str
    generationId: str
    agentA: MatchParticipant
    agentB: MatchParticipant
    reversedSides: bool = False
    seed: int = 0


def _mirror_role(role: str | None) -> str | None:
    if role == "agent":
        return "opponent"
    if role == "opponent":
        return "agent"
    return None


class MatchSession:
    """One isolated match (one rally at a time) for a single arena.

    Reward ownership: default orientation right paddle = agent, left =
    opponent. On reversedSides the state is mirrored back to the agent
    identity so rewards always belong to agentA.
    """

    def __init__(self, assignment: MatchAssignment, config: MatchConfig):
        self.assignment = assignment
        self.config = config
        self.seed = assignment.seed
        self.sessionId = 0
        self.sequence = 0
        self.sessionSteps = 0
        self.status = RUNNING
        self.pointWinner: str | None = None
        self.combo = 0
        self.rallyReturnsA = 0
        self.rallyReturnsB = 0
        self.lastRewards = (0.0, 0.0)
        self.reset()

    def _new_env(self) -> PongTrainingEnv:
        return PongTrainingEnv(
            config=PongTrainingEnvConfig(
                width=self.config.width,
                height=self.config.height,
                agentPaddleX=self.config.agentPaddleX,
                opponentPaddleX=self.config.opponentPaddleX,
                paddleHeight=self.config.paddleHeight,
                paddleSpeed=self.config.paddleSpeed,
                ballSpeedX=self.config.ballSpeedX,
                ballSpeedY=self.config.ballSpeedY,
                maxStepsPerEpisode=self.config.maxStepsPerRally,
            ),
            randomGenerator=random.Random(self.seed),
        )

    def reset(self) -> dict:
        """Start a brand new match session: fresh RNG, zeroed scores."""
        self.env = self._new_env()
        self.sessionId += 1
        self.sequence = 1
        self.sessionSteps = 0
        self.status = RUNNING
        self.pointWinner = None
        self.combo = 0
        self.rallyReturnsA = 0
        self.rallyReturnsB = 0
        self.lastRewards = (0.0, 0.0)
        return self.snapshot()

    def step(self, agentAction: str, opponentAction: str) -> dict:
        """Advance one tick. Terminal matches are frozen: no bootstrap."""
        if self.status == TERMINAL:
            return self.snapshot()

        if self.assignment.reversedSides:
            _actionA, _actionB = opponentAction, agentAction
        else:
            _actionA, _actionB = agentAction, opponentAction

        result = self.env.step(_actionA, _actionB)
        self.sessionSteps += 1
        self.sequence += 1
        self._track_returns(result.state)

        if not result.done:
            self.pointWinner = None
            self.lastRewards = calculate_adversarial_rewards(self._identity_state())
            return self.snapshot()

        self.pointWinner = self._to_identity_role(result.state.pointWinner)
        self.lastRewards = calculate_adversarial_rewards(self._identity_state())

        if self._match_ended():
            self.combo = self._rally_streak(self.pointWinner)
            self.status = TERMINAL
            return self.snapshot()

        self.combo = 0
        backToServe = self.snapshot()
        self.env.reset()
        self.rallyReturnsA = 0
        self.rallyReturnsB = 0
        return backToServe

    def set_epsilon(self, identityRole: str, value: float) -> None:
        agentA = self.assignment.agentA
        agentB = self.assignment.agentB
        if identityRole == "agent":
            agentA = MatchParticipant(agentA.agentId, agentA.generation, value)
        else:
            agentB = MatchParticipant(agentB.agentId, agentB.generation, value)
        self.assignment = replace(self.assignment, agentA=agentA, agentB=agentB)

    def _match_ended(self) -> bool:
        scores = self._identity_scores()
        return max(scores) >= self.config.winScore or self.sessionSteps >= self.config.maxTotalSteps

    def _to_identity_role(self, role: str | None) -> str | None:
        return _mirror_role(role) if self.assignment.reversedSides else role

    def _track_returns(self, state: MultiAgentGameState) -> None:
        if state.lastHitBy is None:
            return
        identity = self._to_identity_role(state.lastHitBy)
        if identity == "agent":
            self.rallyReturnsA += 1
        else:
            self.rallyReturnsB += 1

    def _rally_streak(self, identityRole: str | None) -> int:
        if identityRole is None:
            return 0
        return self.rallyReturnsA if identityRole == "agent" else self.rallyReturnsB

    def _identity_scores(self) -> tuple[int, int]:
        if self.assignment.reversedSides:
            return self.env.opponentScore, self.env.agentScore
        return self.env.agentScore, self.env.opponentScore

    def _identity_state(self) -> MultiAgentGameState:
        state = self.env.state()
        if not self.assignment.reversedSides:
            return state
        return MultiAgentGameState(
            gameMode=TRAINING_SELF_PLAY,
            ballX=self.config.width - state.ballX,
            ballY=state.ballY,
            ballVelocityX=-state.ballVelocityX,
            ballVelocityY=state.ballVelocityY,
            agentPaddleY=state.opponentPaddleY,
            opponentPaddleY=state.agentPaddleY,
            agentScore=state.opponentScore,
            opponentScore=state.agentScore,
            width=state.width,
            height=state.height,
            lastHitBy=_mirror_role(state.lastHitBy),
            pointWinner=_mirror_role(state.pointWinner),
            episodeId=state.episodeId,
            previousAgentDistanceToBall=state.previousOpponentDistanceToBall,
            previousOpponentDistanceToBall=state.previousAgentDistanceToBall,
            comboSmash=state.comboSmash,
        )

    def snapshot(self) -> dict:
        """SPEC-03 contract envelope; coordinates normalized to 0..1."""
        state = self._identity_state()
        width, height = self.config.width, self.config.height
        agentAScore, agentBScore = self._identity_scores()
        agentAPaddleX = (
            self.config.opponentPaddleX / width
            if self.assignment.reversedSides
            else self.config.agentPaddleX / width
        )
        agentBPaddleX = (
            self.config.agentPaddleX / width
            if self.assignment.reversedSides
            else self.config.opponentPaddleX / width
        )
        return {
            "type": "match_snapshot",
            "runId": self.assignment.runId,
            "generationId": self.assignment.generationId,
            "matchId": f"match-{self.assignment.arenaId}-session-{self.sessionId}",
            "arenaId": self.assignment.arenaId,
            "sequence": max(self.sequence, 1),
            "stateTimestamp": time.time() * 1000,
            "step": self.sessionSteps,
            "elapsedSteps": self.sessionSteps,
            "status": self.status,
            "agentA": {
                "id": self.assignment.agentA.agentId,
                "generation": self.assignment.agentA.generation,
                "paddleX": agentAPaddleX,
                "paddleY": state.agentPaddleY / height,
                "epsilon": self.assignment.agentA.epsilon,
                "score": agentAScore,
                "returns": self.rallyReturnsA,
            },
            "agentB": {
                "id": self.assignment.agentB.agentId,
                "generation": self.assignment.agentB.generation,
                "paddleX": agentBPaddleX,
                "paddleY": state.opponentPaddleY / height,
                "epsilon": self.assignment.agentB.epsilon,
                "score": agentBScore,
                "returns": self.rallyReturnsB,
            },
            "ball": {
                "x": state.ballX / width,
                "y": state.ballY / height,
                "vx": state.ballVelocityX / VELOCITY_SCALE,
                "vy": state.ballVelocityY / VELOCITY_SCALE,
            },
            "pointWinner": self.pointWinner,
            "combo": self.combo,
        }


class SimulationEngine:
    """Five arenas, five isolated matches. No shared mutable state."""

    def __init__(self, assignments: list[MatchAssignment], config: MatchConfig | None = None):
        self.config = config or MatchConfig()
        self.assignments = list(assignments)
        if len(self.assignments) != len(ARENA_IDS):
            raise ValueError(f"SimulationEngine expects {len(ARENA_IDS)} matches")
        self.sessions = [MatchSession(assignment, self.config) for assignment in self.assignments]
        self.byArena = {session.assignment.arenaId: session for session in self.sessions}
        self.elapsedSteps = 0

    def step(self, actions: dict[str, tuple[str, str]] | None = None) -> list[dict]:
        self.elapsedSteps += 1
        envelopes = []
        for session in self.sessions:
            acts = actions.get(session.assignment.arenaId) if actions else None
            if acts is not None and session.status != TERMINAL:
                envelopes.append(session.step(acts[0], acts[1]))
            else:
                envelopes.append(session.snapshot())
        return envelopes

    def reset_all(self) -> list[dict]:
        self.elapsedSteps = 0
        return [session.reset() for session in self.sessions]

    @property
    def arena_ids(self) -> tuple[str, ...]:
        return tuple(session.assignment.arenaId for session in self.sessions)