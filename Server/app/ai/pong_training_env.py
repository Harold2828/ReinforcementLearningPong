from __future__ import annotations

from dataclasses import dataclass
import math
import random

from .multi_agent_training_service import MultiAgentGameState, TRAINING_SELF_PLAY


@dataclass(frozen=True)
class PongTrainingEnvConfig:
    width: float = 800.0
    height: float = 600.0
    agentPaddleX: float = 700.0
    opponentPaddleX: float = 100.0
    paddleWidth: float = 50.0
    paddleHeight: float = 88.0
    ballWidth: float = 48.1
    ballHeight: float = 51.9
    paddleSpeed: float = 500.0
    ballSpeedX: float = 200.0
    ballSpeedY: float = 200.0
    stepSeconds: float = 1.0 / 60.0
    serveSpeedXMean: float = 175.0
    serveSpeedXDeviation: float = 40.0
    serveSpeedYMean: float = 170.0
    serveSpeedYDeviation: float = 25.0
    bounceSpeedXMean: float = 1.5
    bounceSpeedXDeviation: float = 0.4
    bounceAngleMean: float = 3.5
    bounceAngleDeviation: float = 1.1
    maxStepsPerEpisode: int = 1000


@dataclass
class PongTrainingSnapshot:
    state: MultiAgentGameState
    done: bool


class PongTrainingEnv:
    """
    Lightweight gym-like Pong simulator used for fast pretraining without Phaser rendering.
    """

    def __init__(
        self,
        config: PongTrainingEnvConfig | None = None,
        randomGenerator: random.Random | None = None,
    ):
        self.config = config or PongTrainingEnvConfig()
        self.randomGenerator = randomGenerator or random.Random()
        self.episodeId = 0
        self.agentScore = 0
        self.opponentScore = 0
        self.comboSmash = 0
        self.stepCount = 0
        self.previousAgentDistanceToBall: float | None = None
        self.previousOpponentDistanceToBall: float | None = None
        self.ballX = 0.0
        self.ballY = 0.0
        self.ballVelocityX = 0.0
        self.ballVelocityY = 0.0
        self.agentPaddleY = 0.0
        self.opponentPaddleY = 0.0
        self.lastHitBy: str | None = None
        self.pointWinner: str | None = None
        self._reset_positions()
        self._serve_initial_ball()

    def reset(
        self, serveDirection: float = -1.0, resetPaddles: bool = True
    ) -> MultiAgentGameState:
        self.episodeId += 1
        self.comboSmash = 0
        self.stepCount = 0
        self.previousAgentDistanceToBall = None
        self.previousOpponentDistanceToBall = None
        if resetPaddles:
            self._reset_positions()
        self._serve_ball(serveDirection)
        self.lastHitBy = None
        self.pointWinner = None
        return self.state()

    def _reset_positions(self) -> None:
        self.agentPaddleY = self.config.height / 2
        self.opponentPaddleY = self.config.height / 2

    def step(self, agentAction: str, opponentAction: str) -> PongTrainingSnapshot:
        self.lastHitBy = None
        self.pointWinner = None
        self.previousAgentDistanceToBall = abs(self.ballY - self.agentPaddleY)
        self.previousOpponentDistanceToBall = abs(self.ballY - self.opponentPaddleY)

        self.agentPaddleY = self._move_paddle(self.agentPaddleY, agentAction)
        self.opponentPaddleY = self._move_paddle(self.opponentPaddleY, opponentAction)
        previousBallX = self.ballX
        previousBallY = self.ballY
        self.ballX += self.ballVelocityX * self.config.stepSeconds
        self.ballY += self.ballVelocityY * self.config.stepSeconds
        self.stepCount += 1

        self._bounce_vertical_walls()
        self._handle_paddle_collisions(previousBallX, previousBallY)
        self._handle_point()

        if self.stepCount >= self.config.maxStepsPerEpisode and self.pointWinner is None:
            self._finish_forced_point()
        return PongTrainingSnapshot(state=self.state(), done=self.pointWinner is not None)

    def state(self) -> MultiAgentGameState:
        return MultiAgentGameState(
            gameMode=TRAINING_SELF_PLAY,
            ballX=self.ballX,
            ballY=self.ballY,
            ballVelocityX=self.ballVelocityX,
            ballVelocityY=self.ballVelocityY,
            agentPaddleY=self.agentPaddleY,
            opponentPaddleY=self.opponentPaddleY,
            agentScore=self.agentScore,
            opponentScore=self.opponentScore,
            width=self.config.width,
            height=self.config.height,
            lastHitBy=self.lastHitBy,
            pointWinner=self.pointWinner,
            episodeId=self.episodeId,
            previousAgentDistanceToBall=self.previousAgentDistanceToBall,
            previousOpponentDistanceToBall=self.previousOpponentDistanceToBall,
            comboSmash=self.comboSmash,
        )

    def _serve_initial_ball(self) -> None:
        self.ballX = self.config.width / 2
        self.ballY = self.config.height / 2
        self.ballVelocityX = self.config.ballSpeedX
        self.ballVelocityY = self.config.ballSpeedY

    def _serve_ball(self, direction: float) -> None:
        self.ballX = self.config.width / 2
        self.ballY = self.config.height / 2
        direction = 1.0 if direction >= 0 else -1.0
        self.ballVelocityX = self._random_normal(
            self.config.serveSpeedXMean, self.config.serveSpeedXDeviation
        ) * direction
        self.ballVelocityY = self._random_normal(
            self.config.serveSpeedYMean, self.config.serveSpeedYDeviation
        ) * direction

    def _move_paddle(self, paddleY: float, action: str) -> float:
        if action == "UP":
            paddleY -= self.config.paddleSpeed * self.config.stepSeconds
        elif action == "DOWN":
            paddleY += self.config.paddleSpeed * self.config.stepSeconds
        halfHeight = self.config.paddleHeight / 2
        return min(self.config.height - halfHeight, max(halfHeight, paddleY))

    def _bounce_vertical_walls(self) -> None:
        halfHeight = self.config.ballHeight / 2
        if self.ballY < halfHeight:
            self.ballY = halfHeight
            self.ballVelocityY *= -1
        elif self.ballY > self.config.height - halfHeight:
            self.ballY = self.config.height - halfHeight
            self.ballVelocityY *= -1

    def _handle_paddle_collisions(self, previousBallX: float, previousBallY: float) -> None:
        """Swept AABB collision: reflect at the exact plane-crossing moment.

        The ball Y is sampled where its face crosses the paddle plane, so
        fast edge grazes and high-speed balls cannot pass through.
        """
        halfSpan = (self.config.paddleWidth + self.config.ballWidth) / 2
        rightContact = self.config.agentPaddleX - halfSpan
        leftContact = self.config.opponentPaddleX + halfSpan
        if self.ballVelocityX > 0 and previousBallX <= rightContact < self.ballX:
            contactY = self._contact_y(previousBallX, previousBallY, rightContact)
            if self._is_ball_inside_paddle(self.agentPaddleY, contactY):
                self.ballX = rightContact
                self.ballVelocityX *= -1
                self._apply_original_bounce(self.agentPaddleY, contactY)
                self.comboSmash += 1
                self.lastHitBy = "agent"
        elif self.ballVelocityX < 0 and self.ballX <= leftContact < previousBallX:
            contactY = self._contact_y(previousBallX, previousBallY, leftContact)
            if self._is_ball_inside_paddle(self.opponentPaddleY, contactY):
                self.ballX = leftContact
                self.ballVelocityX *= -1
                self._apply_original_bounce(self.opponentPaddleY, contactY)
                self.comboSmash += 1
                self.lastHitBy = "opponent"

    def _contact_y(self, previousBallX: float, previousBallY: float, contactX: float) -> float:
        """Ball Y at the instant its face crossed ``contactX`` this step."""
        stepX = self.ballX - previousBallX
        if abs(stepX) < 1e-9:
            return previousBallY
        progress = (contactX - previousBallX) / stepX
        return previousBallY + (self.ballY - previousBallY) * progress

    def _apply_original_bounce(self, paddleY: float, contactY: float) -> None:
        difference = contactY - paddleY
        self.ballVelocityY = difference * self._random_normal(
            self.config.bounceAngleMean, self.config.bounceAngleDeviation
        )
        self.ballVelocityX *= self._random_normal(
            self.config.bounceSpeedXMean, self.config.bounceSpeedXDeviation
        )

    def _handle_point(self) -> None:
        halfWidth = self.config.ballWidth / 2
        if self.ballX > self.config.width - halfWidth:
            self.opponentScore += 1
            self.pointWinner = "opponent"
            self.comboSmash = 0
        elif self.ballX < halfWidth:
            self.agentScore += 1
            self.pointWinner = "agent"
            self.comboSmash = 0

    def _finish_forced_point(self) -> None:
        if self.ballX >= self.config.width / 2:
            self.opponentScore += 1
            self.pointWinner = "opponent"
        else:
            self.agentScore += 1
            self.pointWinner = "agent"

    def _is_ball_inside_paddle(self, paddleY: float, ballY: float | None = None) -> bool:
        halfHeight = (self.config.paddleHeight + self.config.ballHeight) / 2
        ballY = self.ballY if ballY is None else ballY
        return paddleY - halfHeight <= ballY <= paddleY + halfHeight

    def _random_normal(self, mean: float, standardDeviation: float) -> float:
        first = 0.0
        second = 0.0
        while first == 0.0:
            first = self.randomGenerator.random()
        while second == 0.0:
            second = self.randomGenerator.random()
        gaussian = math.sqrt(-2.0 * math.log(first)) * math.cos(2.0 * math.pi * second)
        return gaussian * standardDeviation + mean
