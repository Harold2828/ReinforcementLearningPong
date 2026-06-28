from __future__ import annotations

from dataclasses import dataclass
import random

from .multi_agent_training_service import MultiAgentGameState, TRAINING_SELF_PLAY


@dataclass(frozen=True)
class PongTrainingEnvConfig:
    width: float = 800.0
    height: float = 600.0
    agentPaddleX: float = 700.0
    opponentPaddleX: float = 100.0
    paddleHeight: float = 96.0
    paddleSpeed: float = 18.0
    ballSpeedX: float = 12.0
    ballSpeedY: float = 7.0
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
        self.reset()

    def reset(self) -> MultiAgentGameState:
        self.episodeId += 1
        self.comboSmash = 0
        self.stepCount = 0
        self.previousAgentDistanceToBall = None
        self.previousOpponentDistanceToBall = None
        self.agentPaddleY = self.config.height / 2
        self.opponentPaddleY = self.config.height / 2
        self._serve_ball()
        self.lastHitBy = None
        self.pointWinner = None
        return self.state()

    def step(self, agentAction: str, opponentAction: str) -> PongTrainingSnapshot:
        self.lastHitBy = None
        self.pointWinner = None
        self.previousAgentDistanceToBall = abs(self.ballY - self.agentPaddleY)
        self.previousOpponentDistanceToBall = abs(self.ballY - self.opponentPaddleY)

        self.agentPaddleY = self._move_paddle(self.agentPaddleY, agentAction)
        self.opponentPaddleY = self._move_paddle(self.opponentPaddleY, opponentAction)
        self.ballX += self.ballVelocityX
        self.ballY += self.ballVelocityY
        self.stepCount += 1

        self._bounce_vertical_walls()
        self._handle_paddle_collisions()
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

    def _serve_ball(self) -> None:
        self.ballX = self.config.width / 2
        self.ballY = self.config.height / 2
        horizontalDirection = self.randomGenerator.choice((-1.0, 1.0))
        verticalDirection = self.randomGenerator.choice((-1.0, 1.0))
        self.ballVelocityX = self.config.ballSpeedX * horizontalDirection
        self.ballVelocityY = self.config.ballSpeedY * verticalDirection

    def _move_paddle(self, paddleY: float, action: str) -> float:
        if action == "UP":
            paddleY -= self.config.paddleSpeed
        elif action == "DOWN":
            paddleY += self.config.paddleSpeed
        return min(self.config.height, max(0.0, paddleY))

    def _bounce_vertical_walls(self) -> None:
        if self.ballY <= 0:
            self.ballY = 0
            self.ballVelocityY = abs(self.ballVelocityY)
        elif self.ballY >= self.config.height:
            self.ballY = self.config.height
            self.ballVelocityY = -abs(self.ballVelocityY)

    def _handle_paddle_collisions(self) -> None:
        if self.ballVelocityX > 0 and self.ballX >= self.config.agentPaddleX:
            if self._is_ball_inside_paddle(self.agentPaddleY):
                self.ballX = self.config.agentPaddleX
                self.ballVelocityX = -abs(self.ballVelocityX)
                self.ballVelocityY += (self.ballY - self.agentPaddleY) * 0.08
                self.comboSmash += 1
                self.lastHitBy = "agent"
        elif self.ballVelocityX < 0 and self.ballX <= self.config.opponentPaddleX:
            if self._is_ball_inside_paddle(self.opponentPaddleY):
                self.ballX = self.config.opponentPaddleX
                self.ballVelocityX = abs(self.ballVelocityX)
                self.ballVelocityY += (self.ballY - self.opponentPaddleY) * 0.08
                self.comboSmash += 1
                self.lastHitBy = "opponent"

    def _handle_point(self) -> None:
        if self.ballX > self.config.width:
            self.opponentScore += 1
            self.pointWinner = "opponent"
        elif self.ballX < 0:
            self.agentScore += 1
            self.pointWinner = "agent"

    def _finish_forced_point(self) -> None:
        if self.ballX >= self.config.width / 2:
            self.opponentScore += 1
            self.pointWinner = "opponent"
        else:
            self.agentScore += 1
            self.pointWinner = "agent"

    def _is_ball_inside_paddle(self, paddleY: float) -> bool:
        halfPaddleHeight = self.config.paddleHeight / 2
        return paddleY - halfPaddleHeight <= self.ballY <= paddleY + halfPaddleHeight
