import Phaser from "phaser";
import SocketManager from "../utils/socketManager";
import { randomNormal } from "../utils/customRandom";
import { DashboardController } from "../utils/dashboardController";
import {
    ACTIONS,
    applyAgentActionToPaddle,
    buildQlearningState,
    normalizeAgentAction,
    normalizeOpponentAction,
} from "../utils/gameState";
import { GameModeManager } from "../utils/gameModeManager";

class BasicGame extends Phaser.Scene {
    constructor() {
        super({ key: "BasicGame" });

        this.background = {
            court: { image: null },
            ball: {
                image: null,
                sound: null,
                initial: {
                    position: { x: 400, y: 300 },
                    velocity: { x: 200, y: 200 },
                },
            },
            text: {
                scores: {
                    comboSmash: null,
                    team1: null,
                    team2: null,
                },
            },
        };
        this.actors = {
            point: { sound: null },
            numberOfPlayers: 2,
            scores: { comboSmash: 0 },
            players: [
                { racket: null, score: 0, team: null },
                { racket: null, score: 0, team: null },
            ],
        };
        this.cursors = { keyboard: null };
        this.socketManager = null;
        this.dashboardController = null;
        this.gameModeManager = new GameModeManager();
        this.trainingActive = true;
        this.pendingAgentAction = ACTIONS.STAY;
        this.pendingOpponentAction = ACTIONS.STAY;
        this.lastScoredBy = null;
        this.lastHitBy = null;
        this.agentMissedBall = false;
        this.previousAgentDistanceToBall = null;
        this.previousOpponentDistanceToBall = null;
        this.latestReward = 0;
        this.latestOpponentReward = 0;
        this.latestEpsilon = 0;
        this.latestOpponentEpsilon = 0;
        this.currentEpisode = 1;
    }

    preload() {
        this.load.image("court", "assets/background/court.png");
        this.load.image("ball", "assets/background/ball.png");
        this.load.atlas("rackets", "assets/player/rackets.png", "assets/player/rackets.json");
        this.load.audio("smash", "assets/background/smash.wav");
        this.load.audio("point", "assets/background/point.mp3");
    }

    create() {
        const { width, height } = this.sys.game.canvas;
        this.width = width;
        this.height = height;

        this.dashboardController = new DashboardController();
        this.dashboardController.bindControls({
            onStartTraining: () => this.startTraining(),
            onStopTraining: () => this.stopTraining(),
            onResetEpisode: () => this.resetEpisode(),
            onGameModeChange: (gameMode) => this.changeGameMode(gameMode),
        });
        this.dashboardController.setTrainingActive(true);
        this.dashboardController.setSelectedMode(this.gameModeManager.currentMode);

        this.configureSocket();
        this.configureCourt();
        this.configureBall();
        this.configureScores();
        this.configurePlayers();
        this.configureKeyboard();

        this.physics.world.on("worldbounds", this.handleGoal, this);
        this.background.ball.image.body.onWorldBounds = true;
    }

    update() {
        this.actors.players[0].racket.setVelocityY(0);
        this.actors.players[1].racket.setVelocityY(0);

        if (this.trainingActive) {
            if (this.gameModeManager.isAgentAiControlled()) {
                applyAgentActionToPaddle(this.actors.players[0].racket, this.pendingAgentAction);
            }
            if (this.gameModeManager.isOpponentAiControlled()) {
                applyAgentActionToPaddle(this.actors.players[1].racket, this.pendingOpponentAction);
            }
            this.handleHumanControlsByMode();
            this.sendTrainingState();
        } else {
            this.handleHumanControlsByMode();
        }

        this.updateScoreText();
        this.updateDashboard();
        this.resetTransientEvents();
    }

    handleHumanControlsByMode() {
        if (this.gameModeManager.isAgentHumanControlled()) {
            this.handleManualAgentControls();
        }
        if (this.gameModeManager.isOpponentHumanControlled()) {
            this.handleOpponentControls();
        }
    }

    configureSocket() {
        const backendUrl = import.meta.env.VITE_BACKEND_URL || "http://127.0.0.1:5001";
        this.socketManager = new SocketManager(backendUrl);

        this.socketManager.onConnectionChange((isConnected) => {
            this.dashboardController.update({
                connected: isConnected,
                mode: this.gameModeManager.getModeLabel(),
                feedback: isConnected ? "Backend connected" : "Backend disconnected",
            });
        });

        this.socketManager.onAiMove((payload) => {
            const selectedAction = normalizeAgentAction(payload);
            const selectedOpponentAction = normalizeOpponentAction(payload);
            if (selectedAction) {
                this.pendingAgentAction = selectedAction;
            } else {
                console.warn("Invalid backend agent action ignored.", payload);
            }
            if (selectedOpponentAction) {
                this.pendingOpponentAction = selectedOpponentAction;
            } else if (this.gameModeManager.isOpponentAiControlled()) {
                console.warn("Invalid backend opponent action ignored.", payload);
            }
            this.latestReward = Number(payload.agentReward ?? payload.reward ?? this.latestReward);
            this.latestOpponentReward = Number(payload.opponentReward ?? this.latestOpponentReward);
            this.latestEpsilon = Number(payload.epsilon ?? this.latestEpsilon);
            this.latestOpponentEpsilon = Number(payload.opponentEpsilon ?? this.latestOpponentEpsilon);
            this.currentEpisode = Number(payload.episode ?? this.currentEpisode);
            this.updateMetrics(payload.metrics);
        });

        this.socketManager.onTrainingStatus((payload) => {
            this.currentEpisode = Number(payload.episode ?? this.currentEpisode);
            this.latestEpsilon = Number(payload.epsilon ?? this.latestEpsilon);
            this.dashboardController.update({
                connected: Boolean(payload.connected),
                mode: payload.mode ?? (this.trainingActive ? "Training" : "Manual"),
                episode: this.currentEpisode,
                epsilon: this.latestEpsilon,
                opponentEpsilon: this.latestOpponentEpsilon,
                learningEnabled: Boolean(payload.learningEnabled),
            });
            this.updateMetrics(payload.metrics);
        });

        this.socketManager.onStateError((payload) => {
            console.warn("Backend rejected state payload.", payload);
            this.dashboardController.update({ feedback: payload.message ?? "State rejected" });
        });
    }

    configureCourt() {
        this.background.court.image = this.add.image(400, 300, "court");
        this.background.court.image.setOrigin(0.5, 0.5);
        this.background.court.image.setScale(1, 0.6);
    }

    configureBall() {
        const { x, y } = this.background.ball.initial.position;
        this.background.ball.sound = this.sound.add("smash");
        this.background.ball.image = this.physics.add.image(x, y, "ball");
        this.background.ball.image.setOrigin(0.5, 0.5);
        this.background.ball.image.setScale(0.1, 0.1);
        this.background.ball.image.setCollideWorldBounds(true);
        this.background.ball.image.setBounce(1, 1);
        this.background.ball.image.setVelocity(
            this.background.ball.initial.velocity.x,
            this.background.ball.initial.velocity.y,
        );
    }

    configureScores() {
        const textStyle = {
            fontSize: "10px",
            fontFamily: "'Press Start 2P', 'Courier New', monospace",
            fill: "black",
        };
        this.background.text.scores.team1 = this.add.text(100, 20, "Agent: 0", textStyle);
        this.background.text.scores.team2 = this.add.text(600, 20, "Opponent: 0", textStyle);
        this.background.text.scores.comboSmash = this.add.text(300, 20, "Combo smash: 0", {
            fontSize: "15px",
            fontFamily: "'Press Start 2P', 'Courier New', monospace",
            fill: "green",
            backgroundColor: "black",
        });
    }

    configurePlayers() {
        this.actors.point.sound = this.sound.add("point");
        for (let playerIndex = 0; playerIndex < this.actors.numberOfPlayers; playerIndex += 1) {
            let xCoordinate = 100;
            this.actors.players[playerIndex].team = "opponent";
            if (playerIndex === 0) {
                xCoordinate = 700;
                this.actors.players[playerIndex].team = "agent";
            }

            const racket = this.physics.add.image(xCoordinate, 300, "rackets", this.getRandomRacket());
            racket.setOrigin(0.5, 0.5);
            racket.setScale(0.2, 0.2);
            racket.setCollideWorldBounds(true);
            racket.setImmovable(true);
            this.actors.players[playerIndex].racket = racket;

            this.physics.add.collider(racket, this.background.ball.image, this.handleBallCollision, null, this);
        }
    }

    configureKeyboard() {
        this.cursors.keyboard = this.input.keyboard.createCursorKeys();
        this.cursors.keyboard.w = this.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.W);
        this.cursors.keyboard.s = this.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.S);
    }

    sendTrainingState() {
        const currentAgentDistanceToBall = Math.abs(this.background.ball.image.y - this.actors.players[0].racket.y);
        const currentOpponentDistanceToBall = Math.abs(this.background.ball.image.y - this.actors.players[1].racket.y);
        const environmentState = buildQlearningState({
            gameMode: this.gameModeManager.currentMode,
            width: this.width,
            height: this.height,
            ball: this.background.ball.image,
            agentPaddle: this.actors.players[0].racket,
            opponentPaddle: this.actors.players[1].racket,
            agentScore: this.actors.players[0].score,
            opponentScore: this.actors.players[1].score,
            lastHitBy: this.lastHitBy,
            pointWinner: this.lastScoredBy,
            episodeId: this.currentEpisode,
            previousAgentDistanceToBall: this.previousAgentDistanceToBall,
            previousOpponentDistanceToBall: this.previousOpponentDistanceToBall,
            comboSmash: this.actors.scores.comboSmash,
        });

        const stateWasSent = this.socketManager.sendStateUpdate(environmentState);
        if (!stateWasSent) {
            this.dashboardController.update({
                connected: false,
                feedback: "Training paused until backend reconnects",
            });
        }
        this.previousAgentDistanceToBall = currentAgentDistanceToBall;
        this.previousOpponentDistanceToBall = currentOpponentDistanceToBall;
    }

    handleManualAgentControls() {
        if (this.cursors.keyboard.up.isDown) {
            this.actors.players[0].racket.setVelocityY(-500);
        } else if (this.cursors.keyboard.down.isDown) {
            this.actors.players[0].racket.setVelocityY(500);
        }
    }

    handleOpponentControls() {
        if (this.cursors.keyboard.w.isDown) {
            this.actors.players[1].racket.setVelocityY(-500);
        } else if (this.cursors.keyboard.s.isDown) {
            this.actors.players[1].racket.setVelocityY(500);
        }
    }

    startTraining() {
        this.trainingActive = true;
        this.dashboardController.setTrainingActive(true);
        this.socketManager.startTraining();
        this.dashboardController.update({ feedback: "Training started" });
    }

    stopTraining() {
        this.trainingActive = false;
        this.pendingAgentAction = ACTIONS.STAY;
        this.pendingOpponentAction = ACTIONS.STAY;
        this.dashboardController.setTrainingActive(false);
        this.socketManager.stopTraining();
        this.dashboardController.update({ feedback: "Training stopped" });
    }

    changeGameMode(gameMode) {
        if (!this.gameModeManager.setMode(gameMode)) {
            return;
        }

        this.pendingAgentAction = ACTIONS.STAY;
        this.pendingOpponentAction = ACTIONS.STAY;
        this.resetEpisode();
        this.dashboardController.setSelectedMode(this.gameModeManager.currentMode);
        this.dashboardController.update({
            mode: this.gameModeManager.getModeLabel(),
            learningEnabled: this.gameModeManager.isLearningEnabled(),
            feedback: `Mode changed to ${this.gameModeManager.getModeLabel()}`,
        });
    }

    resetEpisode() {
        this.actors.players[0].score = 0;
        this.actors.players[1].score = 0;
        this.actors.scores.comboSmash = 0;
        this.lastScoredBy = null;
        this.lastHitBy = null;
        this.agentMissedBall = false;
        this.previousAgentDistanceToBall = null;
        this.previousOpponentDistanceToBall = null;
        this.pendingAgentAction = ACTIONS.STAY;
        this.pendingOpponentAction = ACTIONS.STAY;
        this.resetBall();
        this.socketManager.resetEpisode();
        this.dashboardController.update({
            scoreAgent: 0,
            scoreOpponent: 0,
            reward: 0,
            opponentReward: 0,
            feedback: "Episode reset",
        });
    }

    updateDashboard() {
        const ownership = this.gameModeManager.getOwnership();
        this.dashboardController.update({
            mode: this.gameModeManager.getModeLabel(),
            episode: this.currentEpisode,
            reward: this.latestReward,
            opponentReward: this.latestOpponentReward,
            epsilon: this.latestEpsilon,
            opponentEpsilon: this.latestOpponentEpsilon,
            scoreAgent: this.actors.players[0].score,
            scoreOpponent: this.actors.players[1].score,
            agentOwner: ownership.agent,
            opponentOwner: ownership.opponent,
            learningEnabled: this.gameModeManager.isLearningEnabled() && this.trainingActive,
            feedback: this.lastHitBy ? `${this.lastHitBy} hit the ball` : this.agentMissedBall ? "Agent missed the ball" : "Playing",
        });
    }

    updateMetrics(metrics) {
        if (!metrics) {
            return;
        }

        this.dashboardController.update({
            agentWinRate: Number(metrics.agentWinRate ?? 0),
            opponentWinRate: Number(metrics.opponentWinRate ?? 0),
            agentAverageReward: Number(metrics.agentAverageReward ?? 0),
            opponentAverageReward: Number(metrics.opponentAverageReward ?? 0),
            agentHitRate: Number(metrics.agentHitRate ?? 0),
            opponentHitRate: Number(metrics.opponentHitRate ?? 0),
            selfPlayEpisodes: Number(metrics.selfPlayEpisodes ?? 0),
        });
    }

    updateScoreText() {
        this.background.text.scores.team1.setText(`Agent: ${this.actors.players[0].score}`);
        this.background.text.scores.team2.setText(`Opponent: ${this.actors.players[1].score}`);
        this.background.text.scores.comboSmash.setText(`Combo smash: ${this.actors.scores.comboSmash}`);
    }

    resetTransientEvents() {
        this.lastScoredBy = null;
        this.lastHitBy = null;
        this.agentMissedBall = false;
    }

    getRandomRacket() {
        return Phaser.Utils.Array.GetRandom(this.textures.get("rackets").getFrameNames());
    }

    handleBallCollision(racket, ball) {
        this.background.ball.sound.play();
        this.actors.scores.comboSmash += 1;
        if (racket === this.actors.players[0].racket) {
            this.lastHitBy = "agent";
        } else if (racket === this.actors.players[1].racket) {
            this.lastHitBy = "opponent";
        }

        const differenceFromRacket = ball.y - racket.y;
        ball.setVelocityY(differenceFromRacket * randomNormal(3.5, 1.1));
        ball.setVelocityX(ball.body.velocity.x * randomNormal(1.5, 0.4));

        const angle = Math.atan2(ball.body.velocity.y, ball.body.velocity.x);
        ball.setAngle(Phaser.Math.RadToDeg(angle));
    }

    handleGoal(body, up, down, left, right) {
        if (!(left || right)) {
            return;
        }

        this.actors.point.sound.play();
        this.actors.scores.comboSmash = 0;

        if (left) {
            this.actors.players[1].score += 1;
            this.lastScoredBy = "opponent";
            this.agentMissedBall = true;
        } else if (right) {
            this.actors.players[0].score += 1;
            this.lastScoredBy = "agent";
        }

        this.resetBall(left);
    }

    resetBall(leftBoundaryWasHit = false) {
        const { x, y } = this.background.ball.initial.position;
        this.background.ball.image.setPosition(x, y);
        this.background.ball.image.setVelocity(
            randomNormal(175, 40) * (leftBoundaryWasHit ? 1 : -1),
            randomNormal(170, 25) * (leftBoundaryWasHit ? 1 : -1),
        );
    }
}

export default BasicGame;
