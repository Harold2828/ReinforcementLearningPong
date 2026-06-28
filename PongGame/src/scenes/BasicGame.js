import Phaser from "phaser";
import SocketManager from "../utils/socketManager";
import { randomNormal } from "../utils/customRandom";
import { DashboardController } from "../utils/dashboardController";
import {
    ACTIONS,
    applyAgentActionToPaddle,
    buildQlearningState,
    normalizeAgentAction,
} from "../utils/gameState";

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
        this.trainingActive = true;
        this.pendingAgentAction = ACTIONS.STAY;
        this.lastScoredBy = null;
        this.ballWasHit = false;
        this.agentMissedBall = false;
        this.previousDistanceToBall = null;
        this.latestReward = 0;
        this.latestEpsilon = 0;
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
        });
        this.dashboardController.setTrainingActive(true);

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
            applyAgentActionToPaddle(this.actors.players[0].racket, this.pendingAgentAction);
            this.sendTrainingState();
        } else {
            this.handleManualAgentControls();
        }

        this.handleOpponentControls();
        this.updateScoreText();
        this.updateDashboard();
        this.resetTransientEvents();
    }

    configureSocket() {
        const backendUrl = import.meta.env.VITE_BACKEND_URL || "http://127.0.0.1:5001";
        this.socketManager = new SocketManager(backendUrl);

        this.socketManager.onConnectionChange((isConnected) => {
            this.dashboardController.update({
                connected: isConnected,
                mode: this.trainingActive ? "Training" : "Manual",
                feedback: isConnected ? "Backend connected" : "Backend disconnected",
            });
        });

        this.socketManager.onAiMove((payload) => {
            const selectedAction = normalizeAgentAction(payload);
            if (!selectedAction) {
                console.warn("Invalid backend action ignored.", payload);
                return;
            }
            this.pendingAgentAction = selectedAction;
            this.latestReward = Number(payload.reward ?? this.latestReward);
            this.latestEpsilon = Number(payload.epsilon ?? this.latestEpsilon);
            this.currentEpisode = Number(payload.episode ?? this.currentEpisode);
        });

        this.socketManager.onTrainingStatus((payload) => {
            this.currentEpisode = Number(payload.episode ?? this.currentEpisode);
            this.latestEpsilon = Number(payload.epsilon ?? this.latestEpsilon);
            this.dashboardController.update({
                connected: Boolean(payload.connected),
                mode: payload.mode ?? (this.trainingActive ? "Training" : "Manual"),
                episode: this.currentEpisode,
                epsilon: this.latestEpsilon,
            });
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
        const currentDistanceToBall = Math.abs(this.background.ball.image.y - this.actors.players[0].racket.y);
        const environmentState = buildQlearningState({
            width: this.width,
            height: this.height,
            ball: this.background.ball.image,
            agentPaddle: this.actors.players[0].racket,
            opponentPaddle: this.actors.players[1].racket,
            scoreAgent: this.actors.players[0].score,
            scoreOpponent: this.actors.players[1].score,
            done: this.lastScoredBy !== null,
            agentHitBall: this.ballWasHit,
            agentMissedBall: this.agentMissedBall,
            agentScored: this.lastScoredBy === "agent",
            previousDistanceToBall: this.previousDistanceToBall,
        });

        const stateWasSent = this.socketManager.sendStateUpdate(environmentState);
        if (!stateWasSent) {
            this.dashboardController.update({
                connected: false,
                feedback: "Training paused until backend reconnects",
            });
        }
        this.previousDistanceToBall = currentDistanceToBall;
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
        this.dashboardController.update({ mode: "Training", feedback: "Training started" });
    }

    stopTraining() {
        this.trainingActive = false;
        this.pendingAgentAction = ACTIONS.STAY;
        this.dashboardController.setTrainingActive(false);
        this.socketManager.stopTraining();
        this.dashboardController.update({ mode: "Manual", feedback: "Training stopped" });
    }

    resetEpisode() {
        this.actors.players[0].score = 0;
        this.actors.players[1].score = 0;
        this.actors.scores.comboSmash = 0;
        this.lastScoredBy = null;
        this.ballWasHit = false;
        this.agentMissedBall = false;
        this.previousDistanceToBall = null;
        this.pendingAgentAction = ACTIONS.STAY;
        this.resetBall();
        this.socketManager.resetEpisode();
        this.dashboardController.update({
            scoreAgent: 0,
            scoreOpponent: 0,
            reward: 0,
            feedback: "Episode reset",
        });
    }

    updateDashboard() {
        this.dashboardController.update({
            mode: this.trainingActive ? "Training" : "Manual",
            episode: this.currentEpisode,
            reward: this.latestReward,
            epsilon: this.latestEpsilon,
            scoreAgent: this.actors.players[0].score,
            scoreOpponent: this.actors.players[1].score,
            feedback: this.ballWasHit ? "Agent hit the ball" : this.agentMissedBall ? "Agent missed the ball" : "Playing",
        });
    }

    updateScoreText() {
        this.background.text.scores.team1.setText(`Agent: ${this.actors.players[0].score}`);
        this.background.text.scores.team2.setText(`Opponent: ${this.actors.players[1].score}`);
        this.background.text.scores.comboSmash.setText(`Combo smash: ${this.actors.scores.comboSmash}`);
    }

    resetTransientEvents() {
        this.lastScoredBy = null;
        this.ballWasHit = false;
        this.agentMissedBall = false;
    }

    getRandomRacket() {
        return Phaser.Utils.Array.GetRandom(this.textures.get("rackets").getFrameNames());
    }

    handleBallCollision(racket, ball) {
        this.background.ball.sound.play();
        this.actors.scores.comboSmash += 1;
        if (racket === this.actors.players[0].racket) {
            this.ballWasHit = true;
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
