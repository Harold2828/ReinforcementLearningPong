import Phaser from "phaser";
import { EVOLUTION_EVENT_TYPES, SOURCE_LABEL, layoutArenas } from "../evolution/evolutionContract";

/**
 * EVOLUTION_TRAINING view: renders five Pong arenas from snapshot envelopes.
 * The scene is a pure visualization layer — no physics, no training, no ball
 * stepping. Position updates come only from the subscribed feed.
 */

class EvolutionTrainingScene extends Phaser.Scene {
    constructor() {
        super({ key: "EvolutionTraining" });
        this.latest = {};
        this.feed = null;
        this.unsubscribe = null;
    }

    init(data) {
        this.feed = data?.feed ?? null;
    }

    create() {
        const { width, height } = this.sys.game.canvas;
        this.layout = layoutArenas(width, height);

        this.add.text(10, 10, `${SOURCE_LABEL.MOCK} DATA — authoritative backend snapshots pending (SPEC-05/06)`, {
            fontSize: "12px",
            fontFamily: "'Press Start 2P', 'Courier New', monospace",
            fill: "#f4d03f",
        });

        this.arenas = this.layout.map((rect) => this.createArena(rect));
        if (this.feed) {
            this.unsubscribe = this.feed.subscribe((event) => this.handleEvent(event));
        }
        this.events.once(Phaser.Scenes.Events.SHUTDOWN, () => {
            if (this.unsubscribe) {
                this.unsubscribe();
                this.unsubscribe = null;
            }
        });
    }

    createArena(rect) {
        const frame = this.add.graphics();
        frame.lineStyle(2, 0x47d18c, 1);
        frame.strokeRect(rect.x, rect.y, rect.width, rect.height);
        frame.lineStyle(1, 0x2d3138, 1);
        frame.lineBetween(rect.x + rect.width / 2, rect.y, rect.x + rect.width / 2, rect.y + rect.height);

        const paddleA = this.add.graphics();
        const paddleB = this.add.graphics();
        const ball = this.add.graphics();
        const hud = this.add.text(rect.x + 4, rect.y + 4, "", {
            fontSize: "9px",
            fontFamily: "'Press Start 2P', 'Courier New', monospace",
            fill: "#f4f7fb",
            backgroundColor: "rgba(0,0,0,0.6)",
        });

        return { rect, frame, paddleA, paddleB, ball, hud };
    }

    handleEvent(event) {
        if (event?.type === EVOLUTION_EVENT_TYPES.MATCH_SNAPSHOT) {
            this.latest[event.arenaId] = event;
        }
    }

    update() {
        this.arenas.forEach((arena) => {
            const snapshot = this.latest[arena.rect.arenaId];
            if (snapshot) {
                this.redrawArena(arena, snapshot);
            }
        });
    }

    redrawArena(arena, snapshot) {
        const { rect } = arena;
        const ballX = rect.x + snapshot.ball.x * rect.width;
        const ballY = rect.y + snapshot.ball.y * rect.height;
        const paddleHeight = Math.max(12, rect.height * 0.18);
        const paddleWidth = 5;

        arena.paddleA.clear();
        arena.paddleA.fillStyle(0x2f80ed, 1);
        arena.paddleA.fillRect(
            rect.x + snapshot.agentA.paddleX * rect.width - paddleWidth / 2,
            rect.y + snapshot.agentA.paddleY * rect.height - paddleHeight / 2,
            paddleWidth,
            paddleHeight,
        );

        arena.paddleB.clear();
        arena.paddleB.fillStyle(0xe86f2f, 1);
        arena.paddleB.fillRect(
            rect.x + snapshot.agentB.paddleX * rect.width - paddleWidth / 2,
            rect.y + snapshot.agentB.paddleY * rect.height - paddleHeight / 2,
            paddleWidth,
            paddleHeight,
        );

        arena.ball.clear();
        arena.ball.fillStyle(0xffffff, 1);
        arena.ball.fillCircle(ballX, ballY, 4);

        arena.hud.setText(
            [
                `A ${snapshot.agentA.id} g${snapshot.agentA.generation} S:${snapshot.agentA.score} \u03b5${snapshot.agentA.epsilon.toFixed(2)}`,
                `B ${snapshot.agentB.id} g${snapshot.agentB.generation} S:${snapshot.agentB.score} \u03b5${snapshot.agentB.epsilon.toFixed(2)}`,
                `steps ${snapshot.step} elapsed ${snapshot.elapsedSteps} | ${snapshot.status}`,
            ].join("\n"),
        );
    }
}

export default EvolutionTrainingScene;