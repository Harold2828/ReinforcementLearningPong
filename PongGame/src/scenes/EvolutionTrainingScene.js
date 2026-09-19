import Phaser from "phaser";
import { EVOLUTION_EVENT_TYPES, layoutArenas } from "../evolution/evolutionContract";
import { applySnapshot, createArenaBoard, snapshotFor } from "../evolution/arenaState";
import {
    applySnapshotToCourt,
    createSnapshotCourt,
    preloadPongAssets,
} from "../components/pongCourtView";
import { PlaybackMetrics, SnapshotPlayback } from "../evolution/snapshotPlayback";

/** Pure LIVE snapshot renderer; authoritative physics remains on the backend. */
class EvolutionTrainingScene extends Phaser.Scene {
    constructor() {
        super({ key: "EvolutionTraining" });
        this.board = createArenaBoard();
        this.feed = null;
        this.unsubscribe = null;
        this.playbacks = new Map();
        this.metrics = null;
    }

    init(data) {
        this.feed = data?.feed ?? null;
    }

    preload() {
        preloadPongAssets(this);
    }

    create() {
        const { width, height } = this.sys.game.canvas;
        this.layout = layoutArenas(width, height);
        this.title = this.add.text(10, 10, "EVOLUTION TRAINING — LIVE", {
            fontSize: "12px",
            fontFamily: "'Press Start 2P', 'Courier New', monospace",
            fill: "#f4d03f",
        });
        this.arenas = this.layout.map((bounds) => createSnapshotCourt(this, bounds));
        this.metrics = new PlaybackMetrics(this.arenas.length);
        for (const arena of this.arenas) {
            this.playbacks.set(arena.bounds.arenaId, new SnapshotPlayback());
        }
        if (this.feed) {
            this.unsubscribe = this.feed.subscribe((event) => this.handleEvent(event));
            this.feed.startRun?.({ runUuid: `ui-evolution-${Date.now()}` });
        }
        this.events.once(Phaser.Scenes.Events.SHUTDOWN, () => {
            if (this.unsubscribe) this.unsubscribe();
            this.unsubscribe = null;
            this.feed?.stopRun?.();
        });
    }

    handleEvent(event) {
        if (event?.type === EVOLUTION_EVENT_TYPES.MATCH_SNAPSHOT) {
            const result = applySnapshot(this.board, event);
            if (result.accepted) {
                this.playbacks.get(event.arenaId)?.push(event, this.feed?.playbackSpeed ?? 1);
                this.metrics.recordSnapshot(event);
                this.latestPerformance = event.performance;
            }
        }
    }

    update(_time, delta) {
        this.arenas.forEach((arena) => {
            const playback = this.playbacks.get(arena.bounds.arenaId);
            const snapshot = playback?.sample(delta) ?? snapshotFor(this.board, arena.bounds.arenaId);
            if (snapshot) {
                applySnapshotToCourt(arena, snapshot);
            }
        });
        const performance = this.metrics.recordFrame(delta);
        if (performance) {
            const targetSteps = 60 * (this.feed?.playbackSpeed ?? 1);
            const actualSteps = this.latestPerformance?.simulationStepsPerSecond
                ?? performance.simulationStepsPerSecond;
            this.title.setText([
                `LIVE ${this.feed?.playbackSpeed ?? 1}×`,
                `SIM ${actualSteps.toFixed(0)}/${targetSteps}`,
                `SNAP ${performance.snapshotsPerSecond.toFixed(0)}/s`,
                `FPS ${performance.fps.toFixed(0)}`,
                `U/D ${(this.latestPerformance?.updateToDataRatio ?? 0).toFixed(2)}`,
            ].join(" | "));
        }
    }
}

export default EvolutionTrainingScene;
