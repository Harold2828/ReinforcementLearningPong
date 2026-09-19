import Phaser from "phaser";
import { EVOLUTION_EVENT_TYPES, layoutArenas } from "../evolution/evolutionContract";
import { applySnapshot, createArenaBoard, snapshotFor } from "../evolution/arenaState";
import {
    applySnapshotToCourt,
    createSnapshotCourt,
    preloadPongAssets,
} from "../components/pongCourtView";

/** Pure LIVE snapshot renderer; authoritative physics remains on the backend. */
class EvolutionTrainingScene extends Phaser.Scene {
    constructor() {
        super({ key: "EvolutionTraining" });
        this.board = createArenaBoard();
        this.feed = null;
        this.unsubscribe = null;
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
        this.add.text(10, 10, "EVOLUTION TRAINING — LIVE", {
            fontSize: "12px",
            fontFamily: "'Press Start 2P', 'Courier New', monospace",
            fill: "#f4d03f",
        });
        this.arenas = this.layout.map((bounds) => createSnapshotCourt(this, bounds));
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
            applySnapshot(this.board, event);
        }
    }

    update() {
        this.arenas.forEach((arena) => {
            const snapshot = snapshotFor(this.board, arena.bounds.arenaId);
            if (snapshot && snapshot !== arena.previous) {
                applySnapshotToCourt(arena, snapshot);
            }
        });
    }
}

export default EvolutionTrainingScene;
