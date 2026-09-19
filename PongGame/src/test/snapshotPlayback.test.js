import { describe, expect, it } from "vitest";
import {
    PlaybackMetrics,
    SnapshotPlayback,
    interpolateSnapshot,
} from "../evolution/snapshotPlayback";

function snapshot(sequence, x, matchId = "match-1") {
    return {
        matchId,
        arenaId: "arena-0",
        sequence,
        agentA: { paddleX: 0.875, paddleY: x },
        agentB: { paddleX: 0.125, paddleY: 1 - x },
        ball: { x, y: x, vx: 1, vy: 1 },
    };
}

describe("snapshot playback", () => {
    it("interpolates only between real snapshot endpoints", () => {
        const result = interpolateSnapshot(snapshot(1, 0.2), snapshot(2, 0.6), 0.5);
        expect(result.ball.x).toBeCloseTo(0.4);
        expect(result.agentA.paddleY).toBeCloseTo(0.4);
        expect(result.agentB.paddleY).toBeCloseTo(0.6);
    });

    it("uses fixed 60 Hz simulation time and playback speed", () => {
        const playback = new SnapshotPlayback();
        playback.push(snapshot(1, 0.2), 1);
        playback.push(snapshot(3, 0.6), 2);

        expect(playback.sample(1000 / 120).ball.x).toBeCloseTo(0.4);
        expect(playback.sample(1000 / 120).ball.x).toBeCloseTo(0.6);
    });

    it("measures per-arena snapshots, simulation steps, and render FPS", () => {
        const metrics = new PlaybackMetrics(1);
        metrics.recordSnapshot(snapshot(10, 0.2));
        metrics.recordSnapshot(snapshot(70, 0.6));
        let result = null;
        for (let frame = 0; frame < 60; frame += 1) {
            result = metrics.recordFrame(1000 / 60) ?? result;
        }
        expect(result.fps).toBeCloseTo(60);
        expect(result.snapshotsPerSecond).toBeCloseTo(2);
        expect(result.simulationStepsPerSecond).toBeCloseTo(60);
    });
});
