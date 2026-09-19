import { describe, expect, it, vi } from "vitest";
import { LiveEvolutionFeed } from "../evolution/liveEvolutionFeed";
import { createMockPopulation } from "../evolution/mockEvolutionFeed";
import { SOURCE_LABEL } from "../evolution/evolutionContract";

function createSensor() {
    const handlers = {};
    return {
        startEvolutionRun: vi.fn().mockResolvedValue({ status: "started" }),
        stopEvolutionRun: vi.fn().mockResolvedValue({ status: "stopping" }),
        setEvolutionPlaybackSpeed: vi.fn().mockResolvedValue({ status: "running" }),
        onEvolutionEvent(callback) {
            handlers.live = callback;
        },
        onConnectionChange(callback) {
            handlers.connection = callback;
        },
        handlers,
    };
}

function createFallback() {
    const listeners = [];
    return {
        start() {},
        stop() {},
        subscribe(listener) {
            listeners.push(listener);
            return () => {
                listeners.splice(listeners.indexOf(listener), 1);
            };
        },
        publish(event) {
            listeners.forEach((listener) => listener(event));
        },
    };
}

function validSnapshot() {
    return {
        type: "match_snapshot",
        runId: "run-1",
        generationId: "generation-1",
        matchId: "match-1",
        arenaId: "arena-0",
        sequence: 1,
        stateTimestamp: 1_700_000_000_000,
        step: 10,
        elapsedSteps: 10,
        status: "running",
        agentA: { id: "agent-0", generation: 0, paddleX: 0.06, paddleY: 0.5, epsilon: 0.9 },
        agentB: { id: "agent-1", generation: 0, paddleX: 0.94, paddleY: 0.5, epsilon: 0.9 },
        ball: { x: 0.5, y: 0.5, vx: 1, vy: 0.2 },
    };
}

describe("LiveEvolutionFeed", () => {
    it("forwards fallback mock events while disconnected and suppresses them once live", () => {
        const sensor = createSensor();
        const fallback = createFallback();
        const feed = new LiveEvolutionFeed({ sensor, fallback });
        const events = [];
        feed.subscribe((event) => events.push(event));
        feed.start();

        fallback.publish({ type: "population", source: SOURCE_LABEL.MOCK });
        expect(events).toHaveLength(1);

        sensor.handlers.connection(true);
        fallback.publish({ type: "population", source: SOURCE_LABEL.MOCK });
        expect(events).toHaveLength(1);
    });

    it("receives, validates, and forwards a LIVE match snapshot", () => {
        const sensor = createSensor();
        const feed = new LiveEvolutionFeed({ sensor });
        const events = [];
        feed.subscribe((event) => events.push(event));
        feed.start();
        sensor.handlers.connection(true);

        sensor.handlers.live(validSnapshot());

        expect(events).toHaveLength(1);
        expect(events[0]).toMatchObject({ type: "match_snapshot", arenaId: "arena-0" });
        expect(events[0].source).toBe(SOURCE_LABEL.LIVE);
    });

    it("validates and fans out a six-arena snapshot batch", () => {
        const sensor = createSensor();
        const feed = new LiveEvolutionFeed({ sensor });
        const events = [];
        feed.subscribe((event) => events.push(event));
        feed.start();
        sensor.handlers.connection(true);
        const snapshots = Array.from({ length: 6 }, (_, index) => ({
            ...validSnapshot(),
            arenaId: `arena-${index}`,
            matchId: `match-${index}`,
        }));

        sensor.handlers.live({ type: "match_snapshot_batch", snapshots });

        expect(events).toHaveLength(6);
        expect(events.map((event) => event.arenaId)).toEqual([
            "arena-0", "arena-1", "arena-2", "arena-3", "arena-4", "arena-5",
        ]);
    });

    it("starts a requested run after connecting and stops it on demand", async () => {
        const sensor = createSensor();
        const feed = new LiveEvolutionFeed({ sensor });
        feed.start();

        await feed.startRun({ runUuid: "ui-run" });
        expect(sensor.startEvolutionRun).not.toHaveBeenCalled();

        sensor.handlers.connection(true);
        await vi.waitFor(() => expect(sensor.startEvolutionRun).toHaveBeenCalledWith({
            runUuid: "ui-run",
            playbackSpeed: 1,
        }));

        await feed.stopRun();
        expect(sensor.stopEvolutionRun).toHaveBeenCalledTimes(1);
    });

    it("changes the live playback speed without restarting training", async () => {
        const sensor = createSensor();
        const feed = new LiveEvolutionFeed({ sensor });

        await feed.setPlaybackSpeed(4);

        expect(feed.playbackSpeed).toBe(4);
        expect(sensor.setEvolutionPlaybackSpeed).toHaveBeenCalledWith(4);
        expect(() => feed.setPlaybackSpeed(3)).toThrow(/1, 2, or 4/);
    });

    it("drops an invalid match snapshot with a warning instead of rendering it", () => {
        const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
        const sensor = createSensor();
        const feed = new LiveEvolutionFeed({ sensor });
        const events = [];
        feed.subscribe((event) => events.push(event));
        feed.start();
        sensor.handlers.connection(true);

        sensor.handlers.live({ ...validSnapshot(), arenaId: "arena-99", ball: { x: "nope" } });

        expect(events).toHaveLength(0);
        expect(warn).toHaveBeenCalledTimes(1);
        warn.mockRestore();
    });

    it("forwards a contract-conforming population and drops non-conforming ones", () => {
        const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
        const sensor = createSensor();
        const feed = new LiveEvolutionFeed({ sensor });
        const events = [];
        feed.subscribe((event) => events.push(event));
        feed.start();
        sensor.handlers.connection(true);

        sensor.handlers.live({ type: "population", runId: "run-1", agents: createMockPopulation(1) });
        sensor.handlers.live({ type: "population", runId: "run-1", agents: [] });

        expect(events).toHaveLength(1);
        expect(events[0]).toMatchObject({ type: "population", runId: "run-1" });
        expect(events[0].agents).toHaveLength(12);
        expect(warn).toHaveBeenCalledTimes(1);
        warn.mockRestore();
    });
});
