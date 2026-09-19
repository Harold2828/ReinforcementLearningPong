import { describe, expect, it } from "vitest";
import {
    MockEvolutionFeed,
    createMockArenaAssignments,
    createMockPopulation,
    simulateMatchStep,
} from "../evolution/mockEvolutionFeed";
import { AGENT_COUNT, AGENTS_PER_ARENA, ARENA_COUNT, SOURCE_LABEL, validateMatchSnapshot } from "../evolution/evolutionContract";

describe("mock evolution feed generators", () => {
    it("maps ten unique agents onto five arenas with two per arena", () => {
        const assignments = createMockArenaAssignments();
        expect(assignments).toHaveLength(ARENA_COUNT);
        const participantIds = assignments.flatMap((entry) => [entry.agentAId, entry.agentBId]);
        expect(participantIds).toHaveLength(AGENT_COUNT);
        expect(new Set(participantIds).size).toBe(AGENT_COUNT);
        for (const entry of assignments) {
            expect([entry.agentAId, entry.agentBId]).toHaveLength(AGENTS_PER_ARENA);
        }
    });

    it("creates a deterministic ten-member population", () => {
        const first = createMockPopulation(42);
        const second = createMockPopulation(42);
        expect(first).toHaveLength(AGENT_COUNT);
        expect(JSON.stringify(first)).toBe(JSON.stringify(second));
    });

    it("advances match state predictably and decays epsilon", () => {
        const state = {
            ball: { x: 0.5, y: 0.5, vx: -1, vy: 0 },
            agentA: { paddleY: 0.5, score: 0, epsilon: 1.0 },
            agentB: { paddleY: 0.5, score: 0, epsilon: 1.0 },
            steps: 0,
            status: "running",
        };
        const runner = () => 0.5;
        const first = simulateMatchStep(state, runner);
        const second = simulateMatchStep(first, runner);
        expect(first.steps).toBe(1);
        expect(second.steps).toBe(2);
        expect(second.epsilonA).toBeLessThan(first.epsilonA);
        expect(second.ball.x).not.toBe(first.ball.x);
    });
});

describe("MockEvolutionFeed", () => {
    function collectingFeed() {
        const feed = new MockEvolutionFeed({ seed: 7 });
        const events = [];
        feed.subscribe((event) => events.push(event));
        feed.emitInitialEvents();
        return { feed, events };
    }

    it("emits clearly labeled mock lifecycle events on start", () => {
        const { events } = collectingFeed();
        expect(events.map((event) => event.type)).toEqual([
            "population",
            "champion_promotion",
        ]);
        for (const event of events) {
            expect(event.source).toBe(SOURCE_LABEL.MOCK);
        }
    });

    it("publishes per-arena snapshots that conform to the contract and advance monotonically", () => {
        const { feed, events } = collectingFeed();
        feed.tick();
        feed.tick();

        const snapshots = events.filter((event) => event.type === "match_snapshot");
        expect(snapshots).toHaveLength(ARENA_COUNT * 2);
        for (const snapshot of snapshots) {
            expect(validateMatchSnapshot(snapshot).ok).toBe(true);
            expect(snapshot.source).toBe(SOURCE_LABEL.MOCK);
        }
        for (const arenaId of ["arena-0", "arena-1", "arena-2", "arena-3", "arena-4"]) {
            const sequences = snapshots.filter((event) => event.arenaId === arenaId).map((event) => event.sequence);
            expect(sequences[0]).toBe(1);
            expect(sequences[1]).toBeGreaterThan(sequences[0]);
        }
        expect(snapshots[snapshots.length - 1].elapsedSteps).toBeGreaterThan(snapshots[0].elapsedSteps);
    });

    it("restarts sequence and opens a new session when a match reaches terminal", () => {
        const { feed, events } = collectingFeed();
        feed.sessions[0].matchState.status = "terminal";
        feed.tick();
        feed.tick();

        const arenaZero = events.filter((event) => event.arenaId === "arena-0" && event.type === "match_snapshot");
        const terminalSnapshot = arenaZero[arenaZero.length - 2];
        const resumedSnapshot = arenaZero[arenaZero.length - 1];
        expect(terminalSnapshot.status).toBe("terminal");
        expect(resumedSnapshot.matchId).not.toBe(terminalSnapshot.matchId);
        expect(resumedSnapshot.sequence).toBe(1);
        expect(resumedSnapshot.status).toBe("running");
    });

    it("drives every arena session deterministically across ticks", () => {
        const { feed, events } = collectingFeed();
        for (let index = 0; index < 5; index += 1) {
            feed.tick();
        }
        const snapshots = events.filter((event) => event.type === "match_snapshot");
        const perArena = {};
        for (const snapshot of snapshots) {
            perArena[snapshot.arenaId] = (perArena[snapshot.arenaId] ?? 0) + 1;
        }
        expect(Object.values(perArena).every((count) => count === 5)).toBe(true);
    });
});