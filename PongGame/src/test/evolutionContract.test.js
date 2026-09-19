import { describe, expect, it } from "vitest";
import {
    AGENT_COUNT,
    ARENA_COUNT,
    ARENA_IDS,
    EVOLUTION_EVENT_TYPES,
    MATCH_STATUS,
    layoutArenas,
    validateMatchSnapshot,
    validatePopulationEvent,
} from "../evolution/evolutionContract";

function validSnapshot(overrides = {}) {
    return {
        type: EVOLUTION_EVENT_TYPES.MATCH_SNAPSHOT,
        runId: "run-1",
        generationId: "generation-1",
        matchId: "match-1-session-1",
        arenaId: "arena-0",
        sequence: 1,
        stateTimestamp: 1000,
        step: 5,
        elapsedSteps: 60,
        status: MATCH_STATUS.RUNNING,
        agentA: { id: "agent-0", generation: 0, paddleX: 0.06, paddleY: 0.5, epsilon: 0.9, score: 0 },
        agentB: { id: "agent-1", generation: 0, paddleX: 0.94, paddleY: 0.5, epsilon: 0.9, score: 0 },
        ball: { x: 0.5, y: 0.3, vx: -1, vy: 0.4 },
        ...overrides,
    };
}

describe("evolution data contract", () => {
    it("validates a conforming match snapshot", () => {
        const result = validateMatchSnapshot(validSnapshot());
        expect(result.ok).toBe(true);
        expect(result.errors).toEqual([]);
    });

    it.each([
        [{ type: "other" }, "wrong type"],
        [{ arenaId: "arena-9" }, "unknown arena"],
        [{ sequence: 0 }, "sequence not positive"],
        [{ step: -1 }, "negative step"],
        [{ ball: { x: "NaN", y: 0.2, vx: 1, vy: 0 } }, "non-finite ball coordinate"],
        [{ agentA: { ...validSnapshot().agentA, id: "" } }, "empty agent id"],
        [{ status: "finished" }, "unsupported status"],
    ])("rejects invalid snapshot (%s)", (overrides) => {
        const result = validateMatchSnapshot(validSnapshot(overrides));
        expect(result.ok).toBe(false);
        expect(result.errors.length).toBeGreaterThan(0);
    });

    it("configures six arenas and twelve agents", () => {
        expect(ARENA_IDS).toHaveLength(ARENA_COUNT);
        expect(ARENA_IDS).toEqual(["arena-0", "arena-1", "arena-2", "arena-3", "arena-4", "arena-5"]);
        expect(AGENT_COUNT).toBe(12);
    });

    it("lays out six courts in a 3-plus-3 grid within the viewport", () => {
        const arenas = layoutArenas(800, 600);
        expect(arenas).toHaveLength(6);
        expect(arenas[0].arenaId).toBe("arena-0");
        for (const arena of arenas) {
            expect(arena.x).toBeGreaterThanOrEqual(0);
            expect(arena.y).toBeGreaterThanOrEqual(0);
            expect(arena.x + arena.width).toBeLessThanOrEqual(800);
            expect(arena.y + arena.height).toBeLessThanOrEqual(600);
            expect(arena.width / arena.height).toBeCloseTo(4 / 3, 2);
        }
        expect(arenas[0].y).toBe(arenas[1].y);
        expect(arenas[2].y).toBe(arenas[1].y);
        expect(arenas[3].y).toBe(arenas[4].y);
        expect(arenas[3].y).toBeGreaterThan(arenas[0].y);
    });

    it("validates the population event shape", () => {
        const agents = Array.from({ length: AGENT_COUNT }, (_, index) => ({ agentId: `agent-${index}` }));
        const ok = validatePopulationEvent({ type: EVOLUTION_EVENT_TYPES.POPULATION, runId: "run-1", agents });
        expect(ok.ok).toBe(true);
        const bad = validatePopulationEvent({ type: EVOLUTION_EVENT_TYPES.POPULATION, runId: "run-1", agents: [] });
        expect(bad.ok).toBe(false);
    });
});
