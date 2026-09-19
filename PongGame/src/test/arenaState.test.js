import { describe, expect, it } from "vitest";
import { applySnapshot, createArenaBoard, hasExpectedParticipants, participantIds } from "../evolution/arenaState";
import { EVOLUTION_EVENT_TYPES, MATCH_STATUS } from "../evolution/evolutionContract";

function snapshot(arenaId, matchId, sequence, agentA = "agent-0", agentB = "agent-1") {
    return {
        type: EVOLUTION_EVENT_TYPES.MATCH_SNAPSHOT,
        runId: "run-1",
        generationId: "generation-1",
        matchId,
        arenaId,
        sequence,
        stateTimestamp: 1000 + sequence,
        step: sequence * 5,
        elapsedSteps: sequence * 10,
        status: MATCH_STATUS.RUNNING,
        agentA: { id: agentA, generation: 0, paddleX: 0.06, paddleY: 0.5, epsilon: 0.9 },
        agentB: { id: agentB, generation: 0, paddleX: 0.94, paddleY: 0.5, epsilon: 0.9 },
        ball: { x: 0.5, y: 0.3, vx: -1, vy: 0.4 },
    };
}

describe("arena board", () => {
    it("accepts new snapshots for each arena", () => {
        const board = createArenaBoard();
        const first = applySnapshot(board, snapshot("arena-0", "m1", 1));
        const second = applySnapshot(board, snapshot("arena-0", "m1", 2));
        expect(first.accepted).toBe(true);
        expect(second.accepted).toBe(true);
    });

    it("rejects stale or out-of-order snapshots for the same match session", () => {
        const board = createArenaBoard();
        applySnapshot(board, snapshot("arena-0", "m1", 3));
        const stale = applySnapshot(board, snapshot("arena-0", "m1", 3));
        const older = applySnapshot(board, snapshot("arena-0", "m1", 2));
        expect(stale.reason).toBe("stale_or_out_of_order");
        expect(older.reason).toBe("stale_or_out_of_order");
    });

    it("accepts a sequence restart for a new match session in the same arena", () => {
        const board = createArenaBoard();
        applySnapshot(board, snapshot("arena-0", "m1", 12));
        const resumed = applySnapshot(board, snapshot("arena-0", "m2", 1));
        expect(resumed.accepted).toBe(true);
        expect(resumed.arena.matchId).toBe("m2");
    });

    it("rejects unknown arenas and non-snapshot events", () => {
        const board = createArenaBoard();
        expect(applySnapshot(board, snapshot("arena-42", "m1", 1)).reason).toBe("unknown_arena");
        expect(applySnapshot(board, { ...snapshot("arena-0", "m1", 1), type: "population" }).reason).toBe("not_a_snapshot");
    });

    it("tracks twelve unique participants across six arenas", () => {
        const board = createArenaBoard();
        applySnapshot(board, snapshot("arena-0", "m1", 1, "agent-0", "agent-1"));
        applySnapshot(board, snapshot("arena-1", "m1", 1, "agent-2", "agent-3"));
        applySnapshot(board, snapshot("arena-2", "m1", 1, "agent-4", "agent-5"));
        applySnapshot(board, snapshot("arena-3", "m1", 1, "agent-6", "agent-7"));
        applySnapshot(board, snapshot("arena-4", "m1", 1, "agent-8", "agent-9"));
        applySnapshot(board, snapshot("arena-5", "m1", 1, "agent-10", "agent-11"));
        expect(participantIds(board)).toHaveLength(12);
        expect(hasExpectedParticipants(board)).toBe(true);
    });
});
