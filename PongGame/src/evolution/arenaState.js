import { AGENT_COUNT, ARENA_IDS, isStaleMatchSnapshot } from "./evolutionContract";

/**
 * Pure arena board. Holds the latest snapshot per arena, enforces monotonic
 * per-match ordering, and tracks the participants shown across all arenas so
 * cross-arena contamination and duplicate participants are detectable.
 */

function createEmptyArenaState() {
    return { matchId: null, sequence: 0, snapshot: null };
}

export function createArenaBoard() {
    const arenas = {};
    for (const arenaId of ARENA_IDS) {
        arenas[arenaId] = createEmptyArenaState();
    }
    return { arenas, participants: new Set() };
}

/**
 * Applies a validated match snapshot. Returns { accepted, arena, reason }.
 * Rejected when the arena is unknown, the envelope is not a snapshot, or the
 * snapshot is stale/out-of-order for the same match session.
 */
export function applySnapshot(board, envelope) {
    if (envelope?.type !== "match_snapshot") {
        return { accepted: false, reason: "not_a_snapshot" };
    }
    const arenaState = board.arenas[envelope.arenaId];
    if (!arenaState) {
        return { accepted: false, reason: "unknown_arena" };
    }
    if (isStaleMatchSnapshot(arenaState, envelope)) {
        return { accepted: false, reason: "stale_or_out_of_order" };
    }

    if (arenaState.matchId !== envelope.matchId) {
        arenaState.matchId = envelope.matchId;
        arenaState.sequence = envelope.sequence;
    }
    arenaState.snapshot = envelope;

    board.participants.add(envelope.agentA.id);
    board.participants.add(envelope.agentB.id);
    return { accepted: true, arena: arenaState, reason: null };
}

/** All unique participant ids expected across the configured arenas. */
export function participantIds(board) {
    return Array.from(board.participants).sort();
}

export function hasExpectedParticipants(board) {
    return participantIds(board).length === AGENT_COUNT;
}

export function snapshotFor(board, arenaId) {
    return board.arenas[arenaId]?.snapshot ?? null;
}

export function allSnapshots(board) {
    return ARENA_IDS.map((arenaId) => board.arenas[arenaId]?.snapshot).filter(Boolean);
}
