/**
 * SPEC-03 frontend data contract.
 * Backend authoritative simulation is NOT yet wired (SPEC-05/06); the mock
 * feed, evolution scenes, and dashboard are built against this contract.
 */

export const EVOLUTION_VIEWS = Object.freeze({
    CLASSIC: "classic",
    EVOLUTION_TRAINING: "evolution-training",
    EVOLUTION_DASHBOARD: "evolution-dashboard",
    HUMAN_VS_CHAMPION: "human-vs-champion",
});

export const ARENA_COUNT = 5;
export const AGENT_COUNT = 10;
export const AGENTS_PER_ARENA = 2;

export const EVOLUTION_EVENT_TYPES = Object.freeze({
    MATCH_SNAPSHOT: "match_snapshot",
    MATCH_SNAPSHOT_BATCH: "match_snapshot_batch",
    POPULATION: "population",
    EVALUATION: "evaluation",
    CHAMPION_PROMOTION: "champion_promotion",
    CONTROLLED_ERROR: "controlled_error",
});

export const ARENA_IDS = Object.freeze(["arena-0", "arena-1", "arena-2", "arena-3", "arena-4"]);

export const MATCH_STATUS = Object.freeze({
    RUNNING: "running",
    TERMINAL: "terminal",
});

export const SOURCE_LABEL = Object.freeze({
    MOCK: "MOCK",
    LIVE: "LIVE",
    note: "MOCK events are locally generated placeholders; LIVE events arrive over Socket.IO from the SPEC-06 orchestrator.",
});

function presentObjectFields(value) {
    const fields = value ?? {};
    if (typeof fields !== "object" || Array.isArray(fields)) {
        return { x: undefined, y: undefined, vx: undefined, vy: undefined };
    }
    return fields;
}

function finiteNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
}

function isSupportedArena(arenaId) {
    return ARENA_IDS.includes(arenaId);
}

function isSupportedStatus(status) {
    return Object.values(MATCH_STATUS).includes(status);
}

/** Validates a match snapshot envelope. Returns { ok, errors }. */
export function validateMatchSnapshot(envelope) {
    const errors = [];
    if (envelope?.type !== EVOLUTION_EVENT_TYPES.MATCH_SNAPSHOT) {
        errors.push("type must be match_snapshot");
    }
    for (const key of ["runId", "generationId", "matchId", "arenaId"]) {
        if (typeof envelope?.[key] !== "string" || envelope[key].length === 0) {
            errors.push(`${key} must be a non-empty string`);
        }
    }
    if (!isSupportedArena(envelope?.arenaId)) {
        errors.push(`arenaId must be one of ${ARENA_IDS.join(", ")}`);
    }
    if (!Number.isInteger(envelope?.sequence) || envelope.sequence < 1) {
        errors.push("sequence must be a positive integer");
    }
    if (!Number.isInteger(envelope?.step) || envelope.step < 0) {
        errors.push("step must be a non-negative integer");
    }
    if (!Number.isInteger(envelope?.elapsedSteps) || envelope.elapsedSteps < 0) {
        errors.push("elapsedSteps must be a non-negative integer");
    }
    if (!finiteNumber(envelope?.stateTimestamp)) {
        errors.push("stateTimestamp must be a finite number");
    }
    if (!isSupportedStatus(envelope?.status)) {
        errors.push(`status must be one of ${Object.values(MATCH_STATUS).join(", ")}`);
    }
    for (const side of ["agentA", "agentB"]) {
        const participant = envelope?.[side];
        if (typeof participant?.id !== "string" || participant.id.length === 0) {
            errors.push(`${side}.id must be a non-empty string`);
        }
        if (!Number.isInteger(participant?.generation)) {
            errors.push(`${side}.generation must be an integer`);
        }
        if (!finiteNumber(participant?.paddleX) || !finiteNumber(participant?.paddleY)) {
            errors.push(`${side} paddle coordinates must be finite`);
        }
        if (!finiteNumber(participant?.epsilon)) {
            errors.push(`${side}.epsilon must be a finite number`);
        }
    }
    const ball = presentObjectFields(envelope?.ball);
    for (const key of ["x", "y", "vx", "vy"]) {
        if (!finiteNumber(ball[key])) {
            errors.push(`ball.${key} must be a finite number`);
        }
    }
    return { ok: errors.length === 0, errors };
}

/** Validates a population event envelope. Returns { ok, errors }. */
export function validatePopulationEvent(envelope) {
    const errors = [];
    if (envelope?.type !== EVOLUTION_EVENT_TYPES.POPULATION) {
        errors.push("type must be population");
    }
    if (typeof envelope?.runId !== "string") {
        errors.push("runId must be a string");
    }
    if (!Array.isArray(envelope?.agents) || envelope.agents.length !== AGENT_COUNT) {
        errors.push(`agents must be an array of exactly ${AGENT_COUNT} members`);
    }
    return { ok: errors.length === 0, errors };
}

/**
 * True when an incoming snapshot belongs to the same match session as the
 * current one and is not newer (stale or out-of-order). New sessions always
 * reset the sequence baseline so sequence restarts must not be rejected.
 */
export function isStaleMatchSnapshot(currentArena, incoming) {
    if (!currentArena || !incoming) {
        return false;
    }
    if (currentArena.matchId !== incoming.matchId) {
        return false;
    }
    return incoming.sequence <= currentArena.sequence;
}

/** Five letterboxed 4:3 courts; gameplay coordinates are never distorted. */
export function layoutArenas(width, height, arenaCount = ARENA_COUNT) {
    const padding = 10;
    const columns = 3;
    const cellGap = 8;
    const topRowCount = Math.min(columns, arenaCount);
    const bottomRowCount = Math.max(0, arenaCount - topRowCount);
    const headerHeight = 36;
    const widthLimited = (width - padding * 2 - cellGap * (columns - 1)) / columns;
    const heightLimited = ((height - padding * 2 - headerHeight - cellGap) / 2) * 4 / 3;
    const cellWidth = Math.max(1, Math.min(widthLimited, heightLimited));
    const rowHeight = cellWidth * 3 / 4;
    const topWidth = topRowCount * cellWidth + (topRowCount - 1) * cellGap;
    const bottomWidth = bottomRowCount * cellWidth + Math.max(0, bottomRowCount - 1) * cellGap;
    const topX = (width - topWidth) / 2;
    const bottomX = (width - bottomWidth) / 2;
    const contentHeight = rowHeight * 2 + cellGap;
    const topY = headerHeight + (height - headerHeight - contentHeight) / 2;
    const arenas = [];
    for (let index = 0; index < arenaCount; index += 1) {
        const isTopRow = index < topRowCount;
        const rowIndex = isTopRow ? 0 : 1;
        const columnIndex = isTopRow ? index : index - topRowCount;
        arenas.push({
            index,
            arenaId: ARENA_IDS[index],
            x: (isTopRow ? topX : bottomX) + columnIndex * (cellWidth + cellGap),
            y: topY + rowIndex * (rowHeight + cellGap),
            width: Math.round(cellWidth),
            height: Math.round(rowHeight),
        });
    }
    return arenas;
}
