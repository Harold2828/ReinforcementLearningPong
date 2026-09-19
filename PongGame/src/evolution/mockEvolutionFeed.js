import {
    AGENT_COUNT,
    AGENTS_PER_ARENA,
    ARENA_COUNT,
    EVOLUTION_EVENT_TYPES,
    MATCH_STATUS,
    SOURCE_LABEL,
} from "./evolutionContract";

/**
 * Deterministic mock for SPEC-03 visualization. Backend snapshots (SPEC-05/06)
 * do not exist yet, so this feed emits contract-conforming envelopes with a
 * clearly labeled MOCK source. Movement advances on a timer, independent of
 * the Phaser render loop, matching the "frame rate must not drive simulation"
 * rendering rule.
 */

const ALLOWED_WIDTHS = [32, 64, 128, 256];
const BALL_SPEED_FACTOR = 0.006;

function mulberry32(seed) {
    let state = seed >>> 0;
    return function nextRandom() {
        state = (state + 0x6d2b79f5) >>> 0;
        let value = state;
        value = Math.imul(value ^ (value >>> 15), value | 1);
        value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
        return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
}

function randomArchitecture(rng) {
    const layerCount = 1 + Math.floor(rng() * 4);
    const widths = [];
    for (let index = 0; index < layerCount; index += 1) {
        widths.push(ALLOWED_WIDTHS[Math.floor(rng() * ALLOWED_WIDTHS.length)]);
    }
    return { layerCount, widths };
}

/** Generates the 10-member population for the mock run. */
export function createMockPopulation(seed) {
    const rng = mulberry32(seed);
    const agents = [];
    for (let index = 0; index < AGENT_COUNT; index += 1) {
        const role = index < 2 ? "elite" : "offspring";
        agents.push({
            agentId: `agent-${index}`,
            generation: 0,
            architecture: randomArchitecture(rng),
            fitness: {
                winRate: Math.round(rng() * 100) / 100,
                pointDifferential: Math.round(rng() * 100) / 100,
                combo: Math.round(rng() * 100) / 100,
            },
            lineage: index < 2 ? [] : [index - 2, index - 1],
            role,
            status: "available",
        });
    }
    return agents;
}

/** Pairs every agent into the configured arenas, each appearing once. */
export function createMockArenaAssignments() {
    const assignments = [];
    for (let arenaIndex = 0; arenaIndex < ARENA_COUNT; arenaIndex += 1) {
        const offset = arenaIndex * AGENTS_PER_ARENA;
        assignments.push({
            arenaId: `arena-${arenaIndex}`,
            agentAId: `agent-${offset}`,
            agentBId: `agent-${offset + 1}`,
        });
    }
    return assignments;
}

/**
 * Advances one match session by a single tick. Pure: returns a new state
 * object. Positions and epsilon are normalized to the arena court.
 */
export function simulateMatchStep(state, rng) {
    const next = {
        ...state,
        ball: { ...state.ball },
        agentA: { ...state.agentA },
        agentB: { ...state.agentB },
        steps: state.steps + 1,
        epsilonA: Math.max(0.05, state.agentA.epsilon * 0.999),
        epsilonB: Math.max(0.05, state.agentB.epsilon * 0.999),
    };

    next.agentA.epsilon = next.epsilonA;
    next.agentB.epsilon = next.epsilonB;

    next.ball.x += next.ball.vx * BALL_SPEED_FACTOR;
    next.ball.y += next.ball.vy * BALL_SPEED_FACTOR;
    if (next.ball.y <= 0 || next.ball.y >= 1) {
        next.ball.vy *= -1;
    }

    const followA = next.ball.vx < 0 ? next.ball.y : 0.5;
    const followB = next.ball.vx > 0 ? next.ball.y : 0.5;
    next.agentA.paddleY = next.agentA.paddleY + Math.sign(followA - next.agentA.paddleY) * 0.02;
    next.agentB.paddleY = next.agentB.paddleY + Math.sign(followB - next.agentB.paddleY) * 0.02;

    if (next.ball.x <= 0) {
        next.agentB.score += 1;
        resetBall(next, rng);
    } else if (next.ball.x >= 1) {
        next.agentA.score += 1;
        resetBall(next, rng);
    }

    if (next.steps % 400 === 0) {
        next.status = MATCH_STATUS.TERMINAL;
    }
    return next;
}

function resetBall(state, rng) {
    state.ball.x = 0.5;
    state.ball.y = 0.5;
    state.ball.vx = rng() < 0.5 ? -1 : 1;
    state.ball.vy = (rng() - 0.5) * 2;
}

function freshMatchState(rng) {
    return {
        ball: { x: 0.5, y: 0.5, vx: rng() < 0.5 ? -1 : 1, vy: (rng() - 0.5) * 2 },
        agentA: { paddleY: 0.5, score: 0, epsilon: 1.0 },
        agentB: { paddleY: 0.5, score: 0, epsilon: 1.0 },
        steps: 0,
        status: MATCH_STATUS.RUNNING,
    };
}

class MockEvolutionFeed {
    constructor({ seed = 42, tickMs = 100, errorOnStart = false } = {}) {
        this.seed = seed;
        this.tickMs = tickMs;
        this.errorOnStart = errorOnStart;
        this.population = createMockPopulation(seed);
        this.assignments = createMockArenaAssignments();
        this.listeners = [];
        this.timer = null;
        this.runId = "mock-run-1";
        this.generationId = "mock-generation-1";
        this.stateTimestamp = Date.now();
        this.elapsedSteps = 0;
        this.sessionCounters = this.assignments.map(() => 1);
        this.sessions = this.assignments.map((assignment, index) => ({
            assignment,
            matchId: `mock-match-${index + 1}-session-${this.sessionCounters[index]}`,
            sequence: 0,
            matchState: freshMatchState(mulberry32(this.seed + index * 97)),
        }));
        this.epoch = this.seed;
    }

    subscribe(listener) {
        this.listeners.push(listener);
        return () => {
            this.listeners = this.listeners.filter((entry) => entry !== listener);
        };
    }

    start() {
        this.emitInitialEvents();
        this.stop();
        this.timer = setInterval(() => this.tick(), this.tickMs);
    }

    stop() {
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = null;
        }
    }

    emitInitialEvents() {
        this.publish({ type: EVOLUTION_EVENT_TYPES.POPULATION, runId: this.runId, generationId: this.generationId, agents: this.population, source: SOURCE_LABEL.MOCK });
        this.publish({ type: EVOLUTION_EVENT_TYPES.CHAMPION_PROMOTION, runId: this.runId, agentId: "agent-0", checkpointHash: "mock-checkpoint-sha256-00000000000000000000", evidence: { benchmark: "mock-bench-v1", fitness: this.population[0].fitness }, source: SOURCE_LABEL.MOCK });
        if (this.errorOnStart) {
            this.publish({ type: EVOLUTION_EVENT_TYPES.CONTROLLED_ERROR, runId: this.runId, code: "MOCK_NOT_IMPLEMENTED", message: "Authoritative snapshots pending (SPEC-05/06).", source: SOURCE_LABEL.MOCK });
        }
    }

    tick() {
        this.stateTimestamp = Date.now();
        this.elapsedSteps += 1;
        const epochRng = mulberry32(this.epoch + this.elapsedSteps);
        const events = [];
        this.sessions.forEach((session, index) => {
            session.matchState = simulateMatchStep(session.matchState, epochRng);
            session.sequence += 1;
            events.push(this.toSnapshot(session, index));

            if (session.matchState.status === MATCH_STATUS.TERMINAL) {
                this.sessionCounters[index] += 1;
                session.matchId = `mock-match-${index + 1}-session-${this.sessionCounters[index]}`;
                session.sequence = 0;
                session.matchState = freshMatchState(mulberry32(this.seed + index * 97));
            }
        });
        if (this.elapsedSteps % 20 === 0) {
            events.push({
                type: EVOLUTION_EVENT_TYPES.EVALUATION,
                runId: this.runId,
                generationId: this.generationId,
                benchmark: "mock-bench-v1",
                results: this.population.map((agent) => ({ agentId: agent.agentId, fitness: agent.fitness })),
                source: SOURCE_LABEL.MOCK,
            });
        }
        events.forEach((event) => this.publish(event));
    }

    toSnapshot(session, index) {
        const { assignment, matchId, sequence, matchState } = session;
        const agentA = this.population.find((agent) => agent.agentId === assignment.agentAId);
        const agentB = this.population.find((agent) => agent.agentId === assignment.agentBId);
        return {
            type: EVOLUTION_EVENT_TYPES.MATCH_SNAPSHOT,
            runId: this.runId,
            generationId: this.generationId,
            matchId,
            arenaId: assignment.arenaId,
            sequence,
            stateTimestamp: this.stateTimestamp,
            step: matchState.steps,
            elapsedSteps: this.elapsedSteps,
            status: matchState.status,
            agentA: {
                id: agentA.agentId,
                generation: agentA.generation,
                paddleX: 0.06,
                paddleY: matchState.agentA.paddleY,
                epsilon: matchState.agentA.epsilon,
                score: matchState.agentA.score,
            },
            agentB: {
                id: agentB.agentId,
                generation: agentB.generation,
                paddleX: 0.94,
                paddleY: matchState.agentB.paddleY,
                epsilon: matchState.agentB.epsilon,
                score: matchState.agentB.score,
            },
            ball: matchState.ball,
            source: SOURCE_LABEL.MOCK,
        };
    }

    publish(event) {
        this.listeners.forEach((listener) => listener(event));
    }
}

export { MockEvolutionFeed, mulberry32 };
