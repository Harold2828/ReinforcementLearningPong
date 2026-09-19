export const PHYSICS_HZ = 60;

function lerp(start, end, progress) {
    return start + (end - start) * progress;
}

function interpolateSide(start, end, progress) {
    return {
        ...end,
        paddleX: lerp(start.paddleX, end.paddleX, progress),
        paddleY: lerp(start.paddleY, end.paddleY, progress),
    };
}

export function interpolateSnapshot(start, end, progress) {
    const bounded = Math.max(0, Math.min(1, progress));
    return {
        ...end,
        agentA: interpolateSide(start.agentA, end.agentA, bounded),
        agentB: interpolateSide(start.agentB, end.agentB, bounded),
        ball: {
            ...end.ball,
            x: lerp(start.ball.x, end.ball.x, bounded),
            y: lerp(start.ball.y, end.ball.y, bounded),
        },
    };
}

export class SnapshotPlayback {
    constructor() {
        this.start = null;
        this.target = null;
        this.rendered = null;
        this.elapsedMs = 0;
        this.durationMs = 0;
    }

    push(snapshot, speed = 1) {
        if (!this.target || this.target.matchId !== snapshot.matchId) {
            this.start = snapshot;
            this.target = snapshot;
            this.rendered = snapshot;
            this.elapsedMs = 0;
            this.durationMs = 0;
            return;
        }
        const sequenceDelta = Math.max(1, snapshot.sequence - this.target.sequence);
        this.start = this.rendered ?? this.target;
        this.target = snapshot;
        this.elapsedMs = 0;
        this.durationMs = sequenceDelta * 1000 / PHYSICS_HZ / speed;
    }

    sample(deltaMs) {
        if (!this.target || this.durationMs === 0) return this.target;
        this.elapsedMs = Math.min(this.durationMs, this.elapsedMs + deltaMs);
        this.rendered = interpolateSnapshot(
            this.start,
            this.target,
            this.elapsedMs / this.durationMs,
        );
        return this.rendered;
    }
}

export class PlaybackMetrics {
    constructor(arenaCount) {
        this.arenaCount = arenaCount;
        this.reset();
    }

    reset() {
        this.elapsedMs = 0;
        this.frames = 0;
        this.snapshots = 0;
        this.sequenceRanges = new Map();
    }

    recordSnapshot(snapshot) {
        this.snapshots += 1;
        const range = this.sequenceRanges.get(snapshot.arenaId);
        if (!range || range.matchId !== snapshot.matchId) {
            this.sequenceRanges.set(snapshot.arenaId, {
                matchId: snapshot.matchId,
                first: snapshot.sequence,
                last: snapshot.sequence,
            });
        } else {
            range.last = snapshot.sequence;
        }
    }

    recordFrame(deltaMs) {
        this.frames += 1;
        this.elapsedMs += deltaMs;
        if (this.elapsedMs + 1e-6 < 1000) return null;
        const seconds = this.elapsedMs / 1000;
        const ranges = [...this.sequenceRanges.values()];
        const stepDelta = ranges.reduce((total, range) => total + range.last - range.first, 0);
        const metrics = {
            fps: this.frames / seconds,
            snapshotsPerSecond: this.snapshots / this.arenaCount / seconds,
            simulationStepsPerSecond: ranges.length ? stepDelta / ranges.length / seconds : 0,
        };
        this.reset();
        return metrics;
    }
}
