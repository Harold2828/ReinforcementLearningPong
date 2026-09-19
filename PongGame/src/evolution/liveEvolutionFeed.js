import {
    EVOLUTION_EVENT_TYPES,
    SOURCE_LABEL,
    validateMatchSnapshot,
    validatePopulationEvent,
} from "./evolutionContract";

/**
 * Bridge between the SPEC-06 Socket.IO feed and the SPEC-03 views.
 *
 * Receives `evolution_event` payloads from the companion backend, runs the
 * contract validators (dropping structurally invalid envelopes), and forwards
 * conforming events to SPEC-03 subscribers tagged with the LIVE source label.
 *
 * While the socket is disconnected it falls back to a locally driven mock feed
 * so the evolution views still render (clearly labeled) offline.
 */

class LiveEvolutionFeed {
    constructor({ sensor, fallback = null } = {}) {
        this.sensor = sensor;
        this.fallback = fallback;
        this.connected = false;
        this.listeners = [];
        this.unregister = null;
        this.fallbackUnsubscribe = null;
        this.runRequested = false;
        this.runStarting = false;
        this.runPayload = {};
    }

    subscribe(listener) {
        this.listeners.push(listener);
        return () => {
            this.listeners = this.listeners.filter((entry) => entry !== listener);
        };
    }

    start() {
        if (this.fallback) {
            this.fallback.start();
            this.fallbackUnsubscribe = this.fallback.subscribe((event) => {
                if (!this.connected) {
                    this.publish(event);
                }
            });
        }
        if (this.sensor?.onEvolutionEvent) {
            this.sensor.onEvolutionEvent((payload) => {
                if (this.connected) {
                    this.publish(this.validate(payload));
                }
            });
        }
        if (this.sensor?.onConnectionChange) {
            this.unregister = this.sensor.onConnectionChange((isConnected) => {
                this.connected = isConnected;
                if (isConnected) this.startRequestedRun();
            });
        }
    }

    startRun(payload = {}) {
        this.runRequested = true;
        this.runPayload = payload;
        return this.startRequestedRun();
    }

    async startRequestedRun() {
        if (!this.connected || this.runStarting || !this.runRequested) return null;
        this.runStarting = true;
        try {
            return await this.sensor?.startEvolutionRun?.(this.runPayload);
        } finally {
            this.runStarting = false;
        }
    }

    stopRun() {
        this.runRequested = false;
        return this.sensor?.stopEvolutionRun?.() ?? Promise.resolve(null);
    }

    stop() {
        if (this.fallbackUnsubscribe) {
            this.fallbackUnsubscribe();
            this.fallbackUnsubscribe = null;
        }
        if (this.unregister) {
            this.unregister();
            this.unregister = null;
        }
        if (this.fallback) {
            this.fallback.stop();
        }
    }

    validate(payload) {
        if (payload?.type === EVOLUTION_EVENT_TYPES.MATCH_SNAPSHOT) {
            const result = validateMatchSnapshot(payload);
            if (!result.ok) {
                console.warn(`LiveEvolutionFeed: dropping invalid match_snapshot: ${result.errors.join("; ")}`);
                return null;
            }
        } else if (payload?.type === EVOLUTION_EVENT_TYPES.POPULATION) {
            const result = validatePopulationEvent(payload);
            if (!result.ok) {
                console.warn(`LiveEvolutionFeed: dropping invalid population: ${result.errors.join("; ")}`);
                return null;
            }
        }
        if (payload?.source == null) {
            return { ...payload, source: SOURCE_LABEL.LIVE };
        }
        return payload;
    }

    publish(event) {
        if (event == null) {
            return;
        }
        if (event.type === "run_finished" || event.type === "run_error") {
            this.runRequested = false;
        }
        this.listeners.forEach((listener) => listener(event));
    }
}

export { LiveEvolutionFeed };
