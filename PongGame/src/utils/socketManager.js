import { io } from "socket.io-client";

class SocketManager {
    constructor(serverUrl) {
        this.socket = io(serverUrl, {
            transports: ["websocket", "polling"],
            reconnection: true,
        });
    }

    onConnectionChange(callback) {
        this.socket.on("connect", () => callback(true));
        this.socket.on("disconnect", () => callback(false));
        this.socket.on("connect_error", () => callback(false));
    }

    onAiMove(callback) {
        this.socket.on("ai_move", callback);
    }

    onTrainingStatus(callback) {
        this.socket.on("training_status", callback);
    }

    onStateError(callback) {
        this.socket.on("state_error", callback);
    }

    onEvolutionEvent(callback) {
        this.socket.on("evolution_event", callback);
    }

    startEvolutionRun(payload = {}) {
        if (!this.socket?.connected) {
            return Promise.resolve({ status: "disconnected" });
        }
        return this.socket.timeout(10_000).emitWithAck("evolution_start_run", payload);
    }

    stopEvolutionRun() {
        if (!this.socket?.connected) {
            return Promise.resolve({ status: "disconnected" });
        }
        return this.socket.timeout(10_000).emitWithAck("evolution_stop_run");
    }

    setEvolutionPlaybackSpeed(playbackSpeed) {
        if (!this.socket?.connected) {
            return Promise.resolve({ status: "disconnected" });
        }
        return this.socket.timeout(10_000).emitWithAck(
            "evolution_set_playback_speed",
            { playbackSpeed },
        );
    }

    sendStateUpdate(environmentState) {
        if (this.socket?.connected) {
            this.socket.emit("state_update", environmentState);
            return true;
        }
        return false;
    }

    startTraining() {
        this.socket?.emit("start_training");
    }

    stopTraining() {
        this.socket?.emit("stop_training");
    }

    resetEpisode() {
        this.socket?.emit("reset_episode");
    }

    disconnect() {
        this.socket?.disconnect();
    }
}

export default SocketManager;
