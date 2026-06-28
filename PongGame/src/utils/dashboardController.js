const DEFAULT_DASHBOARD_STATE = {
    connected: false,
    mode: "Human vs AI",
    episode: 1,
    reward: 0,
    opponentReward: 0,
    epsilon: 0,
    opponentEpsilon: 0,
    scoreAgent: 0,
    scoreOpponent: 0,
    agentOwner: "AI Agent",
    opponentOwner: "Human",
    learningEnabled: false,
    agentWinRate: 0,
    opponentWinRate: 0,
    agentAverageReward: 0,
    opponentAverageReward: 0,
    agentHitRate: 0,
    opponentHitRate: 0,
    selfPlayEpisodes: 0,
    feedback: "Ready",
};

export class DashboardController {
    constructor(documentReference = document) {
        this.document = documentReference;
        this.state = { ...DEFAULT_DASHBOARD_STATE };
        this.elements = {
            connectionStatus: this.document.querySelector("[data-ui='connection-status']"),
            mode: this.document.querySelector("[data-ui='mode']"),
            gameModeSelect: this.document.querySelector("[data-action='select-game-mode']"),
            episode: this.document.querySelector("[data-ui='episode']"),
            reward: this.document.querySelector("[data-ui='reward']"),
            opponentReward: this.document.querySelector("[data-ui='opponent-reward']"),
            epsilon: this.document.querySelector("[data-ui='epsilon']"),
            opponentEpsilon: this.document.querySelector("[data-ui='opponent-epsilon']"),
            score: this.document.querySelector("[data-ui='score']"),
            ownership: this.document.querySelector("[data-ui='ownership']"),
            learningStatus: this.document.querySelector("[data-ui='learning-status']"),
            agentMetrics: this.document.querySelector("[data-ui='agent-metrics']"),
            opponentMetrics: this.document.querySelector("[data-ui='opponent-metrics']"),
            selfPlayEpisodes: this.document.querySelector("[data-ui='self-play-episodes']"),
            feedback: this.document.querySelector("[data-ui='feedback']"),
            startButton: this.document.querySelector("[data-action='start-training']"),
            stopButton: this.document.querySelector("[data-action='stop-training']"),
            resetButton: this.document.querySelector("[data-action='reset-episode']"),
        };

        this.update(DEFAULT_DASHBOARD_STATE);
    }

    bindControls({ onStartTraining, onStopTraining, onResetEpisode, onGameModeChange }) {
        this.elements.startButton?.addEventListener("click", onStartTraining);
        this.elements.stopButton?.addEventListener("click", onStopTraining);
        this.elements.resetButton?.addEventListener("click", onResetEpisode);
        this.elements.gameModeSelect?.addEventListener("change", (event) => onGameModeChange?.(event.target.value));
    }

    update(partialState) {
        this.state = { ...this.state, ...partialState };
        const state = this.state;
        this.setText("connectionStatus", state.connected ? "Connected" : "Disconnected");
        this.elements.connectionStatus?.classList.toggle("is-connected", state.connected);
        this.setText("mode", state.mode);
        this.setText("episode", String(state.episode));
        this.setText("reward", Number(state.reward).toFixed(2));
        this.setText("opponentReward", Number(state.opponentReward).toFixed(2));
        this.setText("epsilon", Number(state.epsilon).toFixed(3));
        this.setText("opponentEpsilon", Number(state.opponentEpsilon).toFixed(3));
        this.setText("score", `${state.scoreAgent} - ${state.scoreOpponent}`);
        this.setText("ownership", `Left: ${state.opponentOwner} | Right: ${state.agentOwner}`);
        this.setText("learningStatus", state.learningEnabled ? "Learning enabled" : "Learning disabled");
        this.setText(
            "agentMetrics",
            `Win ${(state.agentWinRate * 100).toFixed(0)}% | Hit ${(state.agentHitRate * 100).toFixed(0)}% | Avg ${Number(state.agentAverageReward).toFixed(2)}`,
        );
        this.setText(
            "opponentMetrics",
            `Win ${(state.opponentWinRate * 100).toFixed(0)}% | Hit ${(state.opponentHitRate * 100).toFixed(0)}% | Avg ${Number(state.opponentAverageReward).toFixed(2)}`,
        );
        this.setText("selfPlayEpisodes", String(state.selfPlayEpisodes));
        this.setText("feedback", state.feedback);
    }

    setSelectedMode(gameMode) {
        if (this.elements.gameModeSelect) {
            this.elements.gameModeSelect.value = gameMode;
        }
    }

    setTrainingActive(isTrainingActive) {
        if (this.elements.startButton) {
            this.elements.startButton.disabled = isTrainingActive;
        }
        if (this.elements.stopButton) {
            this.elements.stopButton.disabled = !isTrainingActive;
        }
    }

    setText(elementName, value) {
        if (this.elements[elementName]) {
            this.elements[elementName].textContent = value;
        }
    }
}
