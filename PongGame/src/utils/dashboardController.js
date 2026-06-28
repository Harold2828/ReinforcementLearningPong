const DEFAULT_DASHBOARD_STATE = {
    connected: false,
    mode: "Manual",
    episode: 1,
    reward: 0,
    epsilon: 0,
    scoreAgent: 0,
    scoreOpponent: 0,
    feedback: "Ready",
};

export class DashboardController {
    constructor(documentReference = document) {
        this.document = documentReference;
        this.state = { ...DEFAULT_DASHBOARD_STATE };
        this.elements = {
            connectionStatus: this.document.querySelector("[data-ui='connection-status']"),
            mode: this.document.querySelector("[data-ui='mode']"),
            episode: this.document.querySelector("[data-ui='episode']"),
            reward: this.document.querySelector("[data-ui='reward']"),
            epsilon: this.document.querySelector("[data-ui='epsilon']"),
            score: this.document.querySelector("[data-ui='score']"),
            feedback: this.document.querySelector("[data-ui='feedback']"),
            startButton: this.document.querySelector("[data-action='start-training']"),
            stopButton: this.document.querySelector("[data-action='stop-training']"),
            resetButton: this.document.querySelector("[data-action='reset-episode']"),
        };

        this.update(DEFAULT_DASHBOARD_STATE);
    }

    bindControls({ onStartTraining, onStopTraining, onResetEpisode }) {
        this.elements.startButton?.addEventListener("click", onStartTraining);
        this.elements.stopButton?.addEventListener("click", onStopTraining);
        this.elements.resetButton?.addEventListener("click", onResetEpisode);
    }

    update(partialState) {
        this.state = { ...this.state, ...partialState };
        const state = this.state;
        this.setText("connectionStatus", state.connected ? "Connected" : "Disconnected");
        this.elements.connectionStatus?.classList.toggle("is-connected", state.connected);
        this.setText("mode", state.mode);
        this.setText("episode", String(state.episode));
        this.setText("reward", Number(state.reward).toFixed(2));
        this.setText("epsilon", Number(state.epsilon).toFixed(3));
        this.setText("score", `${state.scoreAgent} - ${state.scoreOpponent}`);
        this.setText("feedback", state.feedback);
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
