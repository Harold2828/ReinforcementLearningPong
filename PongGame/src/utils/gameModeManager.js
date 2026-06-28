export const GAME_MODES = Object.freeze({
    HUMAN_VS_AI: "HUMAN_VS_AI",
    AI_VS_HUMAN: "AI_VS_HUMAN",
    AI_VS_AI: "AI_VS_AI",
    TRAINING_SELF_PLAY: "TRAINING_SELF_PLAY",
    EVALUATION: "EVALUATION",
});

export const GAME_MODE_LABELS = Object.freeze({
    [GAME_MODES.HUMAN_VS_AI]: "Human vs AI",
    [GAME_MODES.AI_VS_HUMAN]: "AI vs Human",
    [GAME_MODES.AI_VS_AI]: "AI vs AI",
    [GAME_MODES.TRAINING_SELF_PLAY]: "Training Self-Play",
    [GAME_MODES.EVALUATION]: "Evaluation",
});

const MODE_OWNERSHIP = Object.freeze({
    [GAME_MODES.HUMAN_VS_AI]: {
        opponent: "Human",
        agent: "AI Agent",
        learningEnabled: false,
    },
    [GAME_MODES.AI_VS_HUMAN]: {
        opponent: "AI Opponent",
        agent: "Human",
        learningEnabled: false,
    },
    [GAME_MODES.AI_VS_AI]: {
        opponent: "AI Opponent",
        agent: "AI Agent",
        learningEnabled: false,
    },
    [GAME_MODES.TRAINING_SELF_PLAY]: {
        opponent: "AI Opponent",
        agent: "AI Agent",
        learningEnabled: true,
    },
    [GAME_MODES.EVALUATION]: {
        opponent: "AI Opponent",
        agent: "AI Agent",
        learningEnabled: false,
    },
});

export class GameModeManager {
    constructor(initialMode = GAME_MODES.HUMAN_VS_AI) {
        this.currentMode = this.isSupportedMode(initialMode) ? initialMode : GAME_MODES.HUMAN_VS_AI;
    }

    setMode(gameMode) {
        if (!this.isSupportedMode(gameMode)) {
            console.warn(`Unsupported game mode ignored: ${gameMode}`);
            return false;
        }

        this.currentMode = gameMode;
        return true;
    }

    isSupportedMode(gameMode) {
        return Object.values(GAME_MODES).includes(gameMode);
    }

    isAgentAiControlled() {
        return [GAME_MODES.HUMAN_VS_AI, GAME_MODES.AI_VS_AI, GAME_MODES.TRAINING_SELF_PLAY, GAME_MODES.EVALUATION]
            .includes(this.currentMode);
    }

    isOpponentAiControlled() {
        return [GAME_MODES.AI_VS_HUMAN, GAME_MODES.AI_VS_AI, GAME_MODES.TRAINING_SELF_PLAY, GAME_MODES.EVALUATION]
            .includes(this.currentMode);
    }

    isAgentHumanControlled() {
        return !this.isAgentAiControlled();
    }

    isOpponentHumanControlled() {
        return !this.isOpponentAiControlled();
    }

    isLearningEnabled() {
        return MODE_OWNERSHIP[this.currentMode].learningEnabled;
    }

    getOwnership() {
        return MODE_OWNERSHIP[this.currentMode];
    }

    getModeLabel() {
        return GAME_MODE_LABELS[this.currentMode];
    }
}
