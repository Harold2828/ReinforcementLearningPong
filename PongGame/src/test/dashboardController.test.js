/**
 * @vitest-environment jsdom
 */
import { describe, expect, it, vi } from "vitest";
import { DashboardController } from "../utils/dashboardController";

function renderDashboard() {
    document.body.innerHTML = `
        <strong data-ui="connection-status"></strong>
        <strong data-ui="mode"></strong>
        <select data-action="select-game-mode"><option value="HUMAN_VS_AI">Human vs AI</option><option value="AI_VS_AI">AI vs AI</option></select>
        <strong data-ui="episode"></strong>
        <strong data-ui="reward"></strong>
        <strong data-ui="opponent-reward"></strong>
        <strong data-ui="epsilon"></strong>
        <strong data-ui="opponent-epsilon"></strong>
        <strong data-ui="score"></strong>
        <strong data-ui="ownership"></strong>
        <strong data-ui="learning-status"></strong>
        <strong data-ui="agent-metrics"></strong>
        <strong data-ui="opponent-metrics"></strong>
        <strong data-ui="self-play-episodes"></strong>
        <strong data-ui="feedback"></strong>
        <button data-action="start-training"></button>
        <button data-action="stop-training"></button>
        <button data-action="reset-episode"></button>
    `;
}

describe("DashboardController", () => {
    it("renders dashboard state", () => {
        renderDashboard();
        const controller = new DashboardController(document);

        controller.update({
            connected: true,
            mode: "Training",
            episode: 3,
            reward: 1.234,
            opponentReward: -0.5,
            epsilon: 0.5,
            opponentEpsilon: 0.4,
            scoreAgent: 2,
            scoreOpponent: 1,
            agentOwner: "AI Agent",
            opponentOwner: "AI Opponent",
            learningEnabled: true,
            agentWinRate: 0.75,
            opponentWinRate: 0.25,
            agentAverageReward: 2.2,
            opponentAverageReward: -1.2,
            agentHitRate: 0.8,
            opponentHitRate: 0.4,
            selfPlayEpisodes: 8,
            feedback: "Agent hit the ball",
        });

        expect(document.querySelector("[data-ui='connection-status']").textContent).toBe("Connected");
        expect(document.querySelector("[data-ui='mode']").textContent).toBe("Training");
        expect(document.querySelector("[data-ui='episode']").textContent).toBe("3");
        expect(document.querySelector("[data-ui='reward']").textContent).toBe("1.23");
        expect(document.querySelector("[data-ui='opponent-reward']").textContent).toBe("-0.50");
        expect(document.querySelector("[data-ui='epsilon']").textContent).toBe("0.500");
        expect(document.querySelector("[data-ui='opponent-epsilon']").textContent).toBe("0.400");
        expect(document.querySelector("[data-ui='score']").textContent).toBe("2 - 1");
        expect(document.querySelector("[data-ui='ownership']").textContent).toBe("Left: AI Opponent | Right: AI Agent");
        expect(document.querySelector("[data-ui='learning-status']").textContent).toBe("Learning enabled");
        expect(document.querySelector("[data-ui='agent-metrics']").textContent).toBe("Win 75% | Hit 80% | Avg 2.20");
        expect(document.querySelector("[data-ui='opponent-metrics']").textContent).toBe("Win 25% | Hit 40% | Avg -1.20");
        expect(document.querySelector("[data-ui='self-play-episodes']").textContent).toBe("8");
        expect(document.querySelector("[data-ui='feedback']").textContent).toBe("Agent hit the ball");
    });

    it("wires start stop and reset controls", () => {
        renderDashboard();
        const controller = new DashboardController(document);
        const onStartTraining = vi.fn();
        const onStopTraining = vi.fn();
        const onResetEpisode = vi.fn();
        const onGameModeChange = vi.fn();

        controller.bindControls({ onStartTraining, onStopTraining, onResetEpisode, onGameModeChange });
        document.querySelector("[data-action='start-training']").click();
        document.querySelector("[data-action='stop-training']").click();
        document.querySelector("[data-action='reset-episode']").click();
        document.querySelector("[data-action='select-game-mode']").value = "AI_VS_AI";
        document.querySelector("[data-action='select-game-mode']").dispatchEvent(new Event("change"));

        expect(onStartTraining).toHaveBeenCalledOnce();
        expect(onStopTraining).toHaveBeenCalledOnce();
        expect(onResetEpisode).toHaveBeenCalledOnce();
        expect(onGameModeChange).toHaveBeenCalledWith("AI_VS_AI");
    });

    it("prevents duplicate training starts while active", () => {
        renderDashboard();
        const controller = new DashboardController(document);

        controller.setTrainingActive(true);

        expect(document.querySelector("[data-action='start-training']").disabled).toBe(true);
        expect(document.querySelector("[data-action='stop-training']").disabled).toBe(false);
    });

    it("keeps previous connection state across partial updates", () => {
        renderDashboard();
        const controller = new DashboardController(document);

        controller.update({ connected: true, mode: "Training" });
        controller.update({ reward: 2 });

        expect(document.querySelector("[data-ui='connection-status']").textContent).toBe("Connected");
        expect(document.querySelector("[data-ui='reward']").textContent).toBe("2.00");
    });
});
