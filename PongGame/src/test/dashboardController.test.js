/**
 * @vitest-environment jsdom
 */
import { describe, expect, it, vi } from "vitest";
import { DashboardController } from "../utils/dashboardController";

function renderDashboard() {
    document.body.innerHTML = `
        <strong data-ui="connection-status"></strong>
        <strong data-ui="mode"></strong>
        <strong data-ui="episode"></strong>
        <strong data-ui="reward"></strong>
        <strong data-ui="epsilon"></strong>
        <strong data-ui="score"></strong>
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
            epsilon: 0.5,
            scoreAgent: 2,
            scoreOpponent: 1,
            feedback: "Agent hit the ball",
        });

        expect(document.querySelector("[data-ui='connection-status']").textContent).toBe("Connected");
        expect(document.querySelector("[data-ui='mode']").textContent).toBe("Training");
        expect(document.querySelector("[data-ui='episode']").textContent).toBe("3");
        expect(document.querySelector("[data-ui='reward']").textContent).toBe("1.23");
        expect(document.querySelector("[data-ui='epsilon']").textContent).toBe("0.500");
        expect(document.querySelector("[data-ui='score']").textContent).toBe("2 - 1");
        expect(document.querySelector("[data-ui='feedback']").textContent).toBe("Agent hit the ball");
    });

    it("wires start stop and reset controls", () => {
        renderDashboard();
        const controller = new DashboardController(document);
        const onStartTraining = vi.fn();
        const onStopTraining = vi.fn();
        const onResetEpisode = vi.fn();

        controller.bindControls({ onStartTraining, onStopTraining, onResetEpisode });
        document.querySelector("[data-action='start-training']").click();
        document.querySelector("[data-action='stop-training']").click();
        document.querySelector("[data-action='reset-episode']").click();

        expect(onStartTraining).toHaveBeenCalledOnce();
        expect(onStopTraining).toHaveBeenCalledOnce();
        expect(onResetEpisode).toHaveBeenCalledOnce();
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
