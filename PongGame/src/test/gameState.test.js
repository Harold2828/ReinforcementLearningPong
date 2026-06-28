import { describe, expect, it, vi } from "vitest";
import {
    ACTIONS,
    applyAgentActionToPaddle,
    buildQlearningState,
    normalizeAgentAction,
    normalizeOpponentAction,
} from "../utils/gameState";
import { GAME_MODES, GameModeManager } from "../utils/gameModeManager";

describe("gameState helpers", () => {
    it("normalizes valid backend actions", () => {
        expect(normalizeAgentAction({ action: "UP" })).toBe(ACTIONS.UP);
        expect(normalizeAgentAction({ agentAction: "STAY" })).toBe(ACTIONS.STAY);
        expect(normalizeAgentAction({ direction: "down" })).toBe(ACTIONS.DOWN);
        expect(normalizeAgentAction({ action: "sideways" })).toBeNull();
        expect(normalizeOpponentAction({ opponentAction: "UP" })).toBe(ACTIONS.UP);
    });

    it("applies valid actions and ignores invalid actions safely", () => {
        const paddle = { setVelocityY: vi.fn() };

        expect(applyAgentActionToPaddle(paddle, ACTIONS.UP)).toBe(true);
        expect(paddle.setVelocityY).toHaveBeenCalledWith(-500);

        expect(applyAgentActionToPaddle(paddle, "JUMP")).toBe(false);
    });

    it("builds the required Q-learning state payload", () => {
        const state = buildQlearningState({
            gameMode: GAME_MODES.TRAINING_SELF_PLAY,
            width: 800,
            height: 600,
            ball: { x: 400, y: 300, body: { velocity: { x: -120, y: 30 } } },
            agentPaddle: { y: 295 },
            opponentPaddle: { y: 310 },
            agentScore: 2,
            opponentScore: 1,
            lastHitBy: "agent",
            pointWinner: null,
            episodeId: "episode-1",
            previousAgentDistanceToBall: 10,
            previousOpponentDistanceToBall: 12,
        });

        expect(state).toMatchObject({
            gameMode: GAME_MODES.TRAINING_SELF_PLAY,
            ballX: 400,
            ballY: 300,
            ballVelocityX: -120,
            ballVelocityY: 30,
            agentPaddleY: 295,
            opponentPaddleY: 310,
            agentScore: 2,
            opponentScore: 1,
            lastHitBy: "agent",
        });
    });

    it("centralizes human and ai ownership by game mode", () => {
        const manager = new GameModeManager(GAME_MODES.AI_VS_AI);

        expect(manager.isAgentAiControlled()).toBe(true);
        expect(manager.isOpponentAiControlled()).toBe(true);
        expect(manager.isAgentHumanControlled()).toBe(false);
        expect(manager.isOpponentHumanControlled()).toBe(false);

        manager.setMode(GAME_MODES.HUMAN_VS_AI);

        expect(manager.isOpponentHumanControlled()).toBe(true);
        expect(manager.isAgentAiControlled()).toBe(true);
    });
});
