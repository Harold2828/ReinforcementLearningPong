import { describe, expect, it, vi } from "vitest";
import {
    ACTIONS,
    applyAgentActionToPaddle,
    buildQlearningState,
    normalizeAgentAction,
} from "../utils/gameState";

describe("gameState helpers", () => {
    it("normalizes valid backend actions", () => {
        expect(normalizeAgentAction({ action: "UP" })).toBe(ACTIONS.UP);
        expect(normalizeAgentAction({ direction: "down" })).toBe(ACTIONS.DOWN);
        expect(normalizeAgentAction({ action: "sideways" })).toBeNull();
    });

    it("applies valid actions and ignores invalid actions safely", () => {
        const paddle = { setVelocityY: vi.fn() };

        expect(applyAgentActionToPaddle(paddle, ACTIONS.UP)).toBe(true);
        expect(paddle.setVelocityY).toHaveBeenCalledWith(-500);

        expect(applyAgentActionToPaddle(paddle, "JUMP")).toBe(false);
    });

    it("builds the required Q-learning state payload", () => {
        const state = buildQlearningState({
            width: 800,
            height: 600,
            ball: { x: 400, y: 300, body: { velocity: { x: -120, y: 30 } } },
            agentPaddle: { y: 295 },
            opponentPaddle: { y: 310 },
            scoreAgent: 2,
            scoreOpponent: 1,
            done: false,
            agentHitBall: true,
            agentMissedBall: false,
            agentScored: false,
            previousDistanceToBall: 10,
        });

        expect(state).toMatchObject({
            ballX: 400,
            ballY: 300,
            ballVelocityX: -120,
            ballVelocityY: 30,
            paddleY: 295,
            opponentPaddleY: 310,
            scoreAgent: 2,
            scoreOpponent: 1,
            done: false,
        });
    });
});
