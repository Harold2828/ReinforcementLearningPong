import { describe, expect, it } from "vitest";
import { ACTIONS } from "../utils/gameState";
import { championDecision } from "../utils/championController";

describe("champion controller", () => {
    it("moves up when the ball is above the reaction band", () => {
        expect(championDecision({ y: 100 }, { y: 300 })).toBe(ACTIONS.UP);
    });

    it("moves down when the ball is below the reaction band", () => {
        expect(championDecision({ y: 450 }, { y: 300 })).toBe(ACTIONS.DOWN);
    });

    it("stays when the ball is within the reaction band", () => {
        expect(championDecision({ y: 306 }, { y: 300 }, 20)).toBe(ACTIONS.STAY);
        expect(championDecision({ y: 285 }, { y: 300 }, 20)).toBe(ACTIONS.STAY);
    });
});