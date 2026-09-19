import { ACTIONS } from "./gameState";

/**
 * Frozen champion steering. Pure decision helper; intentionally has no
 * training/replay side effects so HUMAN_VS_CHAMPION never changes champion
 * state or training service state.
 */

export const CHAMPION_REACTION_BAND = 20;

export function championDecision(ball, championPaddle, reactionBand = CHAMPION_REACTION_BAND) {
    if (ball.y < championPaddle.y - reactionBand) {
        return ACTIONS.UP;
    }
    if (ball.y > championPaddle.y + reactionBand) {
        return ACTIONS.DOWN;
    }
    return ACTIONS.STAY;
}