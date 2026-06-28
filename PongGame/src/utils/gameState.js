export const ACTIONS = Object.freeze({
    UP: "UP",
    DOWN: "DOWN",
    STAY: "STAY",
});

export function isValidAction(actionName) {
    return Object.values(ACTIONS).includes(actionName);
}

export function normalizeAgentAction(responsePayload) {
    const actionName = responsePayload?.action ?? responsePayload?.direction;
    if (typeof actionName !== "string") {
        return null;
    }

    const normalizedActionName = actionName.toUpperCase();
    return isValidAction(normalizedActionName) ? normalizedActionName : null;
}

export function applyAgentActionToPaddle(paddle, actionName, speed = 500) {
    if (!isValidAction(actionName)) {
        console.warn(`Invalid agent action ignored: ${actionName}`);
        return false;
    }

    if (actionName === ACTIONS.UP) {
        paddle.setVelocityY(-speed);
    } else if (actionName === ACTIONS.DOWN) {
        paddle.setVelocityY(speed);
    } else {
        paddle.setVelocityY(0);
    }

    return true;
}

export function buildQlearningState({
    width,
    height,
    ball,
    agentPaddle,
    opponentPaddle,
    scoreAgent,
    scoreOpponent,
    done,
    agentHitBall,
    agentMissedBall,
    agentScored,
    previousDistanceToBall,
}) {
    return {
        ballX: ball.x,
        ballY: ball.y,
        ballVelocityX: ball.body?.velocity?.x ?? 0,
        ballVelocityY: ball.body?.velocity?.y ?? 0,
        paddleY: agentPaddle.y,
        opponentPaddleY: opponentPaddle?.y,
        scoreAgent,
        scoreOpponent,
        done,
        agentHitBall,
        agentMissedBall,
        agentScored,
        previousDistanceToBall,
        width,
        height,
    };
}
