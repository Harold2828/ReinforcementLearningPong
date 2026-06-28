export const ACTIONS = Object.freeze({
    UP: "UP",
    DOWN: "DOWN",
    STAY: "STAY",
});

export function isValidAction(actionName) {
    return Object.values(ACTIONS).includes(actionName);
}

export function normalizeAgentAction(responsePayload) {
    const actionName = responsePayload?.agentAction ?? responsePayload?.action ?? responsePayload?.direction;
    if (typeof actionName !== "string") {
        return null;
    }

    const normalizedActionName = actionName.toUpperCase();
    return isValidAction(normalizedActionName) ? normalizedActionName : null;
}

export function normalizeOpponentAction(responsePayload) {
    const actionName = responsePayload?.opponentAction;
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
    gameMode,
    width,
    height,
    ball,
    agentPaddle,
    opponentPaddle,
    agentScore,
    opponentScore,
    lastHitBy,
    pointWinner,
    episodeId,
    previousAgentDistanceToBall,
    previousOpponentDistanceToBall,
    comboSmash = 0,
}) {
    return {
        gameMode,
        ballX: ball.x,
        ballY: ball.y,
        ballVelocityX: ball.body?.velocity?.x ?? 0,
        ballVelocityY: ball.body?.velocity?.y ?? 0,
        agentPaddleY: agentPaddle.y,
        opponentPaddleY: opponentPaddle?.y,
        agentScore,
        opponentScore,
        lastHitBy,
        pointWinner,
        episodeId,
        previousAgentDistanceToBall,
        previousOpponentDistanceToBall,
        comboSmash,
        width,
        height,
    };
}
