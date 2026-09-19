export const PONG_WORLD = Object.freeze({ width: 800, height: 600 });
export const PONG_PHYSICS = Object.freeze({
    paddleSpeed: 500,
    initialBallVelocity: 200,
    fixedStepSeconds: 1 / 60,
    ballBody: Object.freeze({ width: 48.1, height: 51.9 }),
    paddleBody: Object.freeze({ width: 50, height: 88 }),
    serveX: Object.freeze({ mean: 175, deviation: 40 }),
    serveY: Object.freeze({ mean: 170, deviation: 25 }),
    bounceX: Object.freeze({ mean: 1.5, deviation: 0.4 }),
    bounceY: Object.freeze({ mean: 3.5, deviation: 1.1 }),
});

export function preloadPongAssets(scene) {
    scene.load.image("court", "assets/background/court.png");
    scene.load.image("ball", "assets/background/ball.png");
    scene.load.atlas("rackets", "assets/player/rackets.png", "assets/player/rackets.json");
    scene.load.audio("smash", "assets/background/smash.wav");
    scene.load.audio("point", "assets/background/point.mp3");
}

export function createOriginalCourtImage(scene, bounds = { x: 0, y: 0, width: 800, height: 600 }) {
    const scale = bounds.width / PONG_WORLD.width;
    return scene.add.image(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2, "court")
        .setOrigin(0.5, 0.5)
        .setScale(scale, scale * 0.6);
}

export function createSnapshotCourt(scene, bounds) {
    const scale = bounds.width / PONG_WORLD.width;
    const court = createOriginalCourtImage(scene, bounds);
    const frames = scene.textures.get("rackets").getFrameNames();
    const paddleA = scene.add.image(0, 0, "rackets", frames[0]).setScale(0.2 * scale);
    const paddleB = scene.add.image(0, 0, "rackets", frames[1] ?? frames[0]).setScale(0.2 * scale);
    const ball = scene.add.image(0, 0, "ball").setScale(0.1 * scale);
    const scoreStyle = {
        fontSize: `${Math.max(4, 10 * scale)}px`,
        fontFamily: "'Press Start 2P', 'Courier New', monospace",
        fill: "black",
    };
    const scoreA = scene.add.text(0, 0, "Agent: 0", scoreStyle);
    const scoreB = scene.add.text(0, 0, "Opponent: 0", scoreStyle);
    const combo = scene.add.text(0, 0, "Combo smash: 0", {
        ...scoreStyle,
        fill: "green",
        backgroundColor: "black",
    });
    const maskShape = scene.make.graphics({ add: false });
    maskShape.fillRect(bounds.x, bounds.y, bounds.width, bounds.height);
    const mask = maskShape.createGeometryMask();
    for (const object of [court, paddleA, paddleB, ball, scoreA, scoreB, combo]) {
        object.setMask(mask);
    }
    const view = {
        bounds,
        scale,
        court,
        paddleA,
        paddleB,
        ball,
        scoreA,
        scoreB,
        combo,
        smashSound: scene.sound.add("smash"),
        pointSound: scene.sound.add("point"),
        previous: null,
    };
    applySnapshotToCourt(view, {
        matchId: null,
        status: "waiting",
        combo: 0,
        agentA: { id: "agent", paddleX: 0.875, paddleY: 0.5, score: 0, returns: 0 },
        agentB: { id: "opponent", paddleX: 0.125, paddleY: 0.5, score: 0, returns: 0 },
        ball: { x: 0.5, y: 0.5, vx: 2, vy: 2 },
    });
    return view;
}

export function applySnapshotToCourt(view, snapshot) {
    const { bounds, scale } = view;
    const x = (value) => bounds.x + value * bounds.width;
    const y = (value) => bounds.y + value * bounds.height;
    view.paddleA.setPosition(x(snapshot.agentA.paddleX), y(snapshot.agentA.paddleY));
    view.paddleB.setPosition(x(snapshot.agentB.paddleX), y(snapshot.agentB.paddleY));
    view.ball.setPosition(x(snapshot.ball.x), y(snapshot.ball.y));
    view.scoreA.setPosition(bounds.x + 100 * scale, bounds.y + 20 * scale)
        .setText(`Agent: ${snapshot.agentA.score}`);
    view.scoreB.setPosition(bounds.x + 600 * scale, bounds.y + 20 * scale)
        .setText(`Opponent: ${snapshot.agentB.score}`);
    view.combo.setPosition(bounds.x + 300 * scale, bounds.y + 20 * scale)
        .setText(`Combo smash: ${snapshot.combo ?? 0}`);

    if (view.previous) {
        const returnsBefore = view.previous.agentA.returns + view.previous.agentB.returns;
        const returnsNow = snapshot.agentA.returns + snapshot.agentB.returns;
        if (returnsNow > returnsBefore) {
            view.ball.setAngle(Math.atan2(snapshot.ball.vy, snapshot.ball.vx) * 180 / Math.PI);
            view.smashSound.play();
        }
        const scoreBefore = view.previous.agentA.score + view.previous.agentB.score;
        const scoreNow = snapshot.agentA.score + snapshot.agentB.score;
        if (scoreNow > scoreBefore) view.pointSound.play();
    } else {
        view.ball.setAngle(0);
    }
    view.previous = snapshot;
}
