import random

from app.ai.pong_training_env import PongTrainingEnv, PongTrainingEnvConfig
from app.evolution.simulation import (
    MatchAssignment,
    MatchConfig,
    MatchParticipant,
    MatchSession,
)


def participant(name):
    return MatchParticipant(agentId=name, generation=3)


def assignment(arena_id="arena-0", seed=11):
    return MatchAssignment(
        arenaId=arena_id,
        runId="run-1",
        generationId="gen-1",
        agentA=participant(f"agent-a-{arena_id}"),
        agentB=participant(f"agent-b-{arena_id}"),
        seed=seed,
    )


def session_for(seed=11):
    return MatchSession(assignment(arena_id=f"arena-{seed}", seed=seed), MatchConfig())


def test_edge_graze_bottom_paddle_bounces():
    """Swept hit: contact Y inside paddle span but end-of-step Y outside."""
    session = session_for()
    cfg = session.config
    plane = cfg.agentPaddleX - (cfg.paddleWidth + cfg.ballWidth) / 2
    env = session.env
    env.ballX = plane - 1.0
    env.ballY = 368.0
    env.agentPaddleY = 300
    env.ballVelocityX = cfg.ballSpeedX
    env.ballVelocityY = 240.0

    envelope = session.step("STAY", "STAY")

    assert envelope["ball"]["vx"] < 0
    assert envelope["agentA"]["returns"] == 1
    assert envelope["pointWinner"] is None
    assert session.env.ballX == plane


def test_edge_exact_touch_bounces():
    """Inclusive AABB overlap: ball edge exactly on paddle edge counts."""
    session = session_for()
    cfg = session.config
    plane = cfg.agentPaddleX - (cfg.paddleWidth + cfg.ballWidth) / 2
    half = (cfg.paddleHeight + cfg.ballHeight) / 2
    env = session.env
    env.ballX = plane - 0.01
    env.ballY = env.agentPaddleY + half
    env.ballVelocityX = cfg.ballSpeedX
    env.ballVelocityY = 0

    envelope = session.step("STAY", "STAY")

    assert envelope["ball"]["vx"] < 0
    assert envelope["agentA"]["returns"] == 1
    assert session.env.ballX == plane


def test_high_speed_ball_cannot_tunnel_right_paddle():
    session = session_for()
    cfg = session.config
    plane = cfg.agentPaddleX - (cfg.paddleWidth + cfg.ballWidth) / 2
    env = session.env
    baseCombo = env.comboSmash
    env.ballX = plane - 0.01
    env.ballY = env.agentPaddleY
    env.ballVelocityX = 50_000
    env.ballVelocityY = 0

    first = session.step("STAY", "STAY")
    afterBounceX = env.ballVelocityX
    afterBounceHit = env.lastHitBy

    assert first["pointWinner"] is None
    assert afterBounceHit == "agent"
    assert env.comboSmash == baseCombo + 1
    assert afterBounceX < 0
    assert env.ballX == plane

    env.ballVelocityX = -cfg.ballSpeedX
    second = session.step("STAY", "STAY")

    assert second["ball"]["x"] < plane / cfg.width
    assert env.lastHitBy is None
    assert env.comboSmash == baseCombo + 1


def test_high_speed_ball_cannot_tunnel_left_paddle():
    session = session_for()
    cfg = session.config
    plane = cfg.opponentPaddleX + (cfg.paddleWidth + cfg.ballWidth) / 2
    env = session.env
    env.ballX = plane + 0.01
    env.ballY = env.opponentPaddleY
    env.ballVelocityX = -50_000
    env.ballVelocityY = 0

    envelope = session.step("STAY", "STAY")

    assert envelope["pointWinner"] is None
    assert env.lastHitBy == "opponent"
    assert env.ballX == plane
    assert env.ballVelocityX > 0


def test_genuine_miss_does_not_bounce_and_awards_point():
    env = PongTrainingEnv(
        PongTrainingEnvConfig(), randomGenerator=random.Random(7)
    )
    plane = env.config.agentPaddleX - (env.config.paddleWidth + env.config.ballWidth) / 2
    env.ballX = plane - 0.01
    env.ballY = 100
    env.agentPaddleY = 300
    env.ballVelocityX = env.config.ballSpeedX
    env.ballVelocityY = 0

    env.step("STAY", "STAY")
    assert env.lastHitBy is None

    for _ in range(120):
        if env.pointWinner is not None:
            break
        env.step("STAY", "STAY")

    assert env.opponentScore == 1
    assert env.pointWinner == "opponent"


def test_all_six_courts_edge_hits_bounce_consistently():
    seeds = [11, 108, 205, 302, 399, 496]
    for seed in seeds:
        session = session_for(seed=seed)
        cfg = session.config
        plane = cfg.agentPaddleX - (cfg.paddleWidth + cfg.ballWidth) / 2
        env = session.env
        env.ballX = plane - 1.0
        env.ballY = 368.0
        env.agentPaddleY = 300
        env.ballVelocityX = cfg.ballSpeedX
        env.ballVelocityY = 240.0

        envelope = session.step("STAY", "STAY")

        assert env.lastHitBy == "agent"
        assert env.ballX == plane
        assert envelope["pointWinner"] is None


# Anyone can update this test, but only for an intentional,
# documented physics-contract change. Never weaken or remove
# collision regression coverage.
def test_paddle_collision_contract_deterministic_regression():
    cfg = PongTrainingEnvConfig()
    halfY = (cfg.paddleHeight + cfg.ballHeight) / 2
    rightPlane = cfg.agentPaddleX - (cfg.paddleWidth + cfg.ballWidth) / 2
    leftPlane = cfg.opponentPaddleX + (cfg.paddleWidth + cfg.ballWidth) / 2

    def env_for(seed):
        env = PongTrainingEnv(cfg, randomGenerator=random.Random(seed))
        env._random_normal = lambda mean, deviation: mean
        return env

    def aim(env, side, style):
        paddleY = env.agentPaddleY if side == "right" else env.opponentPaddleY
        if side == "right":
            plane = rightPlane
            env.ballX = plane - 0.01
            env.ballVelocityX = cfg.ballSpeedX
        else:
            plane = leftPlane
            env.ballX = plane + 0.01
            env.ballVelocityX = -cfg.ballSpeedX
        env.ballY = {
            "center": paddleY,
            "top": paddleY - halfY,
            "bottom": paddleY + halfY,
        }[style]
        env.ballVelocityY = 0.0
        return plane

    def assert_rebound(env, side, plane):
        base = env.comboSmash
        env.step("STAY", "STAY")
        expected = "agent" if side == "right" else "opponent"
        assert env.lastHitBy == expected
        assert env.ballX == plane
        assert env.pointWinner is None
        assert env.comboSmash == base + 1

    # Both paddles: center, top edge, and bottom edge hits rebound once.
    for side in ("right", "left"):
        for style in ("center", "top", "bottom"):
            env = env_for(11)
            plane = aim(env, side, style)
            assert_rebound(env, side, plane)

    # Grazing contact: swept Y inside the paddle span, end-of-step Y outside.
    for side in ("right", "left"):
        env = env_for(11)
        plane = rightPlane if side == "right" else leftPlane
        offset = 1.0 if side == "right" else -1.0
        env.ballX = plane - offset * 1.0
        env.ballVelocityX = offset * cfg.ballSpeedX
        env.ballY = 368.0
        env.ballVelocityY = 240.0
        assert_rebound(env, side, plane)

    # Extreme velocity cannot tunnel through either paddle and cannot earn a
    # phantom point before the collision resolves.
    for side in ("right", "left"):
        env = env_for(11)
        plane = rightPlane if side == "right" else leftPlane
        offset = 1.0 if side == "right" else -1.0
        env.ballX = plane - offset * 0.01
        env.ballY = 300.0
        env.ballVelocityX = offset * 50_000
        env.ballVelocityY = 0.0
        assert_rebound(env, side, plane)

    # Genuine miss: no rebound, then the point is awarded to the correct side.
    env = env_for(7)
    env.ballX = rightPlane - 0.01
    env.ballY = 100.0
    env.ballVelocityX = cfg.ballSpeedX
    env.ballVelocityY = 0.0
    env.step("STAY", "STAY")
    assert env.lastHitBy is None
    for _ in range(120):
        if env.pointWinner is not None:
            break
        env.step("STAY", "STAY")
    assert env.pointWinner == "opponent"
    assert env.opponentScore == 1

    # All six courts with independent seeds produce identical grazing rebounds.
    for seed in (11, 108, 205, 302, 399, 496):
        env = env_for(seed)
        env.ballX = rightPlane - 1.0
        env.ballVelocityX = cfg.ballSpeedX
        env.ballY = 368.0
        env.ballVelocityY = 240.0
        assert_rebound(env, "right", rightPlane)