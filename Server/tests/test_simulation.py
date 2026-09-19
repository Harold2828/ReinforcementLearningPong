from app.evolution.simulation import MatchAssignment, MatchConfig, MatchSession, MatchParticipant, SimulationEngine, TERMINAL


def participant(name, epsilon=0.0):
    return MatchParticipant(agentId=name, generation=3, epsilon=epsilon)


def assignment(arena_id="arena-0", seed=11, reversed_sides=False):
    return MatchAssignment(
        arenaId=arena_id,
        runId="run-1",
        generationId="gen-1",
        agentA=participant(f"agent-a-{arena_id}"),
        agentB=participant(f"agent-b-{arena_id}"),
        reversedSides=reversed_sides,
        seed=seed,
    )


def five_assignments(seed=11):
    return [assignment(f"arena-{index}", seed=seed + index * 97) for index in range(5)]


def signature(envelopes):
    return [
        {
            "ballX": envelope["ball"]["x"],
            "ballY": envelope["ball"]["y"],
            "paddleA": envelope["agentA"]["paddleY"],
            "paddleB": envelope["agentB"]["paddleY"],
            "scoreA": envelope["agentA"]["score"],
            "scoreB": envelope["agentB"]["score"],
        }
        for envelope in envelopes
    ]


def without_timestamp(envelope):
    return {key: value for key, value in envelope.items() if key != "stateTimestamp"}


def test_deterministic_trajectories_for_equal_seeds_and_actions():
    actions = {f"arena-{index}": ("UP", "DOWN") for index in range(5)}
    first = SimulationEngine(five_assignments(), MatchConfig(maxTotalSteps=200))
    second = SimulationEngine(five_assignments(), MatchConfig(maxTotalSteps=200))

    for _ in range(120):
        assert signature(first.step(actions)) == signature(second.step(actions))


def test_matches_are_isolated_under_divergent_actions():
    actions = {f"arena-{index}": ("UP", "DOWN") for index in range(5)}
    base = SimulationEngine(five_assignments(), MatchConfig(maxTotalSteps=200))
    altered = SimulationEngine(five_assignments(), MatchConfig(maxTotalSteps=200))

    for _ in range(100):
        base.step(actions)
        altered_actions = dict(actions)
        altered_actions["arena-0"] = ("STAY", "STAY")
        altered.step(altered_actions)

    for arena_id in ["arena-1", "arena-2", "arena-3", "arena-4"]:
        assert signature([base.byArena[arena_id].snapshot()]) == signature(
            [altered.byArena[arena_id].snapshot()]
        )


def test_paddle_collision_precedes_boundary_and_records_returns():
    session = MatchSession(assignment(), MatchConfig())
    contact = session.config.agentPaddleX - (
        session.config.paddleWidth + session.config.ballWidth
    ) / 2
    session.env.ballX = contact - session.env.ballVelocityX * session.config.stepSeconds + 0.01
    session.env.ballY = session.env.agentPaddleY
    session.env.ballVelocityX = session.config.ballSpeedX
    session.env.ballVelocityY = 0

    envelope = session.step("STAY", "STAY")

    assert envelope["ball"]["vx"] < 0
    assert envelope["agentA"]["returns"] == 1
    assert envelope["pointWinner"] is None


def test_wall_bounce_flips_vertical_velocity():
    session = MatchSession(assignment(), MatchConfig())
    session.env.ballY = 1
    session.env.ballVelocityY = -session.config.ballSpeedY
    session.env.ballVelocityX = session.config.ballSpeedX

    session.step("STAY", "STAY")

    assert session.env.ballVelocityY > 0


def test_point_reset_matches_classic_serve_without_resetting_paddles(monkeypatch):
    session = MatchSession(assignment(), MatchConfig())
    session.env.agentPaddleY = 220
    session.env.opponentPaddleY = 380
    samples = iter((175.0, 170.0))
    monkeypatch.setattr(session.env, "_random_normal", lambda *_: next(samples))
    session.env.ballX = session.config.width - session.config.ballWidth / 2 - 1
    session.env.ballVelocityX = session.config.ballSpeedX
    session.env.ballVelocityY = 0

    envelope = session.step("STAY", "STAY")

    assert envelope["pointWinner"] == "opponent"
    assert envelope["ball"]["x"] == 0.5
    assert envelope["ball"]["y"] == 0.5
    assert envelope["ball"]["vx"] == -1.75
    assert envelope["ball"]["vy"] == -1.7
    assert session.env.agentPaddleY == 220
    assert session.env.opponentPaddleY == 380


def test_scoring_ownership_right_paddle_is_agent_left_is_opponent():
    session = MatchSession(assignment(), MatchConfig())
    session.env.ballX = session.config.width - 1
    session.env.ballY = 1
    session.env.ballVelocityX = session.config.ballSpeedX
    session.env.ballVelocityY = 0

    session.step("STAY", "STAY")

    assert session.env.agentScore == 0
    assert session.env.opponentScore == 1
    assert session.pointWinner == "opponent"
    agentReward, opponentReward = session.lastRewards
    assert agentReward < 0 and opponentReward > 0


def test_combo_counts_only_on_a_winning_point():
    session = MatchSession(assignment(seed=5), MatchConfig(winScore=1, maxTotalSteps=2000))
    while session.status != TERMINAL:
        session.step("UP", "DOWN")

    expected = session.rallyReturnsA if session.pointWinner == "agent" else session.rallyReturnsB
    assert session.combo == expected


def test_terminal_scoring_happens_once_and_no_bootstrap():
    session = MatchSession(assignment(seed=5), MatchConfig(winScore=1, maxTotalSteps=2000))
    while session.status != TERMINAL:
        session.step("UP", "DOWN")

    terminal_snapshot = session.snapshot()
    frozen = session.step("UP", "DOWN")
    assert without_timestamp(frozen) == without_timestamp(terminal_snapshot)
    assert session.status == TERMINAL
    assert session.env.agentScore + session.env.opponentScore >= 1

    restarted = session.reset()
    assert session.status != TERMINAL
    assert session.sequence == 1
    assert restarted["matchId"] != terminal_snapshot["matchId"]
    assert restarted["agentA"]["score"] == 0 and restarted["agentB"]["score"] == 0


def test_side_reversal_keeps_rewards_with_the_agent_identity():
    session = MatchSession(assignment(seed=6, reversed_sides=True), MatchConfig())
    session.env.ballX = session.config.width - 1
    session.env.ballY = 1
    session.env.ballVelocityX = session.config.ballSpeedX
    session.env.ballVelocityY = 0

    envelope = session.step("STAY", "STAY")

    assert envelope["agentA"]["paddleX"] < envelope["agentB"]["paddleX"]
    assert session._identity_scores() == (1, 0)
    assert session.pointWinner == "agent"
    assert session.combo == 0
    agentReward, opponentReward = session.lastRewards
    assert agentReward > 0 and opponentReward < 0


def test_reversed_snapshot_labels_right_paddle_points_to_agentb():
    session = MatchSession(assignment(seed=6, reversed_sides=True), MatchConfig())
    session.env.ballX = 1
    session.env.ballY = 1
    session.env.ballVelocityX = -session.config.ballSpeedX
    session.env.ballVelocityY = 0

    envelope = session.step("UP", "DOWN")

    assert envelope["agentA"]["score"] == 0
    assert envelope["agentB"]["score"] == 1
    assert envelope["pointWinner"] == "opponent"
    assert envelope["agentA"]["paddleX"] < envelope["agentB"]["paddleX"]
    assert 0 <= envelope["agentA"]["paddleY"] <= 1
    assert 0 <= envelope["agentB"]["paddleY"] <= 1
    assert session.combo == 0
