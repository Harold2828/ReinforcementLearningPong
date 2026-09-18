from app import create_app, socketio


def valid_state_payload():
    return {
        "gameMode": "HUMAN_VS_AI",
        "ballX": 400,
        "ballY": 300,
        "ballVelocityX": -120,
        "ballVelocityY": 30,
        "agentPaddleY": 295,
        "opponentPaddleY": 310,
        "agentScore": 0,
        "opponentScore": 0,
        "lastHitBy": None,
        "pointWinner": None,
        "comboSmash": 0,
        "width": 800,
        "height": 600,
    }


def test_websocket_valid_state_returns_action(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_MODEL_SAVE_PATH", str(tmp_path / "agent_q_learning_model.json"))
    monkeypatch.setenv("OPPONENT_MODEL_SAVE_PATH", str(tmp_path / "opponent_q_learning_model.json"))
    app = create_app()
    client = socketio.test_client(app)

    response = client.emit("state_update", valid_state_payload(), callback=True)

    assert response["action"] in {"UP", "DOWN", "STAY"}
    assert response["agentAction"] in {"UP", "DOWN", "STAY"}
    assert response["opponentAction"] == "STAY"
    assert isinstance(response["agentReward"], float)


def test_websocket_invalid_state_returns_controlled_error(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_MODEL_SAVE_PATH", str(tmp_path / "agent_q_learning_model.json"))
    monkeypatch.setenv("OPPONENT_MODEL_SAVE_PATH", str(tmp_path / "opponent_q_learning_model.json"))
    app = create_app()
    client = socketio.test_client(app)
    payload = valid_state_payload()
    payload.pop("ballX")

    response = client.emit("state_update", payload, callback=True)

    assert "Missing required state field" in response["message"]


def test_training_control_events_update_status(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_MODEL_SAVE_PATH", str(tmp_path / "agent_q_learning_model.json"))
    monkeypatch.setenv("OPPONENT_MODEL_SAVE_PATH", str(tmp_path / "opponent_q_learning_model.json"))
    app = create_app()
    client = socketio.test_client(app)

    stopResponse = client.emit("stop_training", callback=True)
    startResponse = client.emit("start_training", callback=True)
    resetResponse = client.emit("reset_episode", callback=True)

    assert stopResponse["learningEnabled"] is False
    assert startResponse["connected"] is True
    assert resetResponse["connected"] is True


def test_ai_vs_ai_returns_two_valid_actions(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_MODEL_SAVE_PATH", str(tmp_path / "agent_q_learning_model.json"))
    monkeypatch.setenv("OPPONENT_MODEL_SAVE_PATH", str(tmp_path / "opponent_q_learning_model.json"))
    app = create_app()
    client = socketio.test_client(app)
    payload = valid_state_payload()
    payload["gameMode"] = "AI_VS_AI"

    response = client.emit("state_update", payload, callback=True)

    assert response["agentAction"] in {"UP", "DOWN", "STAY"}
    assert response["opponentAction"] in {"UP", "DOWN", "STAY"}


def test_training_self_play_saves_separate_models(monkeypatch, tmp_path):
    agentModelPath = tmp_path / "agent_q_learning_model.json"
    opponentModelPath = tmp_path / "opponent_q_learning_model.json"
    monkeypatch.setenv("AGENT_MODEL_SAVE_PATH", str(agentModelPath))
    monkeypatch.setenv("OPPONENT_MODEL_SAVE_PATH", str(opponentModelPath))
    app = create_app()
    client = socketio.test_client(app)
    payload = valid_state_payload()
    payload["gameMode"] = "TRAINING_SELF_PLAY"
    payload["pointWinner"] = "agent"

    response = client.emit("state_update", payload, callback=True)

    assert response["learningEnabled"] is True
    assert agentModelPath.exists()
    assert opponentModelPath.exists()


def test_dqn_mode_uses_neural_service_and_saves_checkpoint(monkeypatch, tmp_path):
    checkpoint = tmp_path / "dqn_self_play.pt"
    monkeypatch.setenv("RL_ALGORITHM", "dqn")
    monkeypatch.setenv("DQN_MODEL_SAVE_PATH", str(checkpoint))
    monkeypatch.setenv("DQN_REPLAY_CAPACITY", "20")
    monkeypatch.setenv("DQN_REPLAY_WARMUP", "2")
    monkeypatch.setenv("DQN_BATCH_SIZE", "2")
    app = create_app()
    client = socketio.test_client(app)
    payload = valid_state_payload()
    payload["gameMode"] = "TRAINING_SELF_PLAY"

    firstResponse = client.emit("state_update", payload, callback=True)
    payload["pointWinner"] = "agent"
    payload["agentScore"] = 1
    terminalResponse = client.emit("state_update", payload, callback=True)

    assert firstResponse["algorithm"] == "dueling_double_dqn"
    assert terminalResponse["replaySize"] == 2
    assert terminalResponse["episode"] == 2
    assert checkpoint.exists()


def test_human_vs_ai_training_learns_only_for_ai_paddle(monkeypatch, tmp_path):
    checkpoint = tmp_path / "human_training.pt"
    monkeypatch.setenv("RL_ALGORITHM", "dqn")
    monkeypatch.setenv("DQN_MODEL_SAVE_PATH", str(checkpoint))
    monkeypatch.setenv("DQN_REPLAY_CAPACITY", "20")
    monkeypatch.setenv("DQN_REPLAY_WARMUP", "20")
    monkeypatch.setenv("DQN_BATCH_SIZE", "2")
    monkeypatch.setenv("HUMAN_TRAINING_EPSILON", "0.1")
    app = create_app()
    client = socketio.test_client(app)
    payload = valid_state_payload()
    payload["gameMode"] = "HUMAN_VS_AI_TRAINING"

    client.emit("state_update", payload, callback=True)
    payload["pointWinner"] = "agent"
    payload["agentScore"] = 1
    response = client.emit("state_update", payload, callback=True)

    assert response["learningEnabled"] is True
    assert response["opponentAction"] == "STAY"
    assert response["replaySize"] == 1
    assert response["epsilon"] <= 0.1
    assert response["metrics"]["humanTrainingEpisodes"] == 1
    assert checkpoint.exists()
