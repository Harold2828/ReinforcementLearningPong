from app import create_app, socketio


def valid_state_payload():
    return {
        "ballX": 400,
        "ballY": 300,
        "ballVelocityX": -120,
        "ballVelocityY": 30,
        "paddleY": 295,
        "opponentPaddleY": 310,
        "scoreAgent": 0,
        "scoreOpponent": 0,
        "done": False,
        "width": 800,
        "height": 600,
    }


def test_websocket_valid_state_returns_action(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_SAVE_PATH", str(tmp_path / "q_learning_model.json"))
    app = create_app()
    client = socketio.test_client(app)

    response = client.emit("state_update", valid_state_payload(), callback=True)

    assert response["action"] in {"UP", "DOWN", "STAY"}
    assert isinstance(response["reward"], float)


def test_websocket_invalid_state_returns_controlled_error(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_SAVE_PATH", str(tmp_path / "q_learning_model.json"))
    app = create_app()
    client = socketio.test_client(app)
    payload = valid_state_payload()
    payload.pop("ballX")

    response = client.emit("state_update", payload, callback=True)

    assert "Missing required state field" in response["message"]


def test_training_control_events_update_status(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_SAVE_PATH", str(tmp_path / "q_learning_model.json"))
    app = create_app()
    client = socketio.test_client(app)

    stopResponse = client.emit("stop_training", callback=True)
    startResponse = client.emit("start_training", callback=True)
    resetResponse = client.emit("reset_episode", callback=True)

    assert stopResponse["mode"] == "Evaluation"
    assert startResponse["mode"] == "Training"
    assert resetResponse["episode"] >= 1
