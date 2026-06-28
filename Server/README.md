# Backend Server

The backend exposes a Flask-SocketIO service for the Pong game and hosts two independent tabular Q-learning agents for human play, evaluation, AI-vs-AI play, and self-play training.

## Responsibilities

- Validate game state payloads from the PhaserJS frontend.
- Validate the multi-agent game mode and state payload.
- Select valid actions for both sides: `UP`, `DOWN`, or `STAY`.
- Apply Q-learning updates with epsilon-greedy exploration during `TRAINING_SELF_PLAY`.
- Calculate adversarial rewards for scoring, hits, alignment, pressure, and survival.
- Save and load separate Q-table progress from `models/agent_q_learning_model.json` and `models/opponent_q_learning_model.json`.
- Emit training status, state errors, dual action responses, and self-play metrics.

## Game Modes

| Mode | Agent Paddle | Opponent Paddle | Learning |
|---|---|---|---|
| `HUMAN_VS_AI` | Human | AI | Off |
| `AI_VS_HUMAN` | AI | Human | Off |
| `AI_VS_AI` | AI | AI | Off |
| `TRAINING_SELF_PLAY` | AI | AI | On |
| `EVALUATION` | AI | AI | Off |

## WebSocket Events

| Event | Direction | Purpose |
|---|---|---|
| `state_update` | Frontend to backend | Sends the current multi-agent Pong state. |
| `ai_move` | Backend to frontend | Returns `agentAction`, `opponentAction`, rewards, epsilon values, and metrics. |
| `state_error` | Backend to frontend | Reports invalid payloads. |
| `start_training` | Frontend to backend | Enables Q-learning updates. |
| `stop_training` | Frontend to backend | Disables Q-learning updates and saves the model. |
| `reset_episode` | Frontend to backend | Clears the current episode tracker. |
| `training_status` | Backend to frontend | Reports mode, learning status, epsilon values, and metrics. |

## Run

```bash
pip install -r requirements.txt
python run.py
```

The server listens on `http://localhost:5001`.

## Tests

```bash
pytest tests
```

From the repository root:

```bash
pytest Server/tests
```

## Docker

From the repository root:

```bash
docker compose up --build backend
```

## Configuration

| Variable | Description |
|---|---|
| `AGENT_MODEL_SAVE_PATH` | Agent Q-table save/load path relative to `Server/` unless absolute. |
| `OPPONENT_MODEL_SAVE_PATH` | Opponent Q-table save/load path relative to `Server/` unless absolute. |
| `Q_LEARNING_RATE` | Q-value learning rate, greater than `0` and up to `1`. |
| `Q_DISCOUNT_FACTOR` | Future reward discount factor from `0` to `1`. |
| `Q_EPSILON_START` | Initial exploration rate. |
| `Q_EPSILON_MIN` | Minimum exploration rate. |
| `Q_EPSILON_DECAY` | Epsilon multiplier applied after completed episodes. |
| `Q_MAX_STEPS_PER_EPISODE` | Safety limit for episode length. |

## Model Persistence

If a model file is missing, that agent starts with an empty Q-table and logs a warning. If a model file is corrupted, that agent also starts with an empty Q-table instead of crashing.
