# Backend Server

The backend exposes a Flask-SocketIO service for the Pong game and hosts the tabular Q-learning agent.

## Responsibilities

- Validate game state payloads from the PhaserJS frontend.
- Select valid actions: `UP`, `DOWN`, or `STAY`.
- Apply Q-learning updates with epsilon-greedy exploration.
- Calculate rewards for hits, misses, scoring, alignment, and survival.
- Save and load Q-table progress from `models/q_learning_model.json`.
- Emit training status, state errors, action responses, and episode metrics.

## WebSocket Events

| Event | Direction | Purpose |
|---|---|---|
| `state_update` | Frontend to backend | Sends the current Pong state. |
| `ai_move` | Backend to frontend | Returns action, reward, episode, epsilon, and metrics. |
| `state_error` | Backend to frontend | Reports invalid payloads. |
| `start_training` | Frontend to backend | Enables Q-learning updates. |
| `stop_training` | Frontend to backend | Disables Q-learning updates and saves the model. |
| `reset_episode` | Frontend to backend | Clears the current episode tracker. |
| `training_status` | Backend to frontend | Reports mode, episode, epsilon, and metrics. |

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
| `MODEL_SAVE_PATH` | Q-table save/load path relative to `Server/` unless absolute. |
| `Q_LEARNING_RATE` | Q-value learning rate, greater than `0` and up to `1`. |
| `Q_DISCOUNT_FACTOR` | Future reward discount factor from `0` to `1`. |
| `Q_EPSILON_START` | Initial exploration rate. |
| `Q_EPSILON_MIN` | Minimum exploration rate. |
| `Q_EPSILON_DECAY` | Epsilon multiplier applied after completed episodes. |
| `Q_MAX_STEPS_PER_EPISODE` | Safety limit for episode length. |

## Model Persistence

If the model file is missing, the server starts with an empty Q-table and logs a warning. If the model file is corrupted, the server also starts with an empty Q-table instead of crashing.
