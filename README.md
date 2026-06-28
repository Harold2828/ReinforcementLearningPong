# Reinforcement Learning Pong

This project combines a PhaserJS Pong game with a Flask-SocketIO backend and a tabular Q-learning agent. The game sends structured state updates to the backend, the backend selects one of three valid actions, and training metrics make gameplay improvement measurable over time.

## Project Structure

| Path | Purpose |
|---|---|
| `PongGame/` | Vite + PhaserJS frontend game and training dashboard. |
| `Server/` | Flask-SocketIO backend, Q-learning logic, tests, and model persistence. |
| `docker-compose.yml` | Runs frontend and backend together for local development. |
| `.env.example` | Documents Q-learning configuration values. |

## Q-Learning Contract

The frontend sends `state_update` WebSocket messages with these required fields:

| Field | Type | Description |
|---|---:|---|
| `ballX` | number | Ball horizontal position. |
| `ballY` | number | Ball vertical position. |
| `ballVelocityX` | number | Ball horizontal velocity. |
| `ballVelocityY` | number | Ball vertical velocity. |
| `paddleY` | number | Agent paddle vertical position. |
| `scoreAgent` | number | Agent score. |
| `scoreOpponent` | number | Opponent score. |
| `done` | boolean | Whether the episode ended. |

Optional fields include `opponentPaddleY`, `agentHitBall`, `agentMissedBall`, `agentScored`, `previousDistanceToBall`, `width`, and `height`.

The backend returns one action in `ai_move`: `UP`, `DOWN`, or `STAY`. Invalid frontend payloads are rejected through `state_error`.

## Reward Rules

The Q-learning reward function uses:

| Event | Reward |
|---|---:|
| Paddle hits ball | `+1.0` |
| Agent scores point | `+5.0` |
| Agent misses ball | `-5.0` |
| Paddle moves closer to ball | `+0.1` |
| Paddle moves away from ball | `-0.1` |
| Episode survives one step | `+0.01` |

Metrics tracked per completed episode include average reward, paddle hit rate, miss rate, episode length, win rate, and epsilon value.

## Run Locally

Backend:

```bash
cd Server
pip install -r requirements.txt
python run.py
```

Frontend:

```bash
cd PongGame
npm install
npm run dev
```

Open `http://localhost:5173`. The game dashboard shows backend connection status, mode, episode, reward, epsilon, score, and hit/miss feedback.

## Tests

Backend:

```bash
pytest Server/tests
```

Frontend:

```bash
cd PongGame
npm test
```

Build check:

```bash
cd PongGame
npm run build
```

## Docker

Copy or adjust values from `.env.example`, then run:

```bash
docker compose up --build
```

Frontend: `http://localhost:5173`  
Backend: `http://localhost:5001`

The backend stores Q-learning progress at `models/q_learning_model.json` inside the backend container volume.

## Make Commands

Production-ready build:

```bash
make production
```

This command removes local test/build artifacts, stops existing project containers, prunes Docker builder cache, runs backend and frontend tests, builds Docker images without cache, starts backend and frontend in detached mode, and prints container status.

Reset only the backend:

```bash
make reset-backend
```

Alias:

```bash
make reset-back
```

Reset only the frontend:

```bash
make reset-frontend
```

Alias:

```bash
make reset-front
```

Useful support commands:

```bash
make test
make status
make logs
make down
```

## Configuration

Q-learning configuration is read from environment variables:

| Variable | Default |
|---|---:|
| `MODEL_SAVE_PATH` | `models/q_learning_model.json` |
| `Q_LEARNING_RATE` | `0.2` |
| `Q_DISCOUNT_FACTOR` | `0.95` |
| `Q_EPSILON_START` | `1.0` |
| `Q_EPSILON_MIN` | `0.05` |
| `Q_EPSILON_DECAY` | `0.995` |
| `Q_MAX_STEPS_PER_EPISODE` | `1000` |

## Troubleshooting

- If the frontend shows `Disconnected`, confirm the backend is running on port `5001`.
- If Docker cannot rebuild the frontend, run `docker compose build --no-cache frontend`.
- If the Q-learning model file is missing or corrupted, the backend starts with a new Q-table and logs a warning.
