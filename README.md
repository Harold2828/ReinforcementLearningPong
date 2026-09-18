# Reinforcement Learning Pong

This project combines a PhaserJS Pong game with a Flask-SocketIO backend and configurable reinforcement-learning agents. It supports the original independent tabular Q-learning agents and a shared Dueling Double DQN neural policy with replay memory and target-network training.

## Project Structure

| Path | Purpose |
|---|---|
| `PongGame/` | Vite + PhaserJS frontend game and training dashboard. |
| `Server/` | Flask-SocketIO backend, Q-learning logic, tests, and model persistence. |
| `docker-compose.yml` | Runs frontend and backend together for local development. |
| `.env.example` | Documents Q-learning configuration values. |

## Game Modes

| Mode | Left Paddle | Right Paddle | Learning |
|---|---|---|---|
| `Human vs AI` | Human | AI Agent | Disabled |
| `Human vs AI Training` | Human | AI Agent | Enabled for AI only |
| `AI vs Human` | AI Opponent | Human | Disabled |
| `AI vs AI` | AI Opponent | AI Agent | Disabled |
| `Training Self-Play` | AI Opponent | AI Agent | Enabled |
| `Evaluation` | AI Opponent | AI Agent | Disabled |

## Multi-Agent Q-Learning Contract

The frontend sends `state_update` WebSocket messages with these required fields:

| Field | Type | Description |
|---|---:|---|
| `gameMode` | string | Active game mode. |
| `ballX` | number | Ball horizontal position. |
| `ballY` | number | Ball vertical position. |
| `ballVelocityX` | number | Ball horizontal velocity. |
| `ballVelocityY` | number | Ball vertical velocity. |
| `agentPaddleY` | number | Right/main agent paddle vertical position. |
| `opponentPaddleY` | number | Left/opponent paddle vertical position. |
| `agentScore` | number | Right/main agent score. |
| `opponentScore` | number | Left/opponent score. |

Optional fields include `lastHitBy`, `pointWinner`, `episodeId`, `previousAgentDistanceToBall`, `previousOpponentDistanceToBall`, `comboSmash`, `width`, and `height`.

The backend returns `agentAction` and `opponentAction` in `ai_move`. Each action is `UP`, `DOWN`, or `STAY`. Invalid frontend payloads are rejected through `state_error`.

## Reward Rules

The adversarial Q-learning reward function uses:

| Event | Agent Reward | Opponent Reward |
|---|---:|---:|
| Agent scores point | `+5.0` | `-5.0` |
| Opponent scores point | `-5.0` | `+5.0` |
| Agent hits ball | `+1.0 + min(comboSmash * 0.02, 0.5)` | small negative |
| Opponent hits ball | small negative | `+1.0 + min(comboSmash * 0.02, 0.5)` |
| Paddle moves closer to incoming ball | positive alignment reward | positive alignment reward |
| Episode survives one step | small survival reward | small survival reward |

Terminal point updates do not bootstrap from the next state, and training episodes decay epsilon after the point is completed. Metrics tracked per completed episode include agent/opponent win rate, average reward, hit rate, self-play episodes, human-training episodes, and epsilon values.

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

The backend stores Q-learning progress in separate model files inside the backend container volume:

```text
models/agent_q_learning_model.json
models/opponent_q_learning_model.json
```

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
make pretrain
make train-self-play
make reset-models CONFIRM=reset
make status
make logs
make down
```

Headless pretraining runs a fast backend-only Pong environment before visual gameplay. You can tune the run length with:

```bash
EPISODES=2000 MAX_STEPS=1000 make pretrain
```

## Configuration

Set `RL_ALGORITHM=dqn` for neural training or `RL_ALGORITHM=tabular` for the original Q-learning implementation. Docker uses the neural mode from `.env.example` by default.

Q-learning and neural configuration are read from environment variables:

| Variable | Default |
|---|---:|
| `AGENT_MODEL_SAVE_PATH` | `models/agent_q_learning_model.json` |
| `OPPONENT_MODEL_SAVE_PATH` | `models/opponent_q_learning_model.json` |
| `Q_LEARNING_RATE` | `0.2` |
| `Q_DISCOUNT_FACTOR` | `0.95` |
| `Q_EPSILON_START` | `1.0` |
| `Q_EPSILON_MIN` | `0.05` |
| `Q_EPSILON_DECAY` | `0.995` |
| `Q_MAX_STEPS_PER_EPISODE` | `1000` |
| `HUMAN_TRAINING_EPSILON` | `0.1` |
| `DQN_MODEL_SAVE_PATH` | `models/dqn_self_play.pt` |
| `DQN_REPLAY_CAPACITY` | `100000` |
| `DQN_REPLAY_WARMUP` | `5000` |
| `DQN_BATCH_SIZE` | `64` |
| `DQN_LEARNING_RATE` | `0.0001` |
| `DQN_DISCOUNT_FACTOR` | `0.99` |
| `DQN_EPSILON_DECAY_STEPS` | `100000` |
| `DQN_TARGET_UPDATE_INTERVAL` | `1000` |

The DQN consumes normalized structured state rather than pixels. Both paddle perspectives are mirrored into one shared policy, and its checkpoint includes online/target weights, optimizer state, epsilon progress, and training steps.

In `HUMAN_VS_AI_TRAINING`, the human controls the left paddle and only the right AI policy learns. Exploration is capped by `HUMAN_TRAINING_EPSILON` so human-facing play remains reasonably competitive. Normal `HUMAN_VS_AI` never modifies the model.

## Troubleshooting

- If the frontend shows `Disconnected`, confirm the backend is running on port `5001`.
- If Docker cannot rebuild the frontend, run `docker compose build --no-cache frontend`.
- If the Q-learning model file is missing or corrupted, the backend starts with a new Q-table and logs a warning.
