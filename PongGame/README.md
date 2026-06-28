# Pong Game Frontend

This is the PhaserJS game client for the Reinforcement Learning Pong project. It renders Pong, sends multi-agent Q-learning state updates to the backend, and displays a training dashboard for connection status, mode ownership, self-play metrics, rewards, epsilon values, score, and hit feedback.

## Tech Stack

- Vite
- PhaserJS
- Socket.IO client
- Vitest

## Run

```bash
npm install
npm run dev
```

Open `http://localhost:5173`.

To point the game at another backend:

```bash
VITE_BACKEND_URL=http://localhost:5001 npm run dev
```

## Controls

| Control | Purpose |
|---|---|
| Game mode selector | Switches between `HUMAN_VS_AI`, `AI_VS_HUMAN`, `AI_VS_AI`, `TRAINING_SELF_PLAY`, and `EVALUATION`. |
| Start training | Sends `start_training` and resumes backend action requests. |
| Stop training | Sends `stop_training` and pauses training state updates. |
| Reset episode | Resets scores, ball state, episode flags, and backend episode state. |
| Arrow up/down | Moves the agent paddle when that side is human-controlled. |
| W/S | Moves the opponent paddle when that side is human-controlled. |

## Dashboard

The dashboard shows:

- Backend connection status.
- Current game mode and paddle ownership.
- Learning status.
- Episode number.
- Latest agent and opponent reward values.
- Current agent and opponent epsilon values.
- Current score.
- Agent/opponent win rates, average rewards, hit rates, and self-play episode count.
- Hit and connection feedback.

## Tests

```bash
npm test
```

## Build

```bash
npm run build
```

## Docker

From the repository root:

```bash
docker compose up --build frontend
```

The Docker Compose setup passes `VITE_BACKEND_URL=http://localhost:5001` so the browser can connect to the backend port exposed on the host.
