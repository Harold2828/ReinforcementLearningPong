# Pong Game Frontend

This is the PhaserJS game client for the Reinforcement Learning Pong project. It renders Pong, sends Q-learning state updates to the backend, and displays a training dashboard for connection status, episode metrics, rewards, epsilon, score, and hit/miss feedback.

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
| Start training | Sends `start_training` and resumes backend action requests. |
| Stop training | Sends `stop_training` and pauses training state updates. |
| Reset episode | Resets scores, ball state, episode flags, and backend episode state. |
| Arrow up/down | Moves the agent paddle while training is stopped. |
| W/S | Moves the opponent paddle. |

## Dashboard

The dashboard shows:

- Backend connection status.
- Agent mode: `Training`, `Evaluation`, or `Manual`.
- Episode number.
- Latest reward value.
- Current epsilon value.
- Current score.
- Hit/miss and connection feedback.

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
