# Developer Guide

## Modification map

| Change | Primary files | Required verification |
|---|---|---|
| Physics constants/order | `pong_training_env.py`, `simulation.py`, `BasicGame.js` | Deterministic trajectory and Classic regression tests. |
| DQN inputs/network/training | `dqn_agent.py` | DQN architecture, checkpoint, replay, and orchestrator tests. Increment model spec for incompatible checkpoints. |
| Population/crossover/mutation | `genome.py`, `genetic.py` | Genetic and three-generation lineage tests. |
| Fitness/evaluation | `evaluation.py` | Fitness bounds, side reversal, inference-only state checks. |
| Generation lifecycle | `training_orchestrator.py` | Pair persistence, terminal replay, reproduction reset, max-generation tests. |
| Socket contract | `sockets.py`, `socketManager.js`, `evolutionContract.js` | Socket and frontend contract tests. |
| Five-court UI/interpolation | `EvolutionTrainingScene.js`, `pongCourtView.js`, `snapshotPlayback.js` | Playback, feed, arena-state tests and browser smoke. |
| Database/checkpoints | `Server/app/persistence/` | Persistence DB/store/checkpoint tests and migration compatibility. |

Preserve these invariants:

- Ten independent DQNs and five independent arenas.
- Point transitions are terminal even though the continuous match continues.
- No training or exploration during evaluation.
- Pairing state persists for the generation; reproduction is the reset boundary.
- Never load or copy incompatible tensors.
- Frontend interpolation consumes real ordered snapshots only.

## Configuration

Docker Compose loads `.env.example` and overrides evolution settings in `docker-compose.yml`. Important variables include:

| Variable | Docker value | Meaning |
|---|---:|---|
| `EVOLUTION_POPULATION_SIZE` | 12 | Independent DQN agents. |
| `EVOLUTION_COURT_COUNT` | 6 | Simultaneous isolated training arenas. |
| `EVOLUTION_PARENT_COUNT` | 6 | Selected parent pool. |
| `EVOLUTION_ELITE_COUNT` / `OFFSPRING_COUNT` | 2 / 10 | Next-generation composition. |
| `EVOLUTION_STEPS_PER_AGENT` | 100000 | Per-generation training action budget. |
| `EVOLUTION_ROUND_TICKS` | 150 | Scheduler chunk; does not reset a match. |
| `EVOLUTION_SNAPSHOT_INTERVAL` | 10 | Physics steps between emitted snapshot batches. |
| `EVOLUTION_OPTIMIZER_INTERVAL` | 32 | Normal optimizer cadence; points also trigger an update attempt. |
| `EVOLUTION_REPLAY_CAPACITY` | 2048 | Per-agent replay capacity in Docker. |
| `EVOLUTION_REPLAY_WARMUP` / `BATCH_SIZE` | 16 / 16 | Minimum replay and minibatch size. |
| `EVOLUTION_MAX_GENERATIONS` | 3 | Configurable maximum lifecycle iterations. |
| `EVOLUTION_EVALUATION_SEEDS` | 101,211,307 | Comparable evaluation seeds. |
| `EVOLUTION_EVALUATION_MAX_STEPS` | 2000 | Evaluation match cap. |
| `EVOLUTION_EVALUATION_WIN_SCORE` | 7 | Evaluation-only score termination. |
| `EVOLUTION_EVALUATION_CONCURRENCY` | 6 | Maximum tournament matches in one batch. |

## Docker and Makefile

Common commands:

```bash
make reset-backend      # rebuild/recreate only backend
make reset-frontend     # rebuild/recreate only frontend
make test               # backend tests, frontend tests, frontend build
make production         # clean, test, rebuild, start, status
make status
make logs
make down
```

Direct development:

```bash
python -m pytest Server/tests
cd PongGame && npm test && npm run build
docker compose up -d --build
```

The runtime backend image installs `requirements-runtime.txt`, which excludes pytest. Run tests in a development environment using `Server/requirements.txt`, or install pytest only in a disposable test container.

Data volumes:

- `q_learning_models` → `/app/models`
- `evolution_data` → `/app/evolution_data` (SQLite and generation checkpoints)

Do not remove volumes during ordinary rebuilds. `make reset-models CONFIRM=reset` intentionally removes model data.

## Git quality gates

`core.hooksPath` points to `.githooks`.

- Pre-commit checks staged whitespace, rejects generated/sensitive artifacts, parses Python, optionally runs Ruff, and validates JavaScript syntax.
- Pre-push maps changed paths to focused pytest targets. Phaser changes run all frontend tests and the Vite build.
- Never bypass hooks. A changed test is always included in the pre-push selection.

## Targeted lifecycle checks

Key tests are in:

- `Server/tests/test_simulation.py`
- `Server/tests/test_training_orchestrator.py`
- `Server/tests/test_fitness_evaluation_evolution.py`
- `PongGame/src/test/liveEvolutionFeed.test.js`
- `PongGame/src/test/snapshotPlayback.test.js`
- `PongGame/src/test/evolutionContract.test.js`

For browser validation, confirm six courts, twelve paddles, six moving balls, accumulated scores, LIVE metrics, and continued motion across multiple scheduler chunks. Also return to Classic 1v1 to check that its scene remains unaffected.

## Known limitations

- Training pairings stay fixed per generation; evaluation runs a separate full round robin.
- Replay buffers and optimizer state are not inherited into offspring or elites; elites preserve trained network weights only.
- Snapshot backpressure intentionally drops stale batches.
- Pause/stop responsiveness is bounded by `roundTicks`.
- SQLite is suitable for the current single-process service, not concurrent distributed workers.
- No champion promotion workflow (SPEC-08) exists.
- No automated hyperparameter/performance optimizer (SPEC-10) exists.
- The frontend bundle triggers Vite's large-chunk warning.
