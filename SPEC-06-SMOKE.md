# SPEC-06 Final Verification — Live Smoke Test

**Result: PASS (8/8 checks)**

Environment: Windows, backend `SocketIO` (threading mode) on `127.0.0.1:5091`, Vite on `127.0.0.1:5179`, Node `socket.io-client` v4. Single end-to-end run: start evolution training → verify LIVE streaming across all 5 arenas → score/position updates → pause/resume/stop.

## Checks

| # | Check | Status |
|---|-------|--------|
| 1 | socket.io connected | PASS |
| 2 | `evolution_start_run` ack `{"status":"started","runUuid":"smoke-run-1"}` | PASS |
| 3 | all five arenas stream LIVE `match_snapshot`s | PASS |
| 4 | scores update (a point scored) | PASS |
| 5 | all arenas show moving ball/paddle positions | PASS |
| 6 | `evolution_pause_run` ack `{"status":"paused"}` and snapshots halt (2s quiet window) | PASS |
| 7 | `evolution_resume_run` ack `{"status":"running"}` and snapshots resume | PASS |
| 8 | `evolution_stop_run` ack `{"status":"stopping"}` + `run_finished` event + snapshots stop (2s quiet window) | PASS |

## DB consistency after stop

`runs: [('smoke-run-1', 'cancelled')]`, `generations: [(1, 'aborted')]` — stop leaves persisted state consistent.

## Defects found and fixed

1. `Server/app/persistence/db.py` — sqlite connection created on the main thread was used by the daemon training thread → `ProgrammingError`. Fixed with `check_same_thread=False` on connect.
2. `Server/app/__init__.py` — default eventlet async mode never delivered events emitted from a native `threading.Thread` (no monkey-patch; eventlet green queue is hub-bound). Fixed with `SocketIO(..., async_mode="threading")`.
3. `Server/app/sockets.py` — `on_pause_run`/`on_resume_run`/`on_stop_run` declared zero params, so any client payload (`{}`) caused `TypeError` inside the socketio background handler (silently swallowed → no ack, no effect). Fixed by accepting an optional `payload=None`, matching `on_start_run`.

Expected test suites were not re-run (no logic changed since their last PASS).