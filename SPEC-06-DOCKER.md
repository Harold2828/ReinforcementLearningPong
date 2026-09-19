# SPEC-06 Final Verification — Docker

**Result: PASS**

Environment: Docker Desktop 27.2.0, Compose v2.29.2.

## Checks

| # | Check | Status |
|---|-------|--------|
| 1 | Backend starts successfully via `docker compose up backend` | PASS |
| 2 | Previous Werkzeug `RuntimeError` (Werkzeug web server not designed for production) resolved | PASS |
| 3 | No unsafe production-server workarounds used (`allow_unsafe_werkzeug` / `debug=True` / dev reloader) | PASS |
| 4 | All five arenas stream LIVE `match_snapshot` data through Docker (port 5001) | PASS |
| 5 | Pause/resume/stop work end-to-end through Docker; DB consistent after stop | PASS |

## Resolution of the Werkzeug RuntimeError

`flask_socketio.run()` raises its "Werkzeug web server is not designed for production" error whenever the app runs in threading async mode with non-TTY stdin — exactly the Docker Compose case (previous container exited 1 with that traceback). Resolved without the unsafe flag by serving through a production WSGI server:

- `Server/requirements-runtime.txt` — added `gunicorn`.
- `Server/Dockerfile` — CMD now `gunicorn --bind 0.0.0.0:5001 --workers 2 run:app` (the Flask-SocketIO middleware already wraps `app.wsgi_app`; simple-websocket supports the gunicorn environ socket for WebSocket upgrades).
- `Server/run.py` unchanged (still the local-development entrypoint).

## Docker Compose support for evolution

- `docker-compose.yml` backend service now sets the evolution environment (`EVOLUTION_TRAINING_ENABLED`, `EVOLUTION_DATA_PATH=/app/evolution_data`, `EVOLUTION_ROUND_TICKS`, `EVOLUTION_STEPS_PER_AGENT`, replay/training knobs, `EVOLUTION_DEVICE=cpu`) and adds an `evolution_data` named volume.

## Live check through Docker (Node socket.io client -> host port 5001)

8/8 checks PASS: connect; `evolution_start_run` ack `{"status":"started","runUuid":"smoke-run-1"}`; LIVE snapshots on all five arenas; scores update; ball/paddle movement; pause ack + snapshot halt; resume ack + snapshots resume; stop ack + `run_finished` + snapshots stop.

DB inside container after stop: `runs [('smoke-run-1','cancelled')]`, `generations [(1,'aborted')]` — consistent.