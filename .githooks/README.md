# Git quality gates

Install once per clone:

```sh
git config core.hooksPath .githooks
```

`pre-commit` rejects staged whitespace, syntax errors, likely secrets, databases, and checkpoints. It runs Ruff or frontend lint only when already available/configured. Documentation-only commits skip code checks.

`pre-push` selects tests for SPEC-01, SPEC-02, DQN, training, fitness, and integration boundaries. Frontend changes run Vitest and the production build. Unmapped and documentation-only changes do not run tests.

Maintain component routing in `test-map.conf`. Each new specification adds one regex-to-test entry; the hook logic normally stays unchanged.

Hooks never modify or stage files. Set `PYTHON` or `NPM` to override the detected executable.
