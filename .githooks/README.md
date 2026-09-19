# Git quality gates

Install once per clone:

```sh
git config core.hooksPath .githooks
```

`pre-commit` rejects staged whitespace errors and syntax-invalid staged Python or JavaScript. Documentation-only commits skip syntax checks.

`pre-push` selects existing pytest files from changed backend components and runs Vitest for frontend source or package changes. Unmapped and documentation-only changes do not run tests.

Hooks never modify or stage files. Set `PYTHON` or `NPM` to override the detected executable.
