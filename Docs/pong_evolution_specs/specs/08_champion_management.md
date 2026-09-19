# SPEC-08 — Hall of Fame and human-versus-champion mode

## Goal
Keep a stable best-*validated* agent across generations and allow the user to play it in the existing individual Pong view while evolution proceeds.

## Requirements
- Maintain immutable champion snapshot/checkpoint and record its source agent, generation, architecture, evaluation suite version, metrics, timestamp, and hash. Keep previous champions for rollback.
- At generation close, compare candidate against incumbent using a **predeclared held-out protocol**: common opponents/seeds/side order, multiple matches, and a documented threshold or uncertainty criterion. If evidence is inconclusive, retain incumbent; an evolving population member is not automatically promoted.
- `HUMAN_VS_CHAMPION` loads a frozen copy of a selected champion checkpoint into an isolated inference-only service; epsilon=0; no gradient, replay insertion, or training-status mutation.
- Human uses left paddle with `W/S`, champion uses right. Preserve original single arena, scoring, game start/reset behavior.
- **Mandatory dependency (SPEC-05):** `HUMAN_VS_CHAMPION` must run on the authoritative Python physics engine (SPEC-05 `MatchSession`) with Phaser as viewer only. SPEC-08's inference-only service supplies the champion's right-paddle action through the same `MatchSession.step(championAction, humanAction)` interface used by evolutionary matches, replacing the placeholder heuristic champion policy. Do not run Phaser physics for this mode.
- New champion promotions appear in the UI but must not hot-swap the policy mid-match; new match may select latest champion.
- If there is no validated champion yet, show an explicit not-ready state instead of loading a random agent silently.

## Acceptance criteria
- Running a human match leaves champion checkpoint hash, optimizer, and evolution steps unchanged.
- Population training and human match work concurrently without reset collisions.
- A failed promotion preserves previous champion; historical snapshots remain selectable and rollback succeeds.
