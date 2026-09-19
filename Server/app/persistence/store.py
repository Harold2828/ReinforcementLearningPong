from __future__ import annotations

import contextlib
import json
from pathlib import Path
import sqlite3

from .checkpoints import CheckpointManager
from .db import connect, migrate, utc_now
from .errors import ControlledRecoveryError, ProvenanceViolationError

SUPPORTED_MODEL_SPEC_VERSIONS: frozenset[int] = frozenset({2})
LOADABLE_CHECKPOINT_TYPES: frozenset[str] = frozenset({"weights", "full"})


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


class EvolutionStore:
    def __init__(self, db_path: str | Path, checkpoint_root: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_root = Path(checkpoint_root)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self._connection = connect(self.db_path)
        migrate(self._connection)
        self.checkpoints = CheckpointManager(self.checkpoint_root)

    def __enter__(self) -> "EvolutionStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    def _query(self, sql: str, parameters: tuple = ()) -> list[dict]:
        return [dict(row) for row in self._connection.execute(sql, parameters).fetchall()]

    def _query_one(self, sql: str, parameters: tuple = ()) -> dict | None:
        return _row_to_dict(self._connection.execute(sql, parameters).fetchone())

    def _agent_run(self, agent_id: int) -> int | None:
        row = self._query_one(
            "SELECT g.run_id FROM agents a JOIN generations g ON g.id = a.generation_id WHERE a.id = ?",
            (agent_id,),
        )
        return row["run_id"] if row is not None else None

    def _require_agent_in_run(self, agent_id: int, run_id: int) -> None:
        actual = self._agent_run(agent_id)
        if actual is None:
            raise ProvenanceViolationError(f"agent {agent_id} does not exist")
        if actual != run_id:
            raise ProvenanceViolationError(f"agent {agent_id} belongs to run {actual}, not run {run_id}")

    @contextlib.contextmanager
    def transaction(self):
        try:
            yield self._connection
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise

    def create_run(
        self,
        run_uuid: str,
        config: dict,
        seed: int,
        code_revision: str,
        fitness_formula: str,
        benchmark_definition: str,
    ) -> int:
        now = utc_now()
        cursor = self._connection.execute(
            """
            INSERT INTO runs (run_uuid, config_json, seed, code_revision, fitness_formula,
                              benchmark_definition, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?)
            """,
            (
                run_uuid,
                json.dumps(config, sort_keys=True),
                int(seed),
                code_revision,
                fitness_formula,
                benchmark_definition,
                now,
                now,
            ),
        )
        self._connection.commit()
        return cursor.lastrowid

    def get_run(self, run_id: int) -> dict | None:
        return self._query_one("SELECT * FROM runs WHERE id = ?", (run_id,))

    def list_runs(self) -> list[dict]:
        return self._query("SELECT * FROM runs ORDER BY id")

    def set_run_status(self, run_id: int, status: str) -> bool:
        cursor = self._connection.execute(
            "UPDATE runs SET status = ?, updated_at = ? WHERE id = ?",
            (status, utc_now(), run_id),
        )
        self._connection.commit()
        return cursor.rowcount == 1

    def start_generation(self, run_id: int, generation_index: int) -> int:
        cursor = self._connection.execute(
            "INSERT INTO generations (run_id, generation_index, status, started_at) VALUES (?, ?, 'running', ?)",
            (run_id, generation_index, utc_now()),
        )
        self._connection.commit()
        return cursor.lastrowid

    def get_generation(self, generation_id: int) -> dict | None:
        return self._query_one("SELECT * FROM generations WHERE id = ?", (generation_id,))

    def list_generations(self, run_id: int) -> list[dict]:
        return self._query("SELECT * FROM generations WHERE run_id = ? ORDER BY generation_index", (run_id,))

    def complete_generation(self, generation_id: int) -> bool:
        with self.transaction():
            cursor = self._connection.execute(
                "UPDATE generations SET status = 'completed', completed_at = ? WHERE id = ? AND status <> 'completed'",
                (utc_now(), generation_id),
            )
        return cursor.rowcount == 1

    def register_agent(
        self,
        agent_uuid: str,
        generation_id: int,
        role: str,
        architecture_json: str,
    ) -> int:
        cursor = self._connection.execute(
            "INSERT INTO agents (agent_uuid, generation_id, role, architecture_json, status, created_at) "
            "VALUES (?, ?, ?, ?, 'available', ?)",
            (agent_uuid, generation_id, role, architecture_json, utc_now()),
        )
        self._connection.commit()
        return cursor.lastrowid

    def get_agent(self, agent_id: int) -> dict | None:
        return self._query_one("SELECT * FROM agents WHERE id = ?", (agent_id,))

    def list_agents(self, generation_id: int | None = None) -> list[dict]:
        if generation_id is None:
            return self._query("SELECT * FROM agents ORDER BY id")
        return self._query("SELECT * FROM agents WHERE generation_id = ? ORDER BY id", (generation_id,))

    def set_agent_status(self, agent_id: int, status: str) -> bool:
        cursor = self._connection.execute(
            "UPDATE agents SET status = ? WHERE id = ?",
            (status, agent_id),
        )
        self._connection.commit()
        return cursor.rowcount == 1

    def record_parentage(
        self,
        child_agent_id: int,
        parent_agent_id: int,
        mutation_json: str | None = None,
    ) -> int:
        child_run = self._agent_run(child_agent_id)
        parent_run = self._agent_run(parent_agent_id)
        if child_run is None:
            raise ProvenanceViolationError(f"agent {child_agent_id} does not exist")
        if parent_run is None:
            raise ProvenanceViolationError(f"agent {parent_agent_id} does not exist")
        if child_run != parent_run:
            raise ProvenanceViolationError(
                f"parent and child must belong to the same run: child in run {child_run}, parent in run {parent_run}"
            )
        cursor = self._connection.execute(
            "INSERT OR IGNORE INTO parentage (child_agent_id, parent_agent_id, mutation_json) VALUES (?, ?, ?)",
            (child_agent_id, parent_agent_id, mutation_json),
        )
        self._connection.commit()
        return cursor.lastrowid if cursor.rowcount else self._query_one(
            "SELECT id FROM parentage WHERE child_agent_id = ? AND parent_agent_id = ?",
            (child_agent_id, parent_agent_id),
        )["id"]

    def record_checkpoint(
        self,
        agent_id: int,
        relative_path: str,
        sha256: str,
        model_spec_version: int,
        checkpoint_type: str = "full",
    ) -> int:
        with self.transaction():
            cursor = self._connection.execute(
                "INSERT INTO checkpoints (agent_id, relative_path, sha256, model_spec_version, checkpoint_type, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (agent_id, relative_path, sha256, model_spec_version, checkpoint_type, utc_now()),
            )
            checkpoint_id = cursor.lastrowid
            self._connection.execute(
                "UPDATE agents SET checkpoint_id = ? WHERE id = ?",
                (checkpoint_id, agent_id),
            )
        return checkpoint_id

    def get_checkpoint(self, checkpoint_id: int) -> dict | None:
        return self._query_one("SELECT * FROM checkpoints WHERE id = ?", (checkpoint_id,))

    def list_checkpoints(self, agent_id: int) -> list[dict]:
        return self._query("SELECT * FROM checkpoints WHERE agent_id = ? ORDER BY id", (agent_id,))

    def record_match(
        self,
        match_uuid: str,
        run_id: int,
        arena: str,
        agent_a_id: int,
        agent_b_id: int,
        game_seed: int,
        mode: str,
        score_a: int,
        score_b: int,
        combos_a: int,
        combos_b: int,
        duration_steps: int,
        result: dict,
        opponent_type: str | None = None,
    ) -> bool:
        self._require_agent_in_run(agent_a_id, run_id)
        self._require_agent_in_run(agent_b_id, run_id)
        cursor = self._connection.execute(
            """
            INSERT INTO matches (match_uuid, run_id, arena, agent_a_id, agent_b_id, opponent_type,
                                 game_seed, mode, score_a, score_b, combos_a, combos_b,
                                 duration_steps, result_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(match_uuid) DO NOTHING
            """,
            (
                match_uuid,
                run_id,
                arena,
                agent_a_id,
                agent_b_id,
                opponent_type,
                int(game_seed),
                mode,
                int(score_a),
                int(score_b),
                int(combos_a),
                int(combos_b),
                int(duration_steps),
                json.dumps(result, sort_keys=True),
                utc_now(),
            ),
        )
        self._connection.commit()
        return cursor.rowcount == 1

    def get_match(self, match_uuid: str) -> dict | None:
        return self._query_one("SELECT * FROM matches WHERE match_uuid = ?", (match_uuid,))

    def record_evaluation(
        self,
        agent_id: int,
        run_id: int,
        benchmark_version: str,
        metric_components: dict,
        sample_count: int,
        uncertainty: dict | None = None,
        fitness: float | None = None,
    ) -> int:
        self._require_agent_in_run(agent_id, run_id)
        cursor = self._connection.execute(
            "INSERT INTO evaluations (agent_id, run_id, benchmark_version, metric_components_json, "
            "sample_count, uncertainty_json, fitness, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                agent_id,
                run_id,
                benchmark_version,
                json.dumps(metric_components, sort_keys=True),
                int(sample_count),
                json.dumps(uncertainty, sort_keys=True) if uncertainty is not None else None,
                fitness,
                utc_now(),
            ),
        )
        self._connection.commit()
        return cursor.lastrowid

    def record_champion_event(
        self,
        run_id: int,
        agent_id: int,
        event: str,
        evaluation_id: int | None = None,
        checkpoint_id: int | None = None,
    ) -> int:
        self._require_agent_in_run(agent_id, run_id)
        if evaluation_id is not None:
            evaluation = self._query_one(
                "SELECT agent_id, run_id FROM evaluations WHERE id = ?",
                (evaluation_id,),
            )
            if evaluation is None:
                raise ProvenanceViolationError(f"evaluation {evaluation_id} does not exist")
            if evaluation["agent_id"] != agent_id or evaluation["run_id"] != run_id:
                raise ProvenanceViolationError(
                    f"evaluation {evaluation_id} belongs to agent {evaluation['agent_id']} / "
                    f"run {evaluation['run_id']}, not agent {agent_id} / run {run_id}"
                )
        if checkpoint_id is not None:
            checkpoint = self._query_one(
                "SELECT agent_id FROM checkpoints WHERE id = ?",
                (checkpoint_id,),
            )
            if checkpoint is None:
                raise ProvenanceViolationError(f"checkpoint {checkpoint_id} does not exist")
            if checkpoint["agent_id"] != agent_id:
                raise ProvenanceViolationError(
                    f"checkpoint {checkpoint_id} belongs to agent {checkpoint['agent_id']}, not agent {agent_id}"
                )
        cursor = self._connection.execute(
            "INSERT INTO champion_history (run_id, agent_id, event, evaluation_id, checkpoint_id, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, agent_id, event, evaluation_id, checkpoint_id, utc_now()),
        )
        self._connection.commit()
        return cursor.lastrowid

    def list_champion_events(self, run_id: int) -> list[dict]:
        return self._query("SELECT * FROM champion_history WHERE run_id = ? ORDER BY id", (run_id,))

    def reconcile_checkpoints(self, remove_orphans: bool = False) -> dict:
        rows = self._connection.execute(
            "SELECT c.id AS checkpoint_id, c.agent_id, c.relative_path, c.sha256, a.agent_uuid "
            "FROM checkpoints c JOIN agents a ON a.id = c.agent_id"
        ).fetchall()
        report: dict = {"missing": [], "corrupt": [], "orphans": [], "unavailable_agents": []}
        referenced: set[str] = set()

        for row in rows:
            entry = {
                "checkpoint_id": row["checkpoint_id"],
                "agent_id": row["agent_id"],
                "agent_uuid": row["agent_uuid"],
                "relative_path": row["relative_path"],
            }
            referenced.add(row["relative_path"])
            if not self.checkpoints.verify(row["relative_path"], row["sha256"]):
                resolved = None
                try:
                    resolved = self.checkpoints.resolve_owned(row["relative_path"])
                except ControlledRecoveryError:
                    pass
                if resolved is None or not resolved.is_file():
                    entry["reason"] = "file missing"
                    report["missing"].append(entry)
                else:
                    entry["reason"] = "hash mismatch"
                    report["corrupt"].append(entry)
                if row["agent_id"] not in {item["agent_id"] for item in report["unavailable_agents"]}:
                    report["unavailable_agents"].append(row["agent_id"])

        for relative_path in self.checkpoints.list_files():
            if relative_path not in referenced:
                report["orphans"].append(relative_path)
                if remove_orphans:
                    self.checkpoints.delete(relative_path)

        if report["unavailable_agents"]:
            placeholders = ",".join("?" for _ in report["unavailable_agents"])
            with self.transaction():
                self._connection.execute(
                    f"UPDATE agents SET status = 'unavailable' WHERE id IN ({placeholders})",
                    tuple(report["unavailable_agents"]),
                )

        if report["missing"] or report["corrupt"]:
            raise ControlledRecoveryError(
                "checkpoint integrity violated; affected agents marked unavailable", report=report
            )
        return report

    def available_champion_candidates(self, run_id: int) -> list[dict]:
        rows = self._connection.execute(
            """
            SELECT a.id AS agent_id, a.agent_uuid, a.architecture_json,
                   c.id AS checkpoint_id, c.relative_path, c.sha256,
                   c.model_spec_version, c.checkpoint_type
            FROM agents a
            JOIN generations g ON g.id = a.generation_id
            JOIN checkpoints c ON c.id = a.checkpoint_id
            WHERE a.status = 'available' AND g.run_id = ?
            ORDER BY a.id
            """,
            (run_id,),
        ).fetchall()
        candidates: list[dict] = []
        for row in rows:
            if row["model_spec_version"] not in SUPPORTED_MODEL_SPEC_VERSIONS:
                continue
            if row["checkpoint_type"] not in LOADABLE_CHECKPOINT_TYPES:
                continue
            if not self.checkpoints.verify(row["relative_path"], row["sha256"]):
                continue
            candidates.append(dict(row))
        return candidates