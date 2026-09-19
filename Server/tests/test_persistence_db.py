import sqlite3

import pytest

from app.persistence.db import connect, migrate, SCHEMA_VERSION, utc_now
from app.persistence.errors import ControlledRecoveryError
from app.persistence.store import EvolutionStore

EXPECTED_TABLES = {
    "runs",
    "generations",
    "agents",
    "parentage",
    "matches",
    "evaluations",
    "checkpoints",
    "champion_history",
    "_schema_migrations",
}


@pytest.fixture
def store(tmp_path):
    with EvolutionStore(db_path=tmp_path / "evolution.sqlite3", checkpoint_root=tmp_path / "checkpoints") as store:
        yield store


def seed_run_with_agents(store, run_uuid, agent_prefix, count=1) -> tuple[int, int, list[int]]:
    run_id = store.create_run(
        run_uuid=run_uuid,
        config={},
        seed=1,
        code_revision="abc",
        fitness_formula="w",
        benchmark_definition="suite",
    )
    generation_id = store.start_generation(run_id, 0)
    agent_ids = [store.register_agent(f"{agent_prefix}-{index}", generation_id, "initial", "{}") for index in range(count)]
    return run_id, generation_id, agent_ids


def insert_match_raw(connection, run_id, agent_a_id, agent_b_id, match_uuid="raw-match"):
    connection.execute(
        "INSERT INTO matches (match_uuid, run_id, arena, agent_a_id, agent_b_id, opponent_type, game_seed, mode, "
        "score_a, score_b, combos_a, combos_b, duration_steps, result_json, created_at) "
        "VALUES (?, ?, 'raw', ?, ?, NULL, 0, 'raw', 0, 0, 0, 0, 0, '{}', ?)",
        (match_uuid, run_id, agent_a_id, agent_b_id, utc_now()),
    )
    connection.commit()


def test_migrations_create_expected_tables_from_empty_db(store):
    names = {
        row["name"]
        for row in store.connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }
    assert EXPECTED_TABLES.issubset(names)


def test_reopen_does_not_rerun_migrations(tmp_path):
    db_path = tmp_path / "evolution.sqlite3"
    with EvolutionStore(db_path=db_path, checkpoint_root=tmp_path / "checkpoints") as first:
        version = first.connection.execute("SELECT COALESCE(MAX(version), 0) FROM _schema_migrations").fetchone()[0]
    with connect(db_path) as second:
        migrate(second)
        reused = second.execute("SELECT COALESCE(MAX(version), 0) FROM _schema_migrations").fetchone()[0]
    assert version == SCHEMA_VERSION == reused


def test_failed_migration_is_atomic_and_startup_survives(tmp_path):
    db_path = tmp_path / "atomic.sqlite3"
    connection = connect(db_path)
    failing_migrations = [
        (
            99,
            (
                "CREATE TABLE probe_partial (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)",
                "UPDATE missing_table SET payload = 'x'",
            ),
        )
    ]
    with pytest.raises(sqlite3.OperationalError):
        migrate(connection, migrations=failing_migrations)

    version = connection.execute("SELECT COALESCE(MAX(version), 0) FROM _schema_migrations").fetchone()[0]
    assert version == 0
    tables = {row["name"] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert "probe_partial" not in tables

    applied = migrate(connection)
    assert applied == SCHEMA_VERSION
    tables = {row["name"] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert EXPECTED_TABLES.issubset(tables)
    version = connection.execute("SELECT COALESCE(MAX(version), 0) FROM _schema_migrations").fetchone()[0]
    assert version == SCHEMA_VERSION
    connection.close()

    with connect(db_path) as restarted:
        migrate(restarted)
        version = restarted.execute("SELECT COALESCE(MAX(version), 0) FROM _schema_migrations").fetchone()[0]
    assert version == SCHEMA_VERSION


def test_foreign_keys_prevent_dangling_agent_references(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.register_agent("agent-missing-gen", generation_id=999, role="initial", architecture_json="{}")


def test_foreign_keys_prevent_dangling_match_references(store):
    run_id, _, _ = seed_run_with_agents(store, "run-fk", "fk")
    with pytest.raises(sqlite3.IntegrityError):
        insert_match_raw(store.connection, run_id, agent_a_id=404, agent_b_id=405)


def test_matches_reject_cross_run_agents_through_trigger(store):
    run_a, _, agents_a = seed_run_with_agents(store, "run-trig-a", "ta")
    _, _, agents_b = seed_run_with_agents(store, "run-trig-b", "tb")
    with pytest.raises(sqlite3.IntegrityError):
        insert_match_raw(store.connection, run_a, agents_a[0], agents_b[0], match_uuid="raw-cross-run")
    assert store.get_match("raw-cross-run") is None


def test_evaluations_reject_cross_run_agents_through_trigger(store):
    run_a, _, _ = seed_run_with_agents(store, "run-eval-a", "ea")
    _, _, agents_b = seed_run_with_agents(store, "run-eval-b", "eb")
    with pytest.raises(sqlite3.IntegrityError):
        store.connection.execute(
            "INSERT INTO evaluations (agent_id, run_id, benchmark_version, metric_components_json, "
            "sample_count, recorded_at) VALUES (?, ?, 'b1', '{}', 1, ?)",
            (agents_b[0], run_a, utc_now()),
        )
        store.connection.commit()
    count = store.connection.execute("SELECT COUNT(*) AS c FROM evaluations").fetchone()["c"]
    assert count == 0


def test_champion_history_rejects_unrelated_evidence_through_trigger(store):
    run_a, _, agents_a = seed_run_with_agents(store, "run-champ-a", "ca")
    evaluation_id = store.record_evaluation(agents_a[0], run_a, "b1", {"win": 0.5}, 2, fitness=0.5)
    _, _, agents_b = seed_run_with_agents(store, "run-champ-b", "cb")
    with pytest.raises(sqlite3.IntegrityError):
        store.connection.execute(
            "INSERT INTO champion_history (run_id, agent_id, event, evaluation_id, recorded_at) "
            "VALUES (?, ?, 'promoted', ?, ?)",
            (run_a, agents_b[0], evaluation_id, utc_now()),
        )
        store.connection.commit()
    count = store.connection.execute("SELECT COUNT(*) AS c FROM champion_history").fetchone()["c"]
    assert count == 0


def test_matches_reject_cross_run_updates_through_trigger(store):
    run_a, _, agents_a = seed_run_with_agents(store, "run-upd-a", "ua", count=2)
    run_b, _, agents_b = seed_run_with_agents(store, "run-upd-b", "ub")
    store.record_match(
        match_uuid="m-update",
        run_id=run_a,
        arena="arena-0",
        agent_a_id=agents_a[0],
        agent_b_id=agents_a[1],
        game_seed=5,
        mode="selection",
        score_a=1,
        score_b=0,
        combos_a=0,
        combos_b=0,
        duration_steps=20,
        result={},
    )
    with pytest.raises(sqlite3.IntegrityError):
        store.connection.execute("UPDATE matches SET run_id = ? WHERE match_uuid = 'm-update'", (run_b,))
        store.connection.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        store.connection.execute("UPDATE matches SET agent_a_id = ? WHERE match_uuid = 'm-update'", (agents_b[0],))
        store.connection.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        store.connection.execute("UPDATE matches SET agent_b_id = ? WHERE match_uuid = 'm-update'", (agents_b[0],))
        store.connection.rollback()
    row = store.connection.execute(
        "SELECT run_id, agent_a_id, agent_b_id FROM matches WHERE match_uuid = 'm-update'"
    ).fetchone()
    assert row["run_id"] == run_a
    assert row["agent_a_id"] == agents_a[0]
    assert row["agent_b_id"] == agents_a[1]


def test_evaluations_reject_cross_run_updates_through_trigger(store):
    run_a, _, agents_a = seed_run_with_agents(store, "run-upd-eval-a", "uea")
    run_b, _, agents_b = seed_run_with_agents(store, "run-upd-eval-b", "ueb")
    evaluation_id = store.record_evaluation(agents_a[0], run_a, "b1", {"win": 0.5}, 2, fitness=0.5)
    with pytest.raises(sqlite3.IntegrityError):
        store.connection.execute("UPDATE evaluations SET run_id = ? WHERE id = ?", (run_b, evaluation_id))
        store.connection.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        store.connection.execute("UPDATE evaluations SET agent_id = ? WHERE id = ?", (agents_b[0], evaluation_id))
        store.connection.rollback()
    row = store.connection.execute("SELECT run_id, agent_id FROM evaluations WHERE id = ?", (evaluation_id,)).fetchone()
    assert row["run_id"] == run_a
    assert row["agent_id"] == agents_a[0]


def test_champion_history_rejects_unrelated_update_evidence_through_trigger(store):
    run_a, _, agents_a = seed_run_with_agents(store, "run-upd-champ-a", "uca", count=2)
    run_b, _, agents_b = seed_run_with_agents(store, "run-upd-champ-b", "ucb")
    evaluation_a = store.record_evaluation(agents_a[0], run_a, "b1", {"win": 0.5}, 2, fitness=0.5)
    evaluation_b = store.record_evaluation(agents_b[0], run_b, "b1", {"win": 0.4}, 2, fitness=0.4)
    cp_a_path, cp_a_digest = store.checkpoints.write_bytes("uca-0/full.pt", b"weights-a")
    checkpoint_a = store.record_checkpoint(agents_a[0], cp_a_path, cp_a_digest, model_spec_version=2)
    cp_b_path, cp_b_digest = store.checkpoints.write_bytes("ucb-0/full.pt", b"weights-b")
    checkpoint_b = store.record_checkpoint(agents_b[0], cp_b_path, cp_b_digest, model_spec_version=2)
    store.record_champion_event(
        run_id=run_a,
        agent_id=agents_a[0],
        event="promoted",
        evaluation_id=evaluation_a,
        checkpoint_id=checkpoint_a,
    )
    with pytest.raises(sqlite3.IntegrityError):
        store.connection.execute("UPDATE champion_history SET run_id = ?", (run_b,))
        store.connection.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        store.connection.execute("UPDATE champion_history SET agent_id = ?", (agents_b[0],))
        store.connection.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        store.connection.execute("UPDATE champion_history SET evaluation_id = ?", (evaluation_b,))
        store.connection.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        store.connection.execute("UPDATE champion_history SET checkpoint_id = ?", (checkpoint_b,))
        store.connection.rollback()
    row = store.connection.execute(
        "SELECT run_id, agent_id, evaluation_id, checkpoint_id FROM champion_history"
    ).fetchone()
    assert row["run_id"] == run_a
    assert row["agent_id"] == agents_a[0]
    assert row["evaluation_id"] == evaluation_a
    assert row["checkpoint_id"] == checkpoint_a


def test_parentage_reject_cross_run_inserts_through_trigger(store):
    run_a, _, agents_a = seed_run_with_agents(store, "run-pg-a", "pga")
    _, _, agents_b = seed_run_with_agents(store, "run-pg-b", "pgb")
    with pytest.raises(sqlite3.IntegrityError):
        store.connection.execute(
            "INSERT INTO parentage (child_agent_id, parent_agent_id) VALUES (?, ?)",
            (agents_a[0], agents_b[0]),
        )
        store.connection.commit()
    count = store.connection.execute("SELECT COUNT(*) AS c FROM parentage").fetchone()["c"]
    assert count == 0


def test_parentage_reject_cross_run_updates_through_trigger(store):
    run_a, _, agents_a = seed_run_with_agents(store, "run-pgu-a", "pgua", count=2)
    store.record_parentage(child_agent_id=agents_a[1], parent_agent_id=agents_a[0])
    _, _, agents_b = seed_run_with_agents(store, "run-pgu-b", "pgub")
    with pytest.raises(sqlite3.IntegrityError):
        store.connection.execute("UPDATE parentage SET parent_agent_id = ?", (agents_b[0],))
        store.connection.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        store.connection.execute("UPDATE parentage SET child_agent_id = ?", (agents_b[0],))
        store.connection.rollback()
    row = store.connection.execute(
        "SELECT child_agent_id, parent_agent_id FROM parentage"
    ).fetchone()
    assert row["child_agent_id"] == agents_a[1]
    assert row["parent_agent_id"] == agents_a[0]


def test_evaluation_fitness_range_is_enforced(store):
    run_id, generation_id, agent_ids = seed_run_with_agents(store, "run-fitness", "ft")
    with pytest.raises(sqlite3.IntegrityError):
        store.record_evaluation(agent_ids[0], run_id, "bench-v1", {"win": 1.0}, 10, uncertainty=None, fitness=1.5)


def test_checkpoint_path_escaping_root_is_rejected(store):
    outside = str(store.checkpoints.root.parent / "elsewhere.pt")
    with pytest.raises(ControlledRecoveryError):
        store.checkpoints.resolve_owned("../outside-root.pt")
    with pytest.raises(ControlledRecoveryError):
        store.checkpoints.resolve_owned(outside)