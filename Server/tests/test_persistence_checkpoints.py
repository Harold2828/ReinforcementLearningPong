import pytest

from app.persistence.errors import ControlledRecoveryError
from app.persistence.store import EvolutionStore


def make_run_generation_and_agent(store) -> tuple[int, int, int]:
    run_id = store.create_run(
        run_uuid="run-recovery",
        config={},
        seed=11,
        code_revision="abc",
        fitness_formula="w",
        benchmark_definition="suite",
    )
    generation_id = store.start_generation(run_id, 0)
    agent_id = store.register_agent("agent-recovery", generation_id, "initial", "{}")
    return run_id, generation_id, agent_id


def linked_checkpoint(
    store,
    agent_id,
    relative_name="agents/full.pt",
    payload=b"weights",
    checkpoint_type="full",
    model_spec_version=2,
):
    relative_path, digest = store.checkpoints.write_bytes(relative_name, payload)
    checkpoint_id = store.record_checkpoint(
        agent_id, relative_path, digest, model_spec_version=model_spec_version, checkpoint_type=checkpoint_type
    )
    return checkpoint_id, (relative_path, digest)


def make_valid_agent(store, run_uuid, prefix):
    run_id = store.create_run(
        run_uuid=run_uuid,
        config={},
        seed=5,
        code_revision="abc",
        fitness_formula="w",
        benchmark_definition="suite",
    )
    generation_id = store.start_generation(run_id, 0)
    agent_id = store.register_agent(prefix, generation_id, "initial", "{}")
    linked_checkpoint(store, agent_id, relative_name=f"{prefix}/full.pt")
    return run_id, agent_id


def test_atomic_write_persists_payload_and_reports_matching_hash(tmp_path):
    with EvolutionStore(db_path=tmp_path / "db.sqlite3", checkpoint_root=tmp_path / "checkpoints") as store:
        relative_path, digest = store.checkpoints.write_bytes("agent-0/weights.pt", b"pong-weights")
        resolved, actual = store.checkpoints.read_verified(relative_path, digest)
        assert resolved.read_bytes() == b"pong-weights"
        assert actual == digest
        leftovers = [path for path in store.checkpoints.root.rglob("*") if path.is_file() and path.name.endswith(".tmp")]
        assert leftovers == []


def test_missing_checkpoint_marks_agent_unavailable_and_never_champion_candidate(tmp_path):
    with EvolutionStore(db_path=tmp_path / "db.sqlite3", checkpoint_root=tmp_path / "checkpoints") as store:
        run_id, generation_id, agent_id = make_run_generation_and_agent(store)
        _, (relative_path, digest) = linked_checkpoint(store, agent_id)
        store.checkpoints.resolve_owned(relative_path).unlink()

        with pytest.raises(ControlledRecoveryError) as caught:
            store.reconcile_checkpoints()
        assert caught.value.report["missing"]
        assert caught.value.report["corrupt"] == []
        assert agent_id in caught.value.report["unavailable_agents"]
        assert store.get_agent(agent_id)["status"] == "unavailable"
        assert store.available_champion_candidates(run_id) == []


def test_corrupt_checkpoint_marks_agent_unavailable_and_never_champion_candidate(tmp_path):
    with EvolutionStore(db_path=tmp_path / "db.sqlite3", checkpoint_root=tmp_path / "checkpoints") as store:
        run_id, generation_id, agent_id = make_run_generation_and_agent(store)
        _, (relative_path, digest) = linked_checkpoint(store, agent_id)
        store.checkpoints.resolve_owned(relative_path).write_bytes(b"tampered-weights")

        with pytest.raises(ControlledRecoveryError) as caught:
            store.reconcile_checkpoints()
        assert caught.value.report["corrupt"]
        assert caught.value.report["missing"] == []
        assert agent_id in caught.value.report["unavailable_agents"]
        assert store.get_agent(agent_id)["status"] == "unavailable"
        assert store.available_champion_candidates(run_id) == []


def test_reconcile_is_idempotent(tmp_path):
    with EvolutionStore(db_path=tmp_path / "db.sqlite3", checkpoint_root=tmp_path / "checkpoints") as store:
        run_id, generation_id, agent_id = make_run_generation_and_agent(store)
        _, (relative_path, digest) = linked_checkpoint(store, agent_id)
        store.checkpoints.resolve_owned(relative_path).write_bytes(b"tampered")

        with pytest.raises(ControlledRecoveryError) as first:
            store.reconcile_checkpoints()
        with pytest.raises(ControlledRecoveryError) as second:
            store.reconcile_checkpoints()
        assert second.value.report == first.value.report
        assert store.get_agent(agent_id)["status"] == "unavailable"


def test_champion_candidates_only_include_verified_checkpoints(tmp_path):
    with EvolutionStore(db_path=tmp_path / "db.sqlite3", checkpoint_root=tmp_path / "checkpoints") as store:
        run_id, generation_id, valid_agent = make_run_generation_and_agent(store)
        linked_checkpoint(store, valid_agent, relative_name="agents/valid.pt")
        corrupt_agent = store.register_agent("agent-corrupt", generation_id, "initial", "{}")
        _, (relative_path, digest) = linked_checkpoint(store, corrupt_agent, relative_name="agents/corrupt.pt")
        store.checkpoints.resolve_owned(relative_path).write_bytes(b"tampered")

        try:
            store.reconcile_checkpoints()
        except ControlledRecoveryError:
            pass

        candidates = store.available_champion_candidates(run_id)
        assert [candidate["agent_id"] for candidate in candidates] == [valid_agent]
        assert all(candidate["checkpoint_id"] for candidate in candidates)


def test_champion_candidates_exclude_agents_from_other_runs(tmp_path):
    with EvolutionStore(db_path=tmp_path / "db.sqlite3", checkpoint_root=tmp_path / "checkpoints") as store:
        run_a, agent_a = make_valid_agent(store, "run-cand-a", "other-run-agent")
        _, _ = make_valid_agent(store, "run-cand-b", "agent-b")

        candidates = store.available_champion_candidates(run_a)
        assert [candidate["agent_id"] for candidate in candidates] == [agent_a]


def test_champion_candidates_exclude_optimizer_only_checkpoints(tmp_path):
    with EvolutionStore(db_path=tmp_path / "db.sqlite3", checkpoint_root=tmp_path / "checkpoints") as store:
        run_id, generation_id, agent_id = make_run_generation_and_agent(store)
        linked_checkpoint(store, agent_id, relative_name="agents/optimizer.pt", checkpoint_type="optimizer")

        assert store.available_champion_candidates(run_id) == []


def test_champion_candidates_exclude_unsupported_model_versions(tmp_path):
    with EvolutionStore(db_path=tmp_path / "db.sqlite3", checkpoint_root=tmp_path / "checkpoints") as store:
        run_id, generation_id, agent_id = make_run_generation_and_agent(store)
        linked_checkpoint(store, agent_id, relative_name="agents/future.pt", model_spec_version=999)

        assert store.available_champion_candidates(run_id) == []


def test_champion_candidates_exclude_missing_and_corrupt_files(tmp_path):
    with EvolutionStore(db_path=tmp_path / "db.sqlite3", checkpoint_root=tmp_path / "checkpoints") as store:
        run_id, generation_id, agent_id = make_run_generation_and_agent(store)
        linked_checkpoint(store, agent_id, relative_name="agents/good.pt")

        second_agent = store.register_agent("agent-b", generation_id, "initial", "{}")
        _, (missing_path, missing_digest) = linked_checkpoint(store, second_agent, relative_name="agents/missing.pt")
        store.checkpoints.resolve_owned(missing_path).unlink()

        third_agent = store.register_agent("agent-c", generation_id, "initial", "{}")
        _, (corrupt_path, corrupt_digest) = linked_checkpoint(store, third_agent, relative_name="agents/tampered.pt")
        store.checkpoints.resolve_owned(corrupt_path).write_bytes(b"tampered")

        candidates = store.available_champion_candidates(run_id)
        assert [candidate["agent_id"] for candidate in candidates] == [agent_id]


def test_orphan_files_are_reported_and_optionally_removed(tmp_path):
    with EvolutionStore(db_path=tmp_path / "db.sqlite3", checkpoint_root=tmp_path / "checkpoints") as store:
        run_id, generation_id, agent_id = make_run_generation_and_agent(store)
        linked_checkpoint(store, agent_id, relative_name="agents/kept.pt", payload=b"weights")
        orphan_path, _ = store.checkpoints.write_bytes("agents/orphan.pt", b"unreferenced")

        report = store.reconcile_checkpoints()
        assert "agents/orphan.pt" in report["orphans"]
        assert store.checkpoints.resolve_owned(orphan_path).is_file()

        report_cleaned = store.reconcile_checkpoints(remove_orphans=True)
        assert orphan_path in report_cleaned["orphans"]
        assert not store.checkpoints.resolve_owned(orphan_path).exists()
        assert "agents/orphan.pt" not in store.reconcile_checkpoints()["orphans"]