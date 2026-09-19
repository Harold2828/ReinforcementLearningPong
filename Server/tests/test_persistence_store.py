import sqlite3

import pytest

from app.persistence.errors import ProvenanceViolationError
from app.persistence.store import EvolutionStore


def make_run_and_generation(store) -> tuple[int, int]:
    run_id = store.create_run(
        run_uuid="run-001",
        config={"population": 10, "arenas": 5},
        seed=42,
        code_revision="60e819b",
        fitness_formula="0.65*W + 0.25*D + 0.10*C",
        benchmark_definition="bench-v1: scripted predictive controller",
    )
    generation_id = store.start_generation(run_id, 0)
    return run_id, generation_id


def make_agents(store, generation_id, count=2, prefix="agent") -> list[int]:
    return [
        store.register_agent(f"{prefix}-{index}", generation_id, "initial", '{"hiddenWidths":[128,128]}')
        for index in range(count)
    ]


def seed_run(store, run_uuid, prefix, count=1) -> tuple[int, int, list[int]]:
    run_id = store.create_run(
        run_uuid=run_uuid,
        config={},
        seed=99,
        code_revision="abc",
        fitness_formula="w",
        benchmark_definition="suite",
    )
    generation_id = store.start_generation(run_id, 0)
    agent_ids = make_agents(store, generation_id, count=count, prefix=prefix)
    return run_id, generation_id, agent_ids


def test_run_and_generation_survive_restart(tmp_path):
    db_path = tmp_path / "evolution.sqlite3"
    root = tmp_path / "checkpoints"
    with EvolutionStore(db_path=db_path, checkpoint_root=root) as store:
        run_id, generation_id = make_run_and_generation(store)
        make_agents(store, generation_id)
        assert store.complete_generation(generation_id)

    with EvolutionStore(db_path=db_path, checkpoint_root=root) as reopened:
        run = reopened.get_run(run_id)
        generations = reopened.list_generations(run_id)
        assert run is not None
        assert run["status"] == "running"
        assert run["seed"] == 42
        assert len(generations) == 1
        assert generations[0]["status"] == "completed"
        assert generations[0]["completed_at"] is not None
        assert len(reopened.list_agents(generation_id)) == 2


def test_completed_match_submitted_twice_is_stored_once(tmp_path):
    with EvolutionStore(db_path=tmp_path / "evolution.sqlite3", checkpoint_root=tmp_path / "checkpoints") as store:
        run_id, generation_id = make_run_and_generation(store)
        agent_a, agent_b = make_agents(store, generation_id)
        kwargs = dict(
            run_id=run_id,
            arena="arena-0",
            agent_a_id=agent_a,
            agent_b_id=agent_b,
            game_seed=7,
            mode="selection",
            score_a=2,
            score_b=1,
            combos_a=3,
            combos_b=0,
            duration_steps=521,
            result={"events": ["point-a", "point-a", "point-b"]},
        )
        assert store.record_match(match_uuid="match-stable-1", **kwargs) is True
        assert store.record_match(match_uuid="match-stable-1", **kwargs) is False

        count = store.connection.execute("SELECT COUNT(*) AS c FROM matches").fetchone()["c"]
        assert count == 1
        assert store.get_match("match-stable-1")["score_a"] == 2


def test_generation_closeout_is_idempotent(tmp_path):
    with EvolutionStore(db_path=tmp_path / "evolution.sqlite3", checkpoint_root=tmp_path / "checkpoints") as store:
        run_id, generation_id = make_run_and_generation(store)
        assert store.complete_generation(generation_id) is True
        assert store.complete_generation(generation_id) is False
        assert store.complete_generation(generation_id) is False


def test_checkpoint_reference_persists_and_links_agent(tmp_path):
    with EvolutionStore(db_path=tmp_path / "evolution.sqlite3", checkpoint_root=tmp_path / "checkpoints") as store:
        run_id, generation_id = make_run_and_generation(store)
        (agent_id,) = make_agents(store, generation_id, count=1)
        relative_path, digest = store.checkpoints.write_bytes("agents/agent-0/full.pt", b"tensor-payload")
        checkpoint_id = store.record_checkpoint(agent_id, relative_path, digest, model_spec_version=2)

    with EvolutionStore(db_path=tmp_path / "evolution.sqlite3", checkpoint_root=tmp_path / "checkpoints") as reopened:
        agent = reopened.get_agent(agent_id)
        checkpoint = reopened.get_checkpoint(checkpoint_id)
        assert agent["checkpoint_id"] == checkpoint_id
        assert checkpoint["sha256"] == digest
        assert checkpoint["relative_path"] == relative_path
        assert checkpoint["model_spec_version"] == 2
        resolved, actual_digest = reopened.checkpoints.read_verified(relative_path, digest)
        assert resolved.is_file()
        assert actual_digest == digest


def test_parentage_evaluation_and_champion_history_round_trip(tmp_path):
    with EvolutionStore(db_path=tmp_path / "evolution.sqlite3", checkpoint_root=tmp_path / "checkpoints") as store:
        run_id, generation_id = make_run_and_generation(store)
        parent, child = make_agents(store, generation_id)

        store.record_parentage(child_agent_id=child, parent_agent_id=parent, mutation_json='{"hiddenWidths":[192,128]}')
        evaluation_id = store.record_evaluation(
            agent_id=child,
            run_id=run_id,
            benchmark_version="bench-v1",
            metric_components={"win": 0.8, "point_diff": 0.6, "combo": 0.5},
            sample_count=45,
            uncertainty={"se": 0.03},
            fitness=0.65 * 0.8 + 0.25 * 0.6 + 0.10 * 0.5,
        )
        store.record_champion_event(
            run_id=run_id,
            agent_id=child,
            event="promoted",
            evaluation_id=evaluation_id,
        )

    with EvolutionStore(db_path=tmp_path / "evolution.sqlite3", checkpoint_root=tmp_path / "checkpoints") as reopened:
        parentage = reopened.connection.execute(
            "SELECT * FROM parentage WHERE child_agent_id = ?", (child,)
        ).fetchall()
        evaluation = reopened.connection.execute(
            "SELECT fitness FROM evaluations WHERE id = ?", (evaluation_id,)
        ).fetchone()
        events = reopened.list_champion_events(run_id)
        assert len(parentage) == 1
        assert parentage[0]["parent_agent_id"] == parent
        assert abs(evaluation["fitness"] - 0.8 * 0.65 - 0.6 * 0.25 - 0.5 * 0.10) < 1e-9
        assert len(events) == 1
        assert events[0]["event"] == "promoted"


def test_cross_run_parentage_is_rejected(tmp_path):
    with EvolutionStore(db_path=tmp_path / "evolution.sqlite3", checkpoint_root=tmp_path / "checkpoints") as store:
        run_a, _, agents_a = seed_run(store, "run-pa", "pa")
        _, _, agents_b = seed_run(store, "run-pb", "pb")
        with pytest.raises(ProvenanceViolationError):
            store.record_parentage(child_agent_id=agents_a[0], parent_agent_id=agents_b[0])
        with pytest.raises(ProvenanceViolationError) as caught:
            store.record_parentage(child_agent_id=agents_a[0], parent_agent_id=424242)
        assert "does not exist" in str(caught.value)
        count = store.connection.execute("SELECT COUNT(*) AS c FROM parentage").fetchone()["c"]
        assert count == 0


def test_parentage_allows_parents_from_earlier_generations(tmp_path):
    with EvolutionStore(db_path=tmp_path / "evolution.sqlite3", checkpoint_root=tmp_path / "checkpoints") as store:
        run_id, generation_zero = make_run_and_generation(store)
        (parent,) = make_agents(store, generation_zero, count=1, prefix="parent")
        generation_one = store.start_generation(run_id, 1)
        (child,) = make_agents(store, generation_one, count=1, prefix="child")

        parentage_id = store.record_parentage(child_agent_id=child, parent_agent_id=parent)

        row = store.connection.execute(
            "SELECT child_agent_id, parent_agent_id FROM parentage WHERE id = ?", (parentage_id,)
        ).fetchone()
        assert row["child_agent_id"] == child
        assert row["parent_agent_id"] == parent


def test_cross_run_match_is_rejected(tmp_path):
    with EvolutionStore(db_path=tmp_path / "evolution.sqlite3", checkpoint_root=tmp_path / "checkpoints") as store:
        run_a, generation_id, agents_a = seed_run(store, "run-a", "aa")
        _, _, agents_b = seed_run(store, "run-b", "bb")
        with pytest.raises(ProvenanceViolationError):
            store.record_match(
                match_uuid="m-cross",
                run_id=run_a,
                arena="arena-0",
                agent_a_id=agents_a[0],
                agent_b_id=agents_b[0],
                game_seed=1,
                mode="selection",
                score_a=0,
                score_b=0,
                combos_a=0,
                combos_b=0,
                duration_steps=10,
                result={},
            )
        count = store.connection.execute("SELECT COUNT(*) AS c FROM matches").fetchone()["c"]
        assert count == 0


def test_evaluation_must_match_agent_run(tmp_path):
    with EvolutionStore(db_path=tmp_path / "evolution.sqlite3", checkpoint_root=tmp_path / "checkpoints") as store:
        run_a, _, agents_a = seed_run(store, "run-a", "aa")
        run_b, _, _ = seed_run(store, "run-b", "bb")
        with pytest.raises(ProvenanceViolationError):
            store.record_evaluation(agents_a[0], run_b, "bench-v1", {"win": 0.5}, 10, fitness=0.5)
        count = store.connection.execute("SELECT COUNT(*) AS c FROM evaluations").fetchone()["c"]
        assert count == 0


def test_champion_event_rejects_unrelated_evidence(tmp_path):
    with EvolutionStore(db_path=tmp_path / "evolution.sqlite3", checkpoint_root=tmp_path / "checkpoints") as store:
        run_a, generation_id, agents_a = seed_run(store, "run-a", "aa", count=2)
        agent_a, agent_b = agents_a
        evaluation = store.record_evaluation(agent_a, run_a, "bench-v1", {"win": 0.5}, 10, fitness=0.5)
        relative_path, digest = store.checkpoints.write_bytes("agents/aa-1/full.pt", b"weights")
        checkpoint_b = store.record_checkpoint(agent_b, relative_path, digest, model_spec_version=2)

        with pytest.raises(ProvenanceViolationError):
            store.record_champion_event(run_a, agent_b, "promoted", evaluation_id=evaluation)
        with pytest.raises(ProvenanceViolationError):
            store.record_champion_event(run_a, agent_a, "promoted", checkpoint_id=checkpoint_b)
        with pytest.raises(ProvenanceViolationError) as caught:
            store.record_champion_event(run_a, agent_a, "promoted", evaluation_id=424242)
        assert "does not exist" in str(caught.value)

        count = store.connection.execute("SELECT COUNT(*) AS c FROM champion_history").fetchone()["c"]
        assert count == 0


def test_raw_provenance_cannot_be_silently_cascade_deleted(tmp_path):
    with EvolutionStore(db_path=tmp_path / "evolution.sqlite3", checkpoint_root=tmp_path / "checkpoints") as store:
        run_id, generation_id = make_run_and_generation(store)
        parent, child = make_agents(store, generation_id)
        store.record_parentage(child_agent_id=child, parent_agent_id=parent)
        match = store.record_match(
            match_uuid="m-keep",
            run_id=run_id,
            arena="arena-0",
            agent_a_id=parent,
            agent_b_id=child,
            game_seed=3,
            mode="selection",
            score_a=1,
            score_b=1,
            combos_a=0,
            combos_b=0,
            duration_steps=30,
            result={},
        )
        evaluation = store.record_evaluation(child, run_id, "bench-v1", {"win": 1.0}, 5, fitness=0.65)
        store.record_champion_event(
            run_id, child, "promoted", evaluation_id=evaluation,
        )

        with pytest.raises(sqlite3.IntegrityError):
            store.connection.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            store.connection.commit()
        with pytest.raises(sqlite3.IntegrityError):
            store.connection.execute("DELETE FROM agents WHERE id = ?", (parent,))
            store.connection.commit()
        with pytest.raises(sqlite3.IntegrityError):
            store.connection.execute("DELETE FROM agents WHERE id = ?", (child,))
            store.connection.commit()

        assert store.get_match("m-keep")["match_uuid"] == "m-keep"
        assert store.get_agent(parent) is not None
        assert store.get_agent(child) is not None
        assert len(store.list_champion_events(run_id)) == 1


def test_soft_deletion_is_supported(tmp_path):
    with EvolutionStore(db_path=tmp_path / "evolution.sqlite3", checkpoint_root=tmp_path / "checkpoints") as store:
        run_id, generation_id = make_run_and_generation(store)
        (agent_id,) = make_agents(store, generation_id, count=1)
        assert store.set_agent_status(agent_id, "retired") is True
        assert store.get_agent(agent_id)["status"] == "retired"
        assert store.set_run_status(run_id, "cancelled") is True
        assert store.get_run(run_id)["status"] == "cancelled"