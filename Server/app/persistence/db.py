from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from .schema import SCHEMA_MIGRATIONS

SCHEMA_VERSION = max(version for version, _ in SCHEMA_MIGRATIONS)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def connect(db_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def migrate(
    connection: sqlite3.Connection,
    migrations: Sequence[tuple[int, Sequence[str]]] | None = None,
) -> int:
    """Apply each migration and its version marker in one transaction.

    A failure rolls back both the schema statements and the marker, so a
    re-run can safely resume from a clean prior state.
    """
    pending = migrations if migrations is not None else SCHEMA_MIGRATIONS
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS _schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    applied = {row["version"] for row in connection.execute("SELECT version FROM _schema_migrations")}
    for version, statements in pending:
        if version in applied:
            continue
        connection.execute("BEGIN IMMEDIATE")
        try:
            for statement in statements:
                candidate = statement.strip()
                if not candidate:
                    continue
                connection.execute(candidate)
            connection.execute(
                "INSERT INTO _schema_migrations (version, applied_at) VALUES (?, ?)",
                (version, utc_now()),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
    return max(version for version, _ in pending)