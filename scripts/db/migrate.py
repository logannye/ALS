"""Database migration runner for the Erik ALS engine.

Reads and executes core_schema.sql and ops_schema.sql.
Safe to run multiple times (all statements are IF NOT EXISTS).

Usage:
    python -m db.migrate
"""

import os
import pathlib
import psycopg

_SCRIPTS_DIR = pathlib.Path(__file__).parent
_SCHEMA_FILES = [
    _SCRIPTS_DIR / "core_schema.sql",
    _SCRIPTS_DIR / "ops_schema.sql",
    _SCRIPTS_DIR / "trial_watchlist.sql",
]

_DB_NAME = "erik_kg"
_DB_USER = os.environ.get("USER", "logannye")


def _make_conninfo() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    return f"dbname={_DB_NAME} user={_DB_USER}"


def run_migrations(conninfo: str | None = None) -> None:
    """Execute all schema SQL files against the target database.

    Args:
        conninfo: Optional psycopg conninfo string. Defaults to the standard
                  erik_kg connection string.
    """
    if conninfo is None:
        conninfo = _make_conninfo()

    # Use autocommit=True so that each statement is its own transaction.
    # This means a missing optional extension (e.g. pgvector) does not roll
    # back unrelated DDL that ran earlier in the same file.
    conn = psycopg.connect(conninfo, autocommit=True)
    try:
        for sql_path in _SCHEMA_FILES:
            sql = sql_path.read_text(encoding="utf-8")
            # Split on semicolons to handle extensions that must run
            # before tables that depend on them (e.g. citext before CITEXT columns).
            statements = [s.strip() for s in sql.split(";") if s.strip()]
            for stmt in statements:
                try:
                    conn.execute(stmt)
                except (
                    psycopg.errors.FeatureNotSupported,
                    psycopg.errors.UndefinedFile,
                    psycopg.errors.UndefinedObject,
                ) as exc:
                    # Extension not installed on this system (e.g. pgvector) — warn and continue.
                    print(f"WARNING: optional feature unavailable — {exc}")
                except Exception:
                    raise
        print("Migrations complete.")
    finally:
        conn.close()


if __name__ == "__main__":
    run_migrations()
