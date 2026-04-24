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
    _SCRIPTS_DIR / "tcg_schema.sql",
    # Phase 1 SCM Foundation — must run after core_schema (adds scm_edge_id
    # column to erik_core.relationships) and after ops_schema (creates erik_ops).
    _SCRIPTS_DIR / "scm_schema.sql",
    # DrugResponseSimulator (Week 1, 2026-04-24) — depends on scm_schema for
    # the erik_ops schema existing; does not FK to scm_edges (edge_snapshot is
    # captured in JSONB so rollouts stay reproducible across supersession).
    _SCRIPTS_DIR / "simulator_schema.sql",
    # R4 compound-mechanism-refute propagation — depends on scm_write_log
    # (source_write_log_id FK) and scm_edges (refuted_scm_edge_id FK).
    _SCRIPTS_DIR / "propagation_schema.sql",
    # QuantitativeEffectEnricher extension — widens scm_write_log.operation
    # CHECK to include 'effect_updated'.
    _SCRIPTS_DIR / "effect_enricher_schema.sql",
    # Family symptom diary — "how is Erik feeling today?" reports.
    _SCRIPTS_DIR / "symptom_diary_schema.sql",
]

_DB_NAME = "erik_kg"
_DB_USER = os.environ.get("USER", "logannye")


def _make_conninfo() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    return f"dbname={_DB_NAME} user={_DB_USER}"


def _split_sql_statements(sql: str) -> list[str]:
    """Split SQL into top-level statements, preserving $$-delimited bodies.

    Handles:
      * ``--`` line comments (ignored for splitting, preserved in output)
      * ``$$ ... $$`` dollar-quoted plpgsql bodies (internal semicolons kept)
      * Arbitrary dollar-quote tags (``$body$``, ``$func$``, etc.)
      * Single-quoted string literals (internal semicolons ignored)

    A naive ``sql.split(';')`` breaks both plpgsql function bodies and any
    comment or string literal that happens to contain a semicolon.
    """
    import re

    out: list[str] = []
    buf: list[str] = []
    i = 0
    n = len(sql)
    active_tag: str | None = None
    tag_re = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*\$|\$\$")

    while i < n:
        ch = sql[i]

        # Inside a dollar-quoted block — scan for closing tag.
        if active_tag is not None:
            m = tag_re.match(sql, i)
            if m and m.group(0) == active_tag:
                buf.append(m.group(0))
                i = m.end()
                active_tag = None
                continue
            buf.append(ch)
            i += 1
            continue

        # Line comment: consume through end-of-line, keep in buffer.
        if ch == '-' and i + 1 < n and sql[i + 1] == '-':
            eol = sql.find('\n', i)
            if eol == -1:
                eol = n
            buf.append(sql[i:eol])
            i = eol
            continue

        # Single-quoted string literal: consume through closing quote.
        if ch == "'":
            buf.append(ch)
            i += 1
            while i < n:
                buf.append(sql[i])
                if sql[i] == "'" and (i + 1 >= n or sql[i + 1] != "'"):
                    i += 1
                    break
                if sql[i] == "'" and i + 1 < n and sql[i + 1] == "'":
                    buf.append(sql[i + 1])
                    i += 2
                    continue
                i += 1
            continue

        # Dollar-quote opens a plpgsql body.
        m = tag_re.match(sql, i)
        if m:
            active_tag = m.group(0)
            buf.append(m.group(0))
            i = m.end()
            continue

        if ch == ';':
            stmt = ''.join(buf).strip()
            if stmt:
                out.append(stmt)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1

    tail = ''.join(buf).strip()
    if tail:
        out.append(tail)
    return out


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
            statements = _split_sql_statements(sql)
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
