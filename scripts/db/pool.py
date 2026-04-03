"""PostgreSQL connection pool for the Erik ALS engine.

Connects via Unix socket using the $USER environment variable.
Database: erik_kg
Pool: min_size=1, max_size=5
"""

import os
from contextlib import contextmanager
from typing import Generator

import psycopg
from psycopg_pool import ConnectionPool

_pool: ConnectionPool | None = None

_DB_NAME = "erik_kg"
_DB_USER = os.environ.get("USER", "logannye")


def _make_conninfo() -> str:
    """Build connection string.  Prefers DATABASE_URL (Railway, remote) if set."""
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    return f"dbname={_DB_NAME} user={_DB_USER}"


def get_pool() -> ConnectionPool:
    """Return the shared connection pool, creating it on first call."""
    global _pool
    if _pool is None or _pool.closed:
        _pool = ConnectionPool(
            conninfo=_make_conninfo(),
            min_size=1,
            max_size=5,
            open=True,
        )
    return _pool


@contextmanager
def get_connection() -> Generator[psycopg.Connection, None, None]:
    """Context manager that yields a checked-out connection from the pool."""
    pool = get_pool()
    with pool.connection() as conn:
        yield conn


def close_pool() -> None:
    """Close the shared pool. Call during application shutdown."""
    global _pool
    if _pool is not None and not _pool.closed:
        _pool.close()
    _pool = None
