"""Shared pytest fixtures for the Erik test suite."""

import pytest
import psycopg


def _can_connect() -> bool:
    """Return True if the erik_kg database is reachable."""
    import os
    user = os.environ.get("USER", "logannye")
    try:
        conn = psycopg.connect(f"dbname=erik_kg user={user}", connect_timeout=3)
        conn.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def db_available() -> bool:
    """Session-scoped fixture: skip the test if the DB is not accessible."""
    available = _can_connect()
    if not available:
        pytest.skip("erik_kg PostgreSQL database not accessible — skipping DB tests")
    return True
