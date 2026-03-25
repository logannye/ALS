"""Tests for Task 9: PostgreSQL connection pool."""

import pytest


class TestPoolModule:
    def test_pool_module_importable(self):
        from db import pool  # noqa: F401

    def test_get_pool_function_exists(self):
        from db.pool import get_pool
        assert callable(get_pool)

    def test_get_connection_function_exists(self):
        from db.pool import get_connection
        assert callable(get_connection)

    def test_close_pool_function_exists(self):
        from db.pool import close_pool
        assert callable(close_pool)


class TestPoolLive:
    """Live DB tests — require db_available fixture."""

    def test_select_1_via_pool(self, db_available):
        from db.pool import get_connection, close_pool
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT 1")
                result = cur.fetchone()
                assert result[0] == 1
        finally:
            close_pool()

    def test_pool_is_reusable(self, db_available):
        """Pool can be obtained and used multiple times."""
        from db.pool import get_connection, close_pool
        try:
            for _ in range(3):
                with get_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT 42")
                    assert cur.fetchone()[0] == 42
        finally:
            close_pool()

    def test_pool_min_max_sizes(self, db_available):
        """Pool is created with expected min/max sizes."""
        from db.pool import get_pool, close_pool
        try:
            pool = get_pool()
            assert pool.min_size == 1
            assert pool.max_size == 5
        finally:
            close_pool()
