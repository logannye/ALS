"""Tests for Task 9: PostgreSQL schema files and migration idempotency."""

import pathlib
import pytest

_SCRIPTS_DIR = pathlib.Path(__file__).parent.parent / "scripts" / "db"


class TestSQLFilesExist:
    def test_core_schema_sql_exists(self):
        assert (_SCRIPTS_DIR / "core_schema.sql").is_file()

    def test_ops_schema_sql_exists(self):
        assert (_SCRIPTS_DIR / "ops_schema.sql").is_file()


class TestCoreSchemaContents:
    @pytest.fixture(autouse=True)
    def _read(self):
        self.sql = (_SCRIPTS_DIR / "core_schema.sql").read_text()

    def test_contains_objects_table(self):
        assert "erik_core.objects" in self.sql

    def test_contains_entities_table(self):
        assert "erik_core.entities" in self.sql

    def test_contains_relationships_table(self):
        assert "erik_core.relationships" in self.sql

    def test_contains_embeddings_table(self):
        assert "erik_core.embeddings" in self.sql

    def test_uses_if_not_exists(self):
        assert "IF NOT EXISTS" in self.sql

    def test_schema_creation(self):
        assert "CREATE SCHEMA IF NOT EXISTS erik_core" in self.sql

    def test_objects_has_body_jsonb(self):
        assert "body JSONB" in self.sql

    def test_entities_has_pch_layer(self):
        assert "pch_layer" in self.sql

    def test_relationships_has_references(self):
        assert "REFERENCES erik_core.entities" in self.sql


class TestOpsSchemaContents:
    @pytest.fixture(autouse=True)
    def _read(self):
        self.sql = (_SCRIPTS_DIR / "ops_schema.sql").read_text()

    def test_contains_audit_events_table(self):
        assert "erik_ops.audit_events" in self.sql

    def test_contains_config_snapshots_table(self):
        assert "erik_ops.config_snapshots" in self.sql

    def test_schema_creation(self):
        assert "CREATE SCHEMA IF NOT EXISTS erik_ops" in self.sql

    def test_audit_events_has_details_jsonb(self):
        assert "details JSONB" in self.sql

    def test_config_snapshots_has_config_jsonb(self):
        assert "config JSONB" in self.sql


class TestMigrationsIdempotent:
    """Run migrations twice to confirm IF NOT EXISTS guards work."""

    def test_migrations_are_idempotent(self, db_available):
        from db.migrate import run_migrations
        # First run (tables already exist from setup)
        run_migrations()
        # Second run — must not raise
        run_migrations()


class TestSchemaTablesExist:
    """Verify expected tables are present in the live database."""

    def test_erik_core_tables_exist(self, db_available):
        import psycopg, os
        user = os.environ.get("USER", "logannye")
        conn = psycopg.connect(f"dbname=erik_kg user={user}")
        cur = conn.cursor()
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'erik_core' ORDER BY table_name"
        )
        tables = {row[0] for row in cur.fetchall()}
        conn.close()
        assert "objects" in tables
        assert "entities" in tables
        assert "relationships" in tables

    def test_erik_ops_tables_exist(self, db_available):
        import psycopg, os
        user = os.environ.get("USER", "logannye")
        conn = psycopg.connect(f"dbname=erik_kg user={user}")
        cur = conn.cursor()
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'erik_ops' ORDER BY table_name"
        )
        tables = {row[0] for row in cur.fetchall()}
        conn.close()
        assert "audit_events" in tables
        assert "config_snapshots" in tables
