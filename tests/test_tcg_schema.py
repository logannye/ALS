"""Tests for TCG database schema."""
import pytest
from db.pool import get_connection


@pytest.fixture(scope="session")
def db_available() -> bool:
    try:
        with get_connection() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def skip_if_no_db(db_available):
    if not db_available:
        pytest.skip("Database not available")


class TestTCGSchema:
    def test_tcg_nodes_table_exists(self):
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'erik_core' AND table_name = 'tcg_nodes'
                    ORDER BY ordinal_position
                """)
                columns = [row[0] for row in cur.fetchall()]
        assert "id" in columns
        assert "entity_type" in columns
        assert "name" in columns
        assert "pathway_cluster" in columns
        assert "druggability_score" in columns
        assert "metadata" in columns

    def test_tcg_edges_table_exists(self):
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'erik_core' AND table_name = 'tcg_edges'
                    ORDER BY ordinal_position
                """)
                columns = [row[0] for row in cur.fetchall()]
        assert "id" in columns
        assert "source_id" in columns
        assert "target_id" in columns
        assert "edge_type" in columns
        assert "confidence" in columns
        assert "evidence_ids" in columns
        assert "open_questions" in columns

    def test_tcg_hypotheses_table_exists(self):
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'erik_core' AND table_name = 'tcg_hypotheses'
                    ORDER BY ordinal_position
                """)
                columns = [row[0] for row in cur.fetchall()]
        assert "id" in columns
        assert "hypothesis" in columns
        assert "status" in columns
        assert "therapeutic_relevance" in columns

    def test_acquisition_queue_table_exists(self):
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'erik_ops' AND table_name = 'acquisition_queue'
                    ORDER BY ordinal_position
                """)
                columns = [row[0] for row in cur.fetchall()]
        assert "id" in columns
        assert "tcg_edge_id" in columns
        assert "open_question" in columns
        assert "priority" in columns

    def test_activity_feed_table_exists(self):
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'erik_ops' AND table_name = 'activity_feed'
                    ORDER BY ordinal_position
                """)
                columns = [row[0] for row in cur.fetchall()]
        assert "phase" in columns
        assert "event_type" in columns
        assert "summary" in columns

    def test_llm_spend_table_exists(self):
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'erik_ops' AND table_name = 'llm_spend'
                    ORDER BY ordinal_position
                """)
                columns = [row[0] for row in cur.fetchall()]
        assert "model" in columns
        assert "phase" in columns
        assert "cost_usd" in columns

    def test_objects_has_tcg_integrated_column(self):
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'erik_core' AND table_name = 'objects'
                    AND column_name = 'tcg_integrated'
                """)
                assert cur.fetchone() is not None
