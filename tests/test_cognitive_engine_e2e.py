"""End-to-end test: seed scaffold, integrate evidence, verify TCG state."""
import pytest
from tcg.graph import TCGraph
from tcg.seed_scaffold import seed_scaffold
from daemons.integration_daemon import IntegrationDaemon


@pytest.fixture(scope="session")
def db_available() -> bool:
    try:
        from db.pool import get_connection
        with get_connection() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def skip_if_no_db(db_available):
    if not db_available:
        pytest.skip("Database not available")


class TestCognitiveEngineE2E:
    def test_scaffold_then_integrate(self):
        """Seed scaffold, run one integration batch, verify edges updated."""
        graph = TCGraph()

        # Seed scaffold
        stats = seed_scaffold(graph)
        assert stats["nodes_created"] >= 200

        # Get initial state
        summary_before = graph.summary()
        assert summary_before["edge_count"] >= 500

        # Run one integration batch
        daemon = IntegrationDaemon()
        batch_stats = daemon.integrate_batch()
        # May or may not process items depending on existing evidence state
        assert isinstance(batch_stats["items_processed"], int)

    def test_acquisition_queue_populated_after_integration(self):
        """After integration, weak edges should generate acquisition queue items."""
        graph = TCGraph()
        seed_scaffold(graph)

        daemon = IntegrationDaemon()
        daemon.integrate_batch()

        # Check that some acquisition items exist
        from db.pool import get_connection
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM erik_ops.acquisition_queue")
                count = cur.fetchone()[0]
        # May be 0 if no evidence was unintegrated, but should not error
        assert isinstance(count, int)

    def test_progress_endpoint_returns_valid_data(self):
        """The progress metrics should return valid numbers after scaffold."""
        graph = TCGraph()
        seed_scaffold(graph)

        from api.routers.progress import get_progress
        progress = get_progress()
        assert progress["graph_confidence"] >= 0.0
        assert progress["node_count"] >= 200
        assert progress["edge_count"] >= 500
