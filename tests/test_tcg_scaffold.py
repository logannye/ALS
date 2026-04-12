"""Tests for ALS biology scaffold seeding."""
import pytest
from tcg.graph import TCGraph
from tcg.seed_scaffold import seed_scaffold, PATHWAY_CLUSTERS


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


@pytest.fixture
def graph():
    return TCGraph()


class TestScaffoldConstants:
    def test_eight_pathway_clusters(self):
        assert len(PATHWAY_CLUSTERS) == 8
        assert "proteostasis" in PATHWAY_CLUSTERS
        assert "rna_metabolism" in PATHWAY_CLUSTERS
        assert "excitotoxicity" in PATHWAY_CLUSTERS
        assert "neuroinflammation" in PATHWAY_CLUSTERS
        assert "axonal_transport" in PATHWAY_CLUSTERS
        assert "mitochondrial" in PATHWAY_CLUSTERS
        assert "neuromuscular_junction" in PATHWAY_CLUSTERS
        assert "glial_biology" in PATHWAY_CLUSTERS


class TestScaffoldSeeding:
    def test_seed_creates_nodes(self, graph):
        stats = seed_scaffold(graph)
        assert stats["nodes_created"] >= 200
        assert stats["edges_created"] >= 500

    def test_seed_is_idempotent(self, graph):
        stats1 = seed_scaffold(graph)
        stats2 = seed_scaffold(graph)
        # Second run uses ON CONFLICT upsert — same counts
        assert stats2["nodes_created"] == stats1["nodes_created"]

    def test_all_clusters_populated(self, graph):
        seed_scaffold(graph)
        summary = graph.summary()
        for cluster in PATHWAY_CLUSTERS:
            assert summary["clusters"].get(cluster, 0) >= 5, \
                f"Cluster {cluster} has fewer than 5 nodes"

    def test_edges_have_open_questions(self, graph):
        seed_scaffold(graph)
        weak = graph.get_weakest_edges(limit=50)
        with_questions = [e for e in weak if e.open_questions]
        assert len(with_questions) >= 20, \
            "At least 20 weak edges should have open questions"

    def test_confidence_ranges(self, graph):
        seed_scaffold(graph)
        from db.pool import get_connection
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT min(confidence), max(confidence) FROM erik_core.tcg_edges")
                row = cur.fetchone()
        assert row[0] >= 0.1, "Minimum confidence should be >= 0.1"
        assert row[1] <= 0.95, "Maximum confidence should be <= 0.95"
