# tests/test_tcg_graph.py
"""Tests for TCG graph read/write operations."""
import pytest
from tcg.models import TCGNode, TCGEdge, TCGHypothesis, AcquisitionItem
from tcg.graph import TCGraph


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


@pytest.fixture
def sample_nodes():
    return [
        TCGNode(id="gene:tardbp_test", entity_type="gene", name="TARDBP", pathway_cluster="proteostasis"),
        TCGNode(id="protein:tdp43_test", entity_type="protein", name="TDP-43", pathway_cluster="proteostasis"),
        TCGNode(id="gene:stmn2_test", entity_type="gene", name="STMN2", pathway_cluster="rna_metabolism"),
    ]


@pytest.fixture
def cleanup_test_nodes(graph):
    """Remove test nodes after each test."""
    yield
    from db.pool import get_connection
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM erik_ops.acquisition_queue WHERE tcg_edge_id LIKE '%_test%'")
            cur.execute("DELETE FROM erik_core.tcg_hypotheses WHERE id LIKE '%_test%'")
            cur.execute("DELETE FROM erik_core.tcg_edges WHERE id LIKE '%_test%'")
            cur.execute("DELETE FROM erik_core.tcg_nodes WHERE id LIKE '%_test%'")
        conn.commit()


class TestNodeCRUD:
    def test_upsert_and_get_node(self, graph, sample_nodes, cleanup_test_nodes):
        graph.upsert_node(sample_nodes[0])
        result = graph.get_node("gene:tardbp_test")
        assert result is not None
        assert result.name == "TARDBP"
        assert result.pathway_cluster == "proteostasis"

    def test_upsert_updates_existing(self, graph, sample_nodes, cleanup_test_nodes):
        graph.upsert_node(sample_nodes[0])
        updated = TCGNode(id="gene:tardbp_test", entity_type="gene", name="TARDBP",
                          pathway_cluster="proteostasis", druggability_score=0.75)
        graph.upsert_node(updated)
        result = graph.get_node("gene:tardbp_test")
        assert result.druggability_score == 0.75

    def test_get_nonexistent_node_returns_none(self, graph):
        assert graph.get_node("gene:does_not_exist") is None

    def test_list_nodes_by_cluster(self, graph, sample_nodes, cleanup_test_nodes):
        for n in sample_nodes:
            graph.upsert_node(n)
        proteo = graph.list_nodes(pathway_cluster="proteostasis")
        proteo_ids = [n.id for n in proteo]
        assert "gene:tardbp_test" in proteo_ids
        assert "protein:tdp43_test" in proteo_ids
        assert "gene:stmn2_test" not in proteo_ids


class TestEdgeCRUD:
    def test_upsert_and_get_edge(self, graph, sample_nodes, cleanup_test_nodes):
        for n in sample_nodes[:2]:
            graph.upsert_node(n)
        edge = TCGEdge(
            id="edge:tardbp_tdp43_test",
            source_id="gene:tardbp_test",
            target_id="protein:tdp43_test",
            edge_type="encodes",
            confidence=0.95,
        )
        graph.upsert_edge(edge)
        result = graph.get_edge("edge:tardbp_tdp43_test")
        assert result is not None
        assert result.edge_type == "encodes"
        assert result.confidence == 0.95

    def test_update_edge_confidence(self, graph, sample_nodes, cleanup_test_nodes):
        for n in sample_nodes[:2]:
            graph.upsert_node(n)
        edge = TCGEdge(id="edge:tardbp_tdp43_test", source_id="gene:tardbp_test",
                       target_id="protein:tdp43_test", edge_type="encodes", confidence=0.3)
        graph.upsert_edge(edge)
        graph.update_edge_confidence("edge:tardbp_tdp43_test", 0.8, evidence_id="pubmed:123")
        result = graph.get_edge("edge:tardbp_tdp43_test")
        assert result.confidence == 0.8
        assert "pubmed:123" in result.evidence_ids

    def test_get_weakest_edges(self, graph, sample_nodes, cleanup_test_nodes):
        for n in sample_nodes:
            graph.upsert_node(n)
        e1 = TCGEdge(id="edge:e1_test", source_id="gene:tardbp_test", target_id="protein:tdp43_test",
                      edge_type="causes", confidence=0.2)
        e2 = TCGEdge(id="edge:e2_test", source_id="protein:tdp43_test", target_id="gene:stmn2_test",
                      edge_type="causes", confidence=0.8)
        graph.upsert_edge(e1)
        graph.upsert_edge(e2)
        weakest = graph.get_weakest_edges(limit=10)
        ids = [e.id for e in weakest]
        assert ids.index("edge:e1_test") < ids.index("edge:e2_test")

    def test_get_edges_for_node(self, graph, sample_nodes, cleanup_test_nodes):
        for n in sample_nodes[:2]:
            graph.upsert_node(n)
        edge = TCGEdge(id="edge:tardbp_tdp43_test", source_id="gene:tardbp_test",
                       target_id="protein:tdp43_test", edge_type="encodes", confidence=0.5)
        graph.upsert_edge(edge)
        outgoing = graph.get_edges_from("gene:tardbp_test")
        assert len(outgoing) >= 1
        assert outgoing[0].target_id == "protein:tdp43_test"


class TestBayesianUpdate:
    def test_bayesian_confidence_update(self, graph, sample_nodes, cleanup_test_nodes):
        for n in sample_nodes[:2]:
            graph.upsert_node(n)
        edge = TCGEdge(id="edge:bayes_test", source_id="gene:tardbp_test",
                       target_id="protein:tdp43_test", edge_type="causes", confidence=0.3)
        graph.upsert_edge(edge)
        # Bayesian update with strong evidence should increase confidence
        graph.bayesian_update("edge:bayes_test", evidence_strength=0.9, evidence_id="pubmed:999")
        result = graph.get_edge("edge:bayes_test")
        assert result.confidence > 0.3
        assert "pubmed:999" in result.evidence_ids


class TestAcquisitionQueue:
    def test_push_and_pop_queue(self, graph, sample_nodes, cleanup_test_nodes):
        for n in sample_nodes[:2]:
            graph.upsert_node(n)
        edge = TCGEdge(id="edge:queue_test", source_id="gene:tardbp_test",
                       target_id="protein:tdp43_test", edge_type="causes", confidence=0.3)
        graph.upsert_edge(edge)
        item = AcquisitionItem(
            tcg_edge_id="edge:queue_test",
            open_question="What is the binding mechanism?",
            suggested_sources=["chembl", "bindingdb"],
            priority=0.7,
            created_by="integration",
        )
        graph.push_acquisition(item)
        popped = graph.pop_acquisition()
        assert popped is not None
        assert popped.open_question == "What is the binding mechanism?"
        assert popped.status == "in_progress"

    def test_pop_returns_highest_priority(self, graph, sample_nodes, cleanup_test_nodes):
        for n in sample_nodes[:2]:
            graph.upsert_node(n)
        edge = TCGEdge(id="edge:prio_test", source_id="gene:tardbp_test",
                       target_id="protein:tdp43_test", edge_type="causes", confidence=0.3)
        graph.upsert_edge(edge)
        low = AcquisitionItem(tcg_edge_id="edge:prio_test", open_question="low priority",
                              priority=0.1, created_by="test")
        high = AcquisitionItem(tcg_edge_id="edge:prio_test", open_question="high priority",
                               priority=0.9, created_by="test")
        graph.push_acquisition(low)
        graph.push_acquisition(high)
        popped = graph.pop_acquisition()
        assert popped.open_question == "high priority"

    def test_pop_empty_queue_returns_none(self, graph):
        # Pop all existing items first, then verify None
        while graph.pop_acquisition() is not None:
            pass
        assert graph.pop_acquisition() is None


class TestGraphSummary:
    def test_summary_returns_counts(self, graph, sample_nodes, cleanup_test_nodes):
        for n in sample_nodes:
            graph.upsert_node(n)
        summary = graph.summary()
        assert summary["node_count"] >= 3
        assert "edge_count" in summary
        assert "mean_confidence" in summary
        assert "clusters" in summary
