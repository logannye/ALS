"""Tests for TCG data models."""
import pytest
from tcg.models import TCGNode, TCGEdge, TCGHypothesis, AcquisitionItem


class TestTCGNode:
    def test_create_gene_node(self):
        node = TCGNode(
            id="gene:tardbp",
            entity_type="gene",
            name="TARDBP",
            pathway_cluster="proteostasis",
        )
        assert node.id == "gene:tardbp"
        assert node.entity_type == "gene"
        assert node.name == "TARDBP"
        assert node.pathway_cluster == "proteostasis"
        assert node.druggability_score == 0.0
        assert node.metadata == {}

    def test_create_compound_node(self):
        node = TCGNode(
            id="compound:riluzole",
            entity_type="compound",
            name="Riluzole",
            pathway_cluster="excitotoxicity",
            druggability_score=0.9,
        )
        assert node.druggability_score == 0.9

    def test_node_to_dict_roundtrip(self):
        node = TCGNode(
            id="protein:tdp-43",
            entity_type="protein",
            name="TDP-43",
            pathway_cluster="proteostasis",
            description="TAR DNA-binding protein 43",
            metadata={"uniprot": "Q13148"},
        )
        d = node.to_dict()
        restored = TCGNode.from_dict(d)
        assert restored.id == node.id
        assert restored.metadata == {"uniprot": "Q13148"}


class TestTCGEdge:
    def test_create_causal_edge(self):
        edge = TCGEdge(
            id="edge:tdp43_agg->stmn2_splicing",
            source_id="protein:tdp-43",
            target_id="gene:stmn2",
            edge_type="causes",
            confidence=0.4,
            open_questions=["Is the effect mediated by nuclear TDP-43 depletion?"],
        )
        assert edge.confidence == 0.4
        assert len(edge.open_questions) == 1
        assert edge.evidence_ids == []
        assert edge.contradiction_ids == []

    def test_edge_therapeutic_priority(self):
        """Priority = therapeutic_relevance * (1 - confidence). Higher = more important to investigate."""
        edge = TCGEdge(
            id="edge:test",
            source_id="a",
            target_id="b",
            edge_type="causes",
            confidence=0.3,
            intervention_potential={"druggable": True, "therapeutic_relevance": 0.9},
        )
        assert edge.therapeutic_priority() == pytest.approx(0.9 * 0.7, abs=0.01)

    def test_edge_to_dict_roundtrip(self):
        edge = TCGEdge(
            id="edge:test",
            source_id="a",
            target_id="b",
            edge_type="inhibits",
            confidence=0.7,
            evidence_ids=["pubmed:123", "pubmed:456"],
        )
        d = edge.to_dict()
        restored = TCGEdge.from_dict(d)
        assert restored.evidence_ids == ["pubmed:123", "pubmed:456"]


class TestTCGHypothesis:
    def test_create_hypothesis(self):
        hyp = TCGHypothesis(
            id="hyp:vtx002_tdp43_clearance",
            hypothesis="VTx-002 gene therapy reduces TDP-43 aggregation via enhanced autophagy",
            supporting_path=["edge:vtx002->tdp43", "edge:tdp43->autophagy"],
            confidence=0.3,
            status="proposed",
            generated_by="reasoning",
            therapeutic_relevance=0.85,
        )
        assert hyp.status == "proposed"
        assert len(hyp.supporting_path) == 2

    def test_hypothesis_status_values(self):
        for status in ["proposed", "under_investigation", "supported", "refuted", "actionable"]:
            hyp = TCGHypothesis(
                id=f"hyp:test_{status}",
                hypothesis="test",
                status=status,
            )
            assert hyp.status == status


class TestAcquisitionItem:
    def test_create_acquisition_item(self):
        item = AcquisitionItem(
            tcg_edge_id="edge:tdp43_agg->stmn2_splicing",
            open_question="Does TDP-43 aggregation directly cause STMN2 cryptic exon inclusion?",
            suggested_sources=["pubmed", "biorxiv"],
            priority=0.63,
            created_by="reasoning",
        )
        assert item.status == "pending"
        assert item.exhausted_sources == []
