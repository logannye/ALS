"""Tests for the IntegrationDaemon — evidence -> TCG integration."""
import pytest
from unittest.mock import MagicMock, patch
from daemons.integration_daemon import IntegrationDaemon, _classify_question_type, _match_evidence_to_edges


class TestQuestionTypeClassification:
    def test_mechanistic_question(self):
        assert _classify_question_type("Does TDP-43 aggregation cause STMN2 loss?") == "mechanistic"

    def test_binding_question(self):
        assert _classify_question_type("What is the binding affinity of riluzole to EAAT2?") == "binding"

    def test_expression_question(self):
        assert _classify_question_type("Is SOD1 expressed in upper motor neurons?") == "expression"

    def test_genetic_question(self):
        assert _classify_question_type("Is the TARDBP variant pathogenic?") == "genetic"

    def test_clinical_question(self):
        assert _classify_question_type("Is masitinib in clinical trials for ALS?") == "clinical"

    def test_pathway_question(self):
        assert _classify_question_type("What pathway does CSF1R participate in?") == "pathway"


class TestSourceMapping:
    def test_mechanistic_sources(self):
        from daemons.integration_daemon import QUESTION_TYPE_SOURCES
        assert "pubmed" in QUESTION_TYPE_SOURCES["mechanistic"]
        assert "biorxiv" in QUESTION_TYPE_SOURCES["mechanistic"]

    def test_binding_sources(self):
        from daemons.integration_daemon import QUESTION_TYPE_SOURCES
        assert "chembl" in QUESTION_TYPE_SOURCES["binding"]
        assert "bindingdb" in QUESTION_TYPE_SOURCES["binding"]


class TestEntityMatching:
    def test_single_node_match_finds_edges(self):
        node_index = {
            "tardbp": [("edge:tardbp->tdp43", 0.9), ("edge:tardbp->fus_interaction", 0.3)],
            "tdp-43": [("edge:tardbp->tdp43", 0.9), ("edge:tdp43->aggregation", 0.5)],
            "sod1": [("edge:sod1->misfolding", 0.7)],
        }
        evidence = {
            "id": "test:1",
            "body": {"claim": "TARDBP variant classified as Pathogenic in ClinVar"},
            "confidence": 0.8,
        }
        matches = _match_evidence_to_edges(evidence, node_index)
        edge_ids = [m[0] for m in matches]
        assert "edge:tardbp->tdp43" in edge_ids
        assert "edge:tardbp->fus_interaction" in edge_ids

    def test_case_insensitive_matching(self):
        node_index = {
            "riluzole": [("edge:riluzole->eaat2", 0.8)],
        }
        evidence = {
            "id": "test:2",
            "body": {"claim": "Riluzole inhibits glutamate release"},
            "confidence": 0.7,
        }
        matches = _match_evidence_to_edges(evidence, node_index)
        assert len(matches) >= 1

    def test_no_match_returns_empty(self):
        node_index = {
            "tardbp": [("edge:tardbp->tdp43", 0.9)],
        }
        evidence = {
            "id": "test:3",
            "body": {"claim": "Unrelated protein found in kidney tissue"},
            "confidence": 0.5,
        }
        matches = _match_evidence_to_edges(evidence, node_index)
        assert len(matches) == 0

    def test_deduplicates_edges(self):
        node_index = {
            "tardbp": [("edge:tardbp->tdp43", 0.9)],
            "tdp-43": [("edge:tardbp->tdp43", 0.9)],
        }
        evidence = {
            "id": "test:4",
            "body": {"claim": "TARDBP encodes TDP-43 protein"},
            "confidence": 0.9,
        }
        matches = _match_evidence_to_edges(evidence, node_index)
        edge_ids = [m[0] for m in matches]
        assert edge_ids.count("edge:tardbp->tdp43") == 1


class TestIntegrationDaemon:
    def test_init(self):
        daemon = IntegrationDaemon()
        assert daemon._interval_s > 0

    @patch("daemons.integration_daemon.IntegrationDaemon._get_unintegrated_evidence")
    def test_no_evidence_is_noop(self, mock_get):
        mock_get.return_value = []
        daemon = IntegrationDaemon()
        stats = daemon.integrate_batch()
        assert stats["items_processed"] == 0
        assert stats["edges_updated"] == 0
