"""Tests for the IntegrationDaemon — evidence -> TCG integration."""
import pytest
from unittest.mock import MagicMock, patch
from daemons.integration_daemon import IntegrationDaemon, _classify_question_type


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
