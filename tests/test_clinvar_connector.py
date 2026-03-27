# tests/test_clinvar_connector.py
"""Tests for ClinVar genetic variant connector."""
from __future__ import annotations
import pytest
from connectors.clinvar import ClinVarConnector, _parse_variant_xml

class TestParseVariantXml:
    def test_empty_xml(self):
        assert _parse_variant_xml("", gene="SOD1") == []

    def test_handles_invalid_xml(self):
        assert _parse_variant_xml("not xml", gene="SOD1") == []

class TestClinVarConnector:
    def test_instantiates(self):
        connector = ClinVarConnector(evidence_store=None)
        assert connector is not None

    def test_evidence_item_id_format(self):
        connector = ClinVarConnector(evidence_store=None)
        item = connector._build_evidence_item(variation_id="12345", variant_name="SOD1 A4V", gene="SOD1", clinical_significance="Pathogenic", review_status="reviewed")
        assert item.id == "evi:clinvar:12345"
        assert item.provenance.asserted_by == "clinvar_connector"
        assert item.body["clinical_significance"] == "Pathogenic"
        assert item.body["gene"] == "SOD1"

    def test_pathogenic_is_strong(self):
        connector = ClinVarConnector(evidence_store=None)
        item = connector._build_evidence_item(variation_id="1", variant_name="test", gene="SOD1", clinical_significance="Pathogenic", review_status="reviewed")
        assert item.strength.value == "strong"

    def test_uncertain_is_emerging(self):
        connector = ClinVarConnector(evidence_store=None)
        item = connector._build_evidence_item(variation_id="2", variant_name="test", gene="FUS", clinical_significance="Uncertain significance", review_status="")
        assert item.strength.value == "emerging"

    def test_benign_refutes(self):
        connector = ClinVarConnector(evidence_store=None)
        item = connector._build_evidence_item(variation_id="3", variant_name="test", gene="SOD1", clinical_significance="Benign", review_status="reviewed")
        assert item.direction.value == "refutes"
