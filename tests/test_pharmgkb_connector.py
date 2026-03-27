# tests/test_pharmgkb_connector.py
"""Tests for PharmGKB pharmacogenomics connector."""
from __future__ import annotations
import pytest
from connectors.pharmgkb import PharmGKBConnector, _parse_drug_response, _parse_clinical_annotations

class TestParseDrugResponse:
    def test_parses_drug_info(self):
        raw = {"data": {"id": "PA450626", "name": "riluzole", "genericNames": ["riluzole"]}}
        drug = _parse_drug_response(raw)
        assert drug["pharmgkb_id"] == "PA450626"
        assert drug["name"] == "riluzole"

    def test_empty_response(self):
        drug = _parse_drug_response({})
        assert drug["pharmgkb_id"] == ""

class TestParseClinicalAnnotations:
    def test_parses_annotations(self):
        raw = {"data": [{"id": 1, "gene": {"symbol": "CYP1A2"}, "drug": {"name": "riluzole"}, "phenotypeCategory": {"term": "Metabolism/PK"}, "level": "1A", "summary": "CYP1A2 metabolizes riluzole"}]}
        annotations = _parse_clinical_annotations(raw)
        assert len(annotations) == 1
        assert annotations[0]["gene"] == "CYP1A2"

    def test_empty_annotations(self):
        assert _parse_clinical_annotations({"data": []}) == []

class TestPharmGKBConnector:
    def test_instantiates(self):
        connector = PharmGKBConnector(evidence_store=None)
        assert connector is not None

    def test_evidence_item_id_format(self):
        connector = PharmGKBConnector(evidence_store=None)
        item = connector._build_evidence_item(pharmgkb_id="PA450626", drug_name="riluzole", gene="CYP1A2", annotation="CYP1A2 metabolizes riluzole", level="1A", category="Metabolism/PK")
        assert item.id == "evi:pharmgkb:PA450626_CYP1A2"
        assert item.provenance.asserted_by == "pharmgkb_connector"
        assert "riluzole" in item.claim

    def test_level_1a_is_strong(self):
        connector = PharmGKBConnector(evidence_store=None)
        item = connector._build_evidence_item(pharmgkb_id="PA1", drug_name="riluzole", gene="CYP1A2", annotation="test", level="1A", category="Metabolism/PK")
        assert item.strength.value == "strong"

    def test_level_3_is_emerging(self):
        connector = PharmGKBConnector(evidence_store=None)
        item = connector._build_evidence_item(pharmgkb_id="PA2", drug_name="test", gene="TEST", annotation="test", level="3", category="Other")
        assert item.strength.value == "emerging"
