# tests/test_omim_connector.py
"""Tests for OMIM gene-phenotype connector."""
from __future__ import annotations
import pytest
from connectors.omim import OMIMConnector, _parse_entry_response

class TestParseEntryResponse:
    def test_parses_als_phenotype(self):
        raw = {"omim": {"entryList": [{"entry": {
            "mimNumber": 105400,
            "titles": {"preferredTitle": "AMYOTROPHIC LATERAL SCLEROSIS 1; ALS1"},
            "geneMap": {"phenotypeMapList": [
                {"phenotypeMap": {"phenotype": "Amyotrophic lateral sclerosis 1", "phenotypeMimNumber": 105400, "geneSymbols": "SOD1", "phenotypeInheritance": "Autosomal dominant"}}
            ]},
        }}]}}
        entries = _parse_entry_response(raw)
        assert len(entries) == 1
        assert entries[0]["mim_number"] == 105400
        assert "SOD1" in entries[0]["gene_symbols"]

    def test_empty_response(self):
        assert _parse_entry_response({"omim": {"entryList": []}}) == []

class TestOMIMConnector:
    def test_instantiates(self):
        connector = OMIMConnector(evidence_store=None)
        assert connector is not None

    def test_evidence_item_id_format(self):
        connector = OMIMConnector(evidence_store=None)
        item = connector._build_evidence_item(mim_number=105400, title="AMYOTROPHIC LATERAL SCLEROSIS 1; ALS1", gene_symbols="SOD1", inheritance="Autosomal dominant", phenotype="Amyotrophic lateral sclerosis 1")
        assert item.id == "evi:omim:105400"
        assert item.provenance.asserted_by == "omim_connector"
