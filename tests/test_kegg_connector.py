# tests/test_kegg_connector.py
"""Tests for KEGG pathway connector."""
from __future__ import annotations
import pytest
from connectors.kegg import KEGGConnector, _parse_link_response, _parse_pathway_entry

class TestParseLinkResponse:
    def test_parses_gene_pathway_links(self):
        raw_text = "hsa:6647\tpath:hsa04141\nhsa:6647\tpath:hsa04010\n"
        links = _parse_link_response(raw_text)
        assert len(links) == 2
        assert links[0] == ("hsa:6647", "path:hsa04141")

    def test_empty_response(self):
        assert _parse_link_response("") == []

class TestParsePathwayEntry:
    def test_extracts_name(self):
        raw_text = "ENTRY       hsa04141\nNAME        Protein processing in endoplasmic reticulum - Homo sapiens\nDESCRIPTION The endoplasmic reticulum\n///"
        entry = _parse_pathway_entry(raw_text)
        assert "Protein processing" in entry["name"]

    def test_missing_name_returns_empty(self):
        entry = _parse_pathway_entry("ENTRY   hsa04141\n///")
        assert entry["name"] == ""

class TestKEGGConnector:
    def test_instantiates(self):
        connector = KEGGConnector(evidence_store=None)
        assert connector is not None

    def test_evidence_item_id_format(self):
        connector = KEGGConnector(evidence_store=None)
        item = connector._build_evidence_item(pathway_id="hsa04141", pathway_name="Protein processing in endoplasmic reticulum", gene_id="6647", gene_symbol="SOD1")
        assert item.id == "evi:kegg:hsa04141_SOD1"
        assert item.provenance.asserted_by == "kegg_connector"
