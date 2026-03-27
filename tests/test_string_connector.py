# tests/test_string_connector.py
"""Tests for STRING protein-protein interaction connector."""
from __future__ import annotations
import pytest
from connectors.string_db import STRINGConnector, _parse_network_response

class TestParseNetworkResponse:
    def test_parses_interactions(self):
        raw = [
            {"preferredName_A": "TARDBP", "preferredName_B": "HNRNPA1", "score": 972, "escore": 912},
            {"preferredName_A": "TARDBP", "preferredName_B": "FUS", "score": 965, "escore": 880},
        ]
        interactions = _parse_network_response(raw, query_gene="TARDBP")
        assert len(interactions) == 2
        assert interactions[0]["gene_a"] == "TARDBP"
        assert interactions[0]["combined_score"] == 972

    def test_empty_response(self):
        assert _parse_network_response([], query_gene="TARDBP") == []

class TestSTRINGConnector:
    def test_instantiates(self):
        connector = STRINGConnector(evidence_store=None)
        assert connector is not None

    def test_evidence_item_id_format(self):
        connector = STRINGConnector(evidence_store=None)
        item = connector._build_evidence_item(gene_a="TARDBP", gene_b="HNRNPA1", combined_score=972, experimental_score=912)
        assert item.id == "evi:string:HNRNPA1_TARDBP"  # Alphabetical order
        assert item.provenance.asserted_by == "string_connector"
        assert item.body["combined_score"] == 972

    def test_high_score_is_strong(self):
        connector = STRINGConnector(evidence_store=None)
        item = connector._build_evidence_item(gene_a="TARDBP", gene_b="FUS", combined_score=900, experimental_score=800)
        assert item.strength.value == "strong"

    def test_low_score_is_emerging(self):
        connector = STRINGConnector(evidence_store=None)
        item = connector._build_evidence_item(gene_a="TARDBP", gene_b="OBSCURE", combined_score=450, experimental_score=0)
        assert item.strength.value == "moderate"  # 400-699 is moderate
