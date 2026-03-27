"""Tests for Reactome pathway connector."""
from __future__ import annotations
import pytest
from connectors.reactome import ReactomeConnector, _parse_pathway_response, _parse_contained_events

class TestParsePathwayResponse:
    def test_parses_pathway_list(self):
        raw = [
            {"stId": "R-HSA-3371556", "displayName": "Cellular response to heat stress", "speciesName": "Homo sapiens"},
            {"stId": "R-HSA-392499", "displayName": "Metabolism of proteins", "speciesName": "Homo sapiens"},
        ]
        pathways = _parse_pathway_response(raw, uniprot_id="Q13148")
        assert len(pathways) == 2
        assert pathways[0]["pathway_id"] == "R-HSA-3371556"
        assert pathways[0]["uniprot_id"] == "Q13148"

    def test_empty_response(self):
        assert _parse_pathway_response([], uniprot_id="Q13148") == []

class TestParseContainedEvents:
    def test_parses_reaction_steps(self):
        raw = [
            {"stId": "R-HSA-3371568", "displayName": "HSF1 trimerization", "className": "Reaction"},
            {"stId": "R-HSA-3371571", "displayName": "HSP70 binding", "className": "Reaction"},
        ]
        steps = _parse_contained_events(raw, pathway_id="R-HSA-3371556")
        assert len(steps) == 2
        assert steps[0]["reaction_id"] == "R-HSA-3371568"

class TestReactomeConnector:
    def test_instantiates(self):
        connector = ReactomeConnector(evidence_store=None)
        assert connector is not None

    def test_evidence_item_id_format(self):
        connector = ReactomeConnector(evidence_store=None)
        item = connector._build_evidence_item(
            pathway_id="R-HSA-3371556", pathway_name="Cellular response to heat stress",
            uniprot_id="Q13148", gene_symbol="TARDBP", num_reactions=12,
        )
        assert item.id == "evi:reactome:R-HSA-3371556_Q13148"
        assert item.provenance.source_system.value == "database"
        assert item.provenance.asserted_by == "reactome_connector"
        assert "pathway" in item.body
