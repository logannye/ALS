"""Tests for OpenTargetsConnector — uses fixture JSON for unit tests.

Integration tests that require network access are marked @pytest.mark.network
and skipped by default (via -k 'not network').
"""
import pytest

from connectors.opentargets import OpenTargetsConnector, _parse_target_association


# ---------------------------------------------------------------------------
# Fixture: sample association row as returned by the GraphQL API
# ---------------------------------------------------------------------------

SAMPLE_ROW = {
    "target": {
        "id": "ENSG00000196839",
        "approvedSymbol": "SIGMAR1",
    },
    "score": 0.63,
    "datatypeScores": [
        {"id": "genetic_association", "score": 0.41},
        {"id": "known_drug", "score": 0.22},
    ],
}

SAMPLE_ROW_NO_DATATYPE = {
    "target": {
        "id": "ENSG00000197535",
        "approvedSymbol": "MYH9",
    },
    "score": 0.12,
    "datatypeScores": [],
}

SAMPLE_ROW_MINIMAL = {
    "target": {
        "id": "ENSG00000105976",
        "approvedSymbol": "MET",
    },
    "score": 0.05,
    "datatypeScores": [
        {"id": "genetic_association", "score": 0.05},
    ],
}


# ---------------------------------------------------------------------------
# _parse_target_association: unit tests
# ---------------------------------------------------------------------------

def test_parse_id_format():
    item = _parse_target_association(SAMPLE_ROW)
    assert item.id == "evi:ot:ENSG00000196839_als"


def test_parse_claim_contains_symbol():
    item = _parse_target_association(SAMPLE_ROW)
    assert "SIGMAR1" in item.claim


def test_parse_claim_contains_score():
    item = _parse_target_association(SAMPLE_ROW)
    assert "0.63" in item.claim


def test_parse_direction_is_insufficient():
    from ontology.enums import EvidenceDirection
    item = _parse_target_association(SAMPLE_ROW)
    assert item.direction == EvidenceDirection.insufficient


def test_parse_strength_is_unknown():
    from ontology.enums import EvidenceStrength
    item = _parse_target_association(SAMPLE_ROW)
    assert item.strength == EvidenceStrength.unknown


def test_parse_provenance_source_system():
    from ontology.enums import SourceSystem
    item = _parse_target_association(SAMPLE_ROW)
    assert item.provenance.source_system == SourceSystem.database


def test_parse_provenance_asserted_by():
    item = _parse_target_association(SAMPLE_ROW)
    assert item.provenance.asserted_by == "opentargets_connector"


def test_parse_uncertainty_confidence():
    item = _parse_target_association(SAMPLE_ROW)
    assert abs(item.uncertainty.confidence - 0.63) < 1e-6


def test_parse_body_association_score():
    item = _parse_target_association(SAMPLE_ROW)
    assert abs(item.body["association_score"] - 0.63) < 1e-6


def test_parse_body_ensembl_id():
    item = _parse_target_association(SAMPLE_ROW)
    assert item.body["ensembl_id"] == "ENSG00000196839"


def test_parse_body_gene_symbol():
    item = _parse_target_association(SAMPLE_ROW)
    assert item.body["gene_symbol"] == "SIGMAR1"


def test_parse_body_mechanism_target():
    item = _parse_target_association(SAMPLE_ROW)
    assert item.body["mechanism_target"] == "SIGMAR1"


def test_parse_body_pch_layer():
    item = _parse_target_association(SAMPLE_ROW)
    assert item.body["pch_layer"] == 1


def test_parse_body_erik_eligible():
    item = _parse_target_association(SAMPLE_ROW)
    assert item.body["erik_eligible"] is True


def test_parse_body_applicable_subtypes():
    item = _parse_target_association(SAMPLE_ROW)
    assert "sporadic_tdp43" in item.body["applicable_subtypes"]
    assert "unresolved" in item.body["applicable_subtypes"]


def test_parse_body_protocol_layer_empty():
    item = _parse_target_association(SAMPLE_ROW)
    assert item.body["protocol_layer"] == ""


def test_parse_body_genetic_association_score():
    item = _parse_target_association(SAMPLE_ROW)
    assert abs(item.body["genetic_association"] - 0.41) < 1e-6


def test_parse_body_known_drug_count():
    """known_drug_count derived from datatypeScores."""
    item = _parse_target_association(SAMPLE_ROW)
    assert item.body["known_drug_count"] == 0.22


def test_parse_row_with_no_datatype_scores():
    item = _parse_target_association(SAMPLE_ROW_NO_DATATYPE)
    assert item.id == "evi:ot:ENSG00000197535_als"
    assert item.body["genetic_association"] == 0.0
    assert item.body["known_drug_count"] == 0.0


def test_parse_minimal_row():
    item = _parse_target_association(SAMPLE_ROW_MINIMAL)
    assert item.id == "evi:ot:ENSG00000105976_als"
    assert "MET" in item.claim


# ---------------------------------------------------------------------------
# Connector instantiation
# ---------------------------------------------------------------------------

def test_connector_instantiates():
    c = OpenTargetsConnector()
    assert c is not None


def test_connector_has_graphql_url():
    c = OpenTargetsConnector()
    assert "opentargets.org" in c.GRAPHQL_URL


def test_connector_has_als_efo_id():
    c = OpenTargetsConnector()
    assert c.ALS_EFO_ID == "EFO_0000253"


def test_connector_inherits_base():
    from connectors.base import BaseConnector
    c = OpenTargetsConnector()
    assert isinstance(c, BaseConnector)


def test_connector_fetch_method_exists():
    c = OpenTargetsConnector()
    assert callable(c.fetch)


def test_connector_fetch_als_targets_method_exists():
    c = OpenTargetsConnector()
    assert callable(c.fetch_als_targets)


# ---------------------------------------------------------------------------
# GraphQL query structure
# ---------------------------------------------------------------------------

def test_connector_graphql_query_contains_efo_variable():
    c = OpenTargetsConnector()
    assert "$efoId" in c.GRAPHQL_QUERY


def test_connector_graphql_query_contains_size_variable():
    c = OpenTargetsConnector()
    assert "$size" in c.GRAPHQL_QUERY


def test_connector_graphql_query_contains_associated_targets():
    c = OpenTargetsConnector()
    assert "associatedTargets" in c.GRAPHQL_QUERY


def test_connector_graphql_query_contains_datatypescores():
    c = OpenTargetsConnector()
    assert "datatypeScores" in c.GRAPHQL_QUERY


# ---------------------------------------------------------------------------
# Network integration tests
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_fetch_als_targets_real():
    """Integration: query OpenTargets GraphQL for ALS (EFO_0000253)."""
    c = OpenTargetsConnector()
    result = c.fetch_als_targets(min_score=0.1, max_results=10)
    assert result.evidence_items_added >= 0
    assert len(result.errors) == 0


@pytest.mark.network
def test_fetch_via_fetch_method():
    """Integration: top-level fetch() delegates to fetch_als_targets."""
    c = OpenTargetsConnector()
    result = c.fetch()
    assert result.evidence_items_added >= 0
