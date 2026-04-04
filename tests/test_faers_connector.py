"""Tests for FAERSConnector -- FDA Adverse Event Reporting System."""
from __future__ import annotations

import pytest

from connectors.base import BaseConnector, ConnectorResult
from connectors.faers import (
    FAERSConnector,
    _normalize_name,
    _parse_count_response,
    _build_safety_profile,
    _assess_safety_signal,
)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_OPENFDA_RESPONSE = {
    "results": [
        {"term": "NAUSEA", "count": 245},
        {"term": "DIZZINESS", "count": 180},
        {"term": "HEADACHE", "count": 150},
        {"term": "FATIGUE", "count": 120},
        {"term": "VOMITING", "count": 95},
        {"term": "DIARRHOEA", "count": 80},
        {"term": "ASTHENIA", "count": 60},
        {"term": "PAIN", "count": 55},
        {"term": "ABDOMINAL PAIN", "count": 40},
        {"term": "RASH", "count": 30},
        {"term": "INSOMNIA", "count": 20},
        {"term": "PYREXIA", "count": 5},
    ],
}

SAMPLE_OPENFDA_EMPTY = {"results": []}


# ---------------------------------------------------------------------------
# _normalize_name
# ---------------------------------------------------------------------------

def test_normalize_name():
    assert _normalize_name("Riluzole Hydrochloride") == "riluzole_hydrochloride"


def test_normalize_name_hyphens():
    assert _normalize_name("5-alpha-reductase") == "5_alpha_reductase"


def test_normalize_name_special_chars():
    assert _normalize_name("Drug (oral)") == "drug_oral"


# ---------------------------------------------------------------------------
# _parse_count_response
# ---------------------------------------------------------------------------

def test_parse_count_response_extracts_reactions():
    reactions = _parse_count_response(SAMPLE_OPENFDA_RESPONSE, "riluzole")
    assert len(reactions) == 12
    assert reactions[0] == {"reaction": "NAUSEA", "count": 245}
    assert reactions[1] == {"reaction": "DIZZINESS", "count": 180}


def test_parse_count_response_empty():
    reactions = _parse_count_response(SAMPLE_OPENFDA_EMPTY, "unknown_drug")
    assert reactions == []


def test_parse_count_response_missing_results_key():
    reactions = _parse_count_response({}, "test_drug")
    assert reactions == []


# ---------------------------------------------------------------------------
# _build_safety_profile
# ---------------------------------------------------------------------------

def test_build_safety_profile_aggregation():
    reactions = [
        {"reaction": "NAUSEA", "count": 245},
        {"reaction": "DIZZINESS", "count": 180},
        {"reaction": "HEADACHE", "count": 150},
    ]
    profile = _build_safety_profile(reactions, "riluzole")
    assert profile["total_reports"] == 575
    assert profile["unique_reactions"] == 3
    assert profile["drug_name"] == "riluzole"
    assert len(profile["top_reactions"]) == 3
    # Should be sorted by count descending
    assert profile["top_reactions"][0]["reaction"] == "NAUSEA"


def test_build_safety_profile_top_20_limit():
    reactions = [{"reaction": f"REACTION_{i}", "count": 100 - i} for i in range(30)]
    profile = _build_safety_profile(reactions, "test_drug")
    assert len(profile["top_reactions"]) == 20


# ---------------------------------------------------------------------------
# _assess_safety_signal
# ---------------------------------------------------------------------------

def test_assess_safety_signal_returns_tuple():
    profile = {"total_reports": 5000, "unique_reactions": 42}
    is_safe, reasoning = _assess_safety_signal(profile, "riluzole")
    assert isinstance(is_safe, bool)
    assert isinstance(reasoning, str)
    assert is_safe is True
    assert "5000" in reasoning
    assert "42" in reasoning
    assert "riluzole" in reasoning


def test_assess_safety_signal_high_volume():
    """Even high-volume drugs return is_safe=True (count endpoint limitation)."""
    profile = {"total_reports": 50000, "unique_reactions": 200}
    is_safe, reasoning = _assess_safety_signal(profile, "aspirin")
    assert is_safe is True


# ---------------------------------------------------------------------------
# Connector class
# ---------------------------------------------------------------------------

def test_connector_inherits_base():
    c = FAERSConnector()
    assert isinstance(c, BaseConnector)


def test_connector_instantiates_with_defaults():
    c = FAERSConnector()
    assert c._min_report_count == 10
    assert c._store is None


def test_connector_accepts_custom_min_report_count():
    c = FAERSConnector(min_report_count=50)
    assert c._min_report_count == 50


def test_fetch_no_drug_returns_empty():
    c = FAERSConnector()
    result = c.fetch()
    assert isinstance(result, ConnectorResult)
    assert result.evidence_items_added == 0
    assert result.errors == []


def test_fetch_no_drug_explicit_empty_string():
    c = FAERSConnector()
    result = c.fetch(drug_name="")
    assert result.evidence_items_added == 0


def test_evidence_id_format():
    """Verify the expected evidence ID format for FAERS items."""
    drug_norm = _normalize_name("Riluzole")
    expected_profile_id = f"evi:faers:{drug_norm}_safety_profile"
    assert expected_profile_id == "evi:faers:riluzole_safety_profile"

    reaction_norm = _normalize_name("NAUSEA")
    expected_reaction_id = f"evi:faers:{drug_norm}_{reaction_norm}"
    assert expected_reaction_id == "evi:faers:riluzole_nausea"


# ---------------------------------------------------------------------------
# Integration test (requires network)
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_fetch_riluzole_real():
    """Live API call to openFDA for riluzole adverse events."""
    c = FAERSConnector()
    result = c.fetch(drug_name="riluzole")
    assert isinstance(result, ConnectorResult)
    # Riluzole is an approved ALS drug -- should have FAERS reports
    assert result.evidence_items_added > 0
    assert result.errors == []
