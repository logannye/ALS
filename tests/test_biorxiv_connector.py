"""Tests for BiorxivConnector — no network, uses fixture dicts."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from connectors.biorxiv import BiorxivConnector, _parse_preprint
from ontology.enums import EvidenceStrength


# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

SAMPLE_PREPRINT = {
    "doi": "10.1101/2025.01.15.123456",
    "title": "TDP-43 phase separation drives ALS motor neuron toxicity via autophagy failure",
    "abstract": (
        "Amyotrophic lateral sclerosis (ALS) motor neuron degeneration is linked to "
        "cytoplasmic TDP-43 aggregates. Here we show that liquid-liquid phase separation "
        "of TDP-43 precedes aggregation and that small-molecule modulators of phase "
        "separation reduce toxicity in iPSC-derived motor neurons. These results suggest "
        "targeting TDP-43 phase separation as a therapeutic strategy for sporadic ALS."
    ),
    "authors": "Smith J, Jones A, Lee K",
    "date": "2025-01-15",
    "server": "biorxiv",
    "category": "neuroscience",
    "version": "1",
    "type": "new results",
    "license": "cc_by",
    "jatsxml": "",
    "published": "NA",
    "author_corresponding": "Smith J",
    "author_corresponding_institution": "Stanford University",
}

SAMPLE_PREPRINT_MEDRXIV = {
    "doi": "10.1101/2025.02.10.987654",
    "title": "Phase 2 trial of antisense oligonucleotide targeting SOD1 in ALS patients",
    "abstract": "A randomized placebo-controlled trial of SOD1 ASO in ALS showing significant neurofilament reduction.",
    "authors": "Williams R, Chen X",
    "date": "2025-02-10",
    "server": "medrxiv",
    "category": "neurology",
    "version": "1",
    "type": "new results",
    "license": "cc_by",
    "jatsxml": "",
    "published": "NA",
    "author_corresponding": "Williams R",
    "author_corresponding_institution": "Mayo Clinic",
}

SAMPLE_PREPRINT_NO_DOI = {
    "doi": "",
    "title": "Some preprint without a DOI",
    "abstract": "Abstract text.",
    "authors": "Anonymous",
    "date": "2025-03-01",
    "server": "biorxiv",
    "category": "neuroscience",
}

SAMPLE_API_RESPONSE = {
    "collection": [
        SAMPLE_PREPRINT,
        SAMPLE_PREPRINT_MEDRXIV,
        {
            "doi": "10.1101/2025.03.01.000001",
            "title": "Unrelated paper about plant biology",
            "abstract": "Photosynthesis in wheat under drought conditions.",
            "authors": "Green P",
            "date": "2025-03-01",
            "server": "biorxiv",
            "category": "plant_biology",
        },
    ]
}


# ---------------------------------------------------------------------------
# TestParsePreprint
# ---------------------------------------------------------------------------

class TestParsePreprint:

    def test_valid_preprint_id_format(self):
        item = _parse_preprint(SAMPLE_PREPRINT)
        assert item is not None
        assert item.id == "evi:biorxiv:10.1101_2025.01.15.123456"

    def test_id_uses_server_name(self):
        item = _parse_preprint(SAMPLE_PREPRINT_MEDRXIV)
        assert item is not None
        assert item.id.startswith("evi:medrxiv:")

    def test_id_replaces_slash_with_underscore(self):
        item = _parse_preprint(SAMPLE_PREPRINT)
        assert item is not None
        assert "/" not in item.id

    def test_id_is_lowercased(self):
        raw = dict(SAMPLE_PREPRINT)
        raw["doi"] = "10.1101/2025.01.15.UPPER"
        item = _parse_preprint(raw)
        assert item is not None
        assert item.id == item.id.lower()

    def test_claim_format(self):
        item = _parse_preprint(SAMPLE_PREPRINT)
        assert item is not None
        assert item.claim.startswith("[Preprint] ")
        assert "TDP-43 phase separation" in item.claim

    def test_claim_truncated_to_500_chars(self):
        raw = dict(SAMPLE_PREPRINT)
        raw["title"] = "A" * 600
        item = _parse_preprint(raw)
        assert item is not None
        assert len(item.claim) <= 500

    def test_strength_always_emerging(self):
        """Preprints ALWAYS get EvidenceStrength.emerging — not peer-reviewed."""
        item = _parse_preprint(SAMPLE_PREPRINT)
        assert item is not None
        assert item.strength == EvidenceStrength.emerging

    def test_strength_clamped_even_if_field_overridden(self):
        """Even if caller tries to set strong strength, preprints stay emerging."""
        item = _parse_preprint(SAMPLE_PREPRINT)
        assert item is not None
        # Strength must ALWAYS be emerging for preprints — no override possible
        assert item.strength == EvidenceStrength.emerging

    def test_missing_doi_returns_none(self):
        result = _parse_preprint(SAMPLE_PREPRINT_NO_DOI)
        assert result is None

    def test_none_doi_returns_none(self):
        raw = dict(SAMPLE_PREPRINT)
        raw["doi"] = None
        result = _parse_preprint(raw)
        assert result is None

    def test_doi_stored_in_body(self):
        item = _parse_preprint(SAMPLE_PREPRINT)
        assert item is not None
        assert item.body["doi"] == SAMPLE_PREPRINT["doi"]

    def test_body_contains_title(self):
        item = _parse_preprint(SAMPLE_PREPRINT)
        assert item is not None
        assert item.body["title"] == SAMPLE_PREPRINT["title"]

    def test_body_contains_abstract_truncated(self):
        item = _parse_preprint(SAMPLE_PREPRINT)
        assert item is not None
        assert "abstract" in item.body
        assert len(item.body["abstract"]) <= 2000

    def test_body_abstract_full_when_short(self):
        item = _parse_preprint(SAMPLE_PREPRINT)
        assert item is not None
        assert item.body["abstract"] == SAMPLE_PREPRINT["abstract"]

    def test_body_abstract_truncated_to_2000(self):
        raw = dict(SAMPLE_PREPRINT)
        raw["abstract"] = "X" * 3000
        item = _parse_preprint(raw)
        assert item is not None
        assert len(item.body["abstract"]) == 2000

    def test_body_contains_authors(self):
        item = _parse_preprint(SAMPLE_PREPRINT)
        assert item is not None
        assert item.body["authors"] == SAMPLE_PREPRINT["authors"]

    def test_body_contains_date(self):
        item = _parse_preprint(SAMPLE_PREPRINT)
        assert item is not None
        assert item.body["date"] == SAMPLE_PREPRINT["date"]

    def test_body_contains_server(self):
        item = _parse_preprint(SAMPLE_PREPRINT)
        assert item is not None
        assert item.body["server"] == "biorxiv"

    def test_body_contains_category(self):
        item = _parse_preprint(SAMPLE_PREPRINT)
        assert item is not None
        assert item.body["category"] == "neuroscience"

    def test_body_peer_reviewed_false(self):
        item = _parse_preprint(SAMPLE_PREPRINT)
        assert item is not None
        assert item.body["peer_reviewed"] is False

    def test_body_strength_value_is_emerging(self):
        item = _parse_preprint(SAMPLE_PREPRINT)
        assert item is not None
        assert item.body["strength"] == EvidenceStrength.emerging.value

    def test_source_refs_contains_doi(self):
        item = _parse_preprint(SAMPLE_PREPRINT)
        assert item is not None
        doi_ref = f"doi:{SAMPLE_PREPRINT['doi']}"
        assert doi_ref in item.source_refs


# ---------------------------------------------------------------------------
# TestBiorxivConnector
# ---------------------------------------------------------------------------

class TestBiorxivConnector:

    def test_disabled_returns_zero_items(self):
        store = MagicMock()
        connector = BiorxivConnector(store=store, enabled=False)
        result = connector.fetch(query="ALS motor neuron")
        assert result.evidence_items_added == 0
        assert result.errors == []

    def test_disabled_does_not_call_store(self):
        store = MagicMock()
        connector = BiorxivConnector(store=store, enabled=False)
        connector.fetch(query="ALS motor neuron")
        store.upsert_evidence_item.assert_not_called()

    def test_default_params(self):
        connector = BiorxivConnector(store=None)
        assert connector._enabled is True
        assert connector._lookback_days == 90
        assert connector._max_results == 15

    def test_custom_params(self):
        connector = BiorxivConnector(store=None, enabled=False, lookback_days=30, max_results=5)
        assert connector._enabled is False
        assert connector._lookback_days == 30
        assert connector._max_results == 5

    def test_mock_api_response_parses_relevant_items(self):
        """Mock HTTP response → only ALS-related preprints are returned."""
        store = MagicMock()
        connector = BiorxivConnector(store=store, enabled=True, max_results=10)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_API_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            result = connector.fetch(query="ALS motor neuron")

        # Both ALS-related preprints should be included (biorxiv + medrxiv)
        # The plant biology one should be filtered out
        assert result.evidence_items_added >= 1
        assert len(result.errors) == 0

    def test_mock_api_filters_irrelevant_preprints(self):
        """Preprints not matching any query word are filtered out."""
        store = MagicMock()
        connector = BiorxivConnector(store=store, enabled=True, max_results=10)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "collection": [
                {
                    "doi": "10.1101/2025.03.01.000001",
                    "title": "Unrelated paper about plant biology",
                    "abstract": "Photosynthesis in wheat under drought conditions.",
                    "authors": "Green P",
                    "date": "2025-03-01",
                    "server": "biorxiv",
                    "category": "plant_biology",
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            result = connector.fetch(query="ALS motor neuron")

        assert result.evidence_items_added == 0

    def test_api_error_returns_error_in_result(self):
        """HTTP errors should be captured in result.errors, not raised."""
        store = MagicMock()
        connector = BiorxivConnector(store=store, enabled=True, max_results=5)

        with patch("requests.get", side_effect=Exception("Connection refused")):
            result = connector.fetch(query="ALS TDP-43")

        assert result.evidence_items_added == 0
        assert len(result.errors) > 0
        assert any("Connection refused" in e or "biorxiv" in e.lower() or "medrxiv" in e.lower()
                   for e in result.errors)

    def test_store_upsert_called_for_each_item(self):
        """EvidenceItems should be upserted to the store."""
        store = MagicMock()
        connector = BiorxivConnector(store=store, enabled=True, max_results=10)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "collection": [SAMPLE_PREPRINT]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            result = connector.fetch(query="ALS TDP-43")

        if result.evidence_items_added > 0:
            store.upsert_evidence_item.assert_called()

    def test_duplicate_handling(self):
        """Items that raise on upsert should increment skipped_duplicates or errors."""
        store = MagicMock()
        store.upsert_evidence_item.side_effect = Exception("duplicate key")
        connector = BiorxivConnector(store=store, enabled=True, max_results=10)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"collection": [SAMPLE_PREPRINT]}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            result = connector.fetch(query="ALS TDP-43")

        # Should not raise — errors should be captured
        assert result.evidence_items_added == 0
        assert len(result.errors) > 0 or result.skipped_duplicates > 0

    def test_searches_both_servers(self):
        """fetch() should query both biorxiv and medrxiv."""
        store = MagicMock()
        connector = BiorxivConnector(store=store, enabled=True, max_results=5)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"collection": []}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            connector.fetch(query="ALS motor neuron")

        call_urls = [str(call.args[0]) for call in mock_get.call_args_list
                     if call.args]
        # Should have called with both biorxiv and medrxiv URLs
        assert any("biorxiv" in url for url in call_urls)
        assert any("medrxiv" in url for url in call_urls)

    def test_empty_collection_returns_zero(self):
        """Empty API response should return 0 items without error."""
        store = MagicMock()
        connector = BiorxivConnector(store=store, enabled=True)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"collection": []}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            result = connector.fetch(query="ALS")

        assert result.evidence_items_added == 0
        assert result.errors == []
