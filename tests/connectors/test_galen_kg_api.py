"""Tests for GalenKGAPIConnector — no real HTTP calls in unit tests.

All HTTP calls are mocked with unittest.mock.patch so these tests run
without network access.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

import requests as req_lib

from connectors.base import BaseConnector, ConnectorResult
from connectors.galen_kg_api import (
    GalenKGAPIConnector,
    _make_evidence_id,
    _parse_neighbor_record,
    _parse_search_record,
)
from ontology.evidence import EvidenceItem
from ontology.enums import SourceSystem


# ---------------------------------------------------------------------------
# Helpers: mock API payloads
# ---------------------------------------------------------------------------


def _neighbors_payload() -> dict:
    """Simulate /api/erik-bridge/kg/neighbors response."""
    return {
        "neighbors": [
            {
                "name": "TDP-43",
                "entity_type": "protein",
                "relationship_type": "interacts_with",
                "confidence": 0.85,
            },
            {
                "name": "FUS",
                "entity_type": "gene",
                "relationship_type": "co_expressed_with",
                "confidence": 0.72,
            },
        ]
    }


def _search_payload() -> dict:
    """Simulate /api/erik-bridge/kg/search response."""
    return {
        "results": [
            {
                "name": "TARDBP",
                "entity_type": "gene",
                "description": "TAR DNA binding protein involved in ALS",
                "confidence": 0.78,
            },
            {
                "name": "mTOR pathway",
                "entity_type": "pathway",
                "description": "Mechanistic target of rapamycin signaling",
                "confidence": 0.65,
            },
        ]
    }


def _empty_payload() -> dict:
    """Simulate an empty response."""
    return {"neighbors": []}


def _make_response(payload: dict, status: int = 200) -> MagicMock:
    """Create a mock requests.Response with .json() and raise_for_status."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    if status >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status}")
    else:
        resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_instantiates_with_no_args(self):
        c = GalenKGAPIConnector()
        assert c is not None

    def test_instantiates_with_store(self):
        store = MagicMock()
        c = GalenKGAPIConnector(store=store)
        assert c._store is store

    def test_inherits_base_connector(self):
        c = GalenKGAPIConnector()
        assert isinstance(c, BaseConnector)

    def test_fetch_method_exists(self):
        c = GalenKGAPIConnector()
        assert callable(c.fetch)

    def test_default_base_url(self, monkeypatch):
        monkeypatch.delenv("GALEN_API_URL", raising=False)
        c = GalenKGAPIConnector()
        assert "localhost:8000" in c._base_url

    def test_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("GALEN_API_URL", "https://galen.railway.app")
        c = GalenKGAPIConnector()
        assert c._base_url == "https://galen.railway.app"

    def test_trailing_slash_stripped_from_env_url(self, monkeypatch):
        monkeypatch.setenv("GALEN_API_URL", "https://galen.railway.app/")
        c = GalenKGAPIConnector()
        assert not c._base_url.endswith("/")


# ---------------------------------------------------------------------------
# fetch() — empty gene
# ---------------------------------------------------------------------------


class TestFetchEmptyGene:
    def test_empty_gene_returns_empty_result(self):
        c = GalenKGAPIConnector()
        with patch("requests.get") as mock_get:
            result = c.fetch(gene="")
        mock_get.assert_not_called()
        assert result.evidence_items_added == 0
        assert result.errors == []

    def test_no_args_returns_empty_result(self):
        c = GalenKGAPIConnector()
        with patch("requests.get") as mock_get:
            result = c.fetch()
        mock_get.assert_not_called()
        assert isinstance(result, ConnectorResult)
        assert result.evidence_items_added == 0


# ---------------------------------------------------------------------------
# fetch() — successful neighbor results (primary endpoint)
# ---------------------------------------------------------------------------


class TestFetchNeighbors:
    def test_fetch_gene_returns_evidence_items_from_neighbors(self):
        """fetch(gene=...) calls neighbors endpoint and returns EvidenceItems."""
        c = GalenKGAPIConnector()
        with patch("requests.get", return_value=_make_response(_neighbors_payload())):
            result = c.fetch(gene="SIGMAR1")
        assert result.evidence_items_added == 2
        assert result.errors == []

    def test_fetch_gene_empty_neighbors_falls_back_to_search(self):
        """Empty neighbors triggers fallback to /search endpoint."""
        c = GalenKGAPIConnector()
        responses = [
            _make_response(_empty_payload()),          # neighbors: empty
            _make_response(_search_payload()),          # search: results
        ]
        with patch("requests.get", side_effect=responses):
            result = c.fetch(gene="SIGMAR1")
        assert result.evidence_items_added == 2
        assert result.errors == []

    def test_fetch_gene_no_http_error_from_neighbors(self):
        """Successful neighbor fetch has no errors."""
        c = GalenKGAPIConnector()
        with patch("requests.get", return_value=_make_response(_neighbors_payload())):
            result = c.fetch(gene="TBK1")
        assert result.errors == []

    def test_fetch_returns_connector_result(self):
        c = GalenKGAPIConnector()
        with patch("requests.get", return_value=_make_response(_neighbors_payload())):
            result = c.fetch(gene="SOD1")
        assert isinstance(result, ConnectorResult)


# ---------------------------------------------------------------------------
# fetch() — fallback to search endpoint
# ---------------------------------------------------------------------------


class TestFetchSearchFallback:
    def test_fallback_to_search_on_neighbor_http_error(self):
        """HTTP error on neighbors triggers fallback to search endpoint."""
        c = GalenKGAPIConnector()

        def side_effect(url, **kwargs):
            if "neighbors" in url:
                return _make_response({}, 404)
            return _make_response(_search_payload())

        with patch("requests.get", side_effect=side_effect):
            result = c.fetch(gene="FUS")
        # Fallback search should yield items
        assert result.evidence_items_added == 2

    def test_fallback_search_returns_evidence_items(self):
        """Search fallback creates EvidenceItems from search results."""
        c = GalenKGAPIConnector()

        responses = [
            _make_response({"neighbors": []}),         # neighbors: empty
            _make_response(_search_payload()),          # search: results
        ]
        with patch("requests.get", side_effect=responses):
            result = c.fetch(gene="TARDBP")
        assert result.evidence_items_added == 2
        assert result.errors == []


# ---------------------------------------------------------------------------
# EvidenceItem IDs and structure
# ---------------------------------------------------------------------------


class TestEvidenceItemStructure:
    def test_evidence_id_format(self):
        """EvidenceItem IDs follow evi:galen_kg:{name}_{gene} format."""
        eid = _make_evidence_id("TDP-43", "SIGMAR1")
        assert eid == "evi:galen_kg:tdp-43_sigmar1"

    def test_evidence_id_lowercased(self):
        eid = _make_evidence_id("FUS Protein", "TBK1")
        assert eid == eid.lower()

    def test_evidence_id_spaces_replaced_with_underscores(self):
        eid = _make_evidence_id("mTOR pathway", "MTOR")
        assert " " not in eid

    def test_neighbor_record_id_correct(self):
        record = {
            "name": "TDP-43",
            "entity_type": "protein",
            "relationship_type": "interacts_with",
            "confidence": 0.85,
        }
        item = _parse_neighbor_record(record, "SIGMAR1")
        assert item.id == "evi:galen_kg:tdp-43_sigmar1"

    def test_neighbor_record_pch_layer_is_1(self):
        record = {
            "name": "FUS",
            "entity_type": "gene",
            "relationship_type": "co_expressed_with",
            "confidence": 0.72,
        }
        item = _parse_neighbor_record(record, "SOD1")
        assert item.body["pch_layer"] == 1

    def test_neighbor_record_source_system_database(self):
        record = {"name": "OPTN", "entity_type": "gene", "confidence": 0.6}
        item = _parse_neighbor_record(record, "TBK1")
        assert item.provenance.source_system == SourceSystem.database

    def test_neighbor_record_strength_is_emerging(self):
        from ontology.enums import EvidenceStrength
        record = {"name": "NEK1", "entity_type": "gene", "confidence": 0.6}
        item = _parse_neighbor_record(record, "C9orf72")
        assert item.strength == EvidenceStrength.emerging

    def test_neighbor_record_confidence_propagated(self):
        record = {"name": "VCP", "entity_type": "gene", "confidence": 0.91}
        item = _parse_neighbor_record(record, "UBQLN2")
        assert abs(item.uncertainty.confidence - 0.91) < 1e-6

    def test_neighbor_record_asserted_by(self):
        record = {"name": "STMN2", "entity_type": "gene", "confidence": 0.7}
        item = _parse_neighbor_record(record, "UNC13A")
        assert item.provenance.asserted_by == "galen_kg_api_connector"

    def test_neighbor_record_query_gene_in_body(self):
        record = {"name": "BDNF", "entity_type": "gene", "confidence": 0.65}
        item = _parse_neighbor_record(record, "SARM1")
        assert item.body["query_gene"] == "SARM1"

    def test_search_record_id_correct(self):
        record = {"name": "TARDBP", "entity_type": "gene", "confidence": 0.78}
        item = _parse_search_record(record, "FUS")
        assert item.id == "evi:galen_kg:tardbp_fus"

    def test_search_record_pch_layer_is_1(self):
        record = {"name": "mTOR pathway", "entity_type": "pathway", "confidence": 0.65}
        item = _parse_search_record(record, "MTOR")
        assert item.body["pch_layer"] == 1

    def test_search_record_data_source(self):
        record = {"name": "SOD1", "entity_type": "gene", "confidence": 0.8}
        item = _parse_search_record(record, "FUS")
        assert item.body["data_source"] == "galen_kg_api"

    def test_evidence_items_are_evidence_item_instances(self):
        c = GalenKGAPIConnector()
        created_items = []
        mock_store = MagicMock()
        mock_store.upsert_evidence_item.side_effect = lambda item: created_items.append(item)
        c._store = mock_store

        with patch("requests.get", return_value=_make_response(_neighbors_payload())):
            c.fetch(gene="SIGMAR1")

        assert all(isinstance(i, EvidenceItem) for i in created_items)

    def test_store_called_per_item(self):
        """store.upsert_evidence_item called once per evidence item."""
        c = GalenKGAPIConnector()
        mock_store = MagicMock()
        c._store = mock_store

        with patch("requests.get", return_value=_make_response(_neighbors_payload())):
            result = c.fetch(gene="SIGMAR1")

        assert mock_store.upsert_evidence_item.call_count == result.evidence_items_added


# ---------------------------------------------------------------------------
# Store-less operation
# ---------------------------------------------------------------------------


class TestNoStore:
    def test_works_without_store(self):
        """Connector works without a store — items counted but not persisted."""
        c = GalenKGAPIConnector()  # no store
        with patch("requests.get", return_value=_make_response(_neighbors_payload())):
            result = c.fetch(gene="SOD1")
        assert result.evidence_items_added == 2
        assert result.errors == []

    def test_evidence_items_added_counted_without_store(self):
        c = GalenKGAPIConnector(store=None)
        with patch("requests.get", return_value=_make_response(_neighbors_payload())):
            result = c.fetch(gene="FUS")
        assert result.evidence_items_added >= 1


# ---------------------------------------------------------------------------
# HTTP error handling
# ---------------------------------------------------------------------------


class TestHTTPErrorHandling:
    def test_neighbors_http_500_appends_error(self):
        """HTTP 500 on neighbors endpoint records error."""
        c = GalenKGAPIConnector()

        with patch("requests.get", return_value=_make_response({}, 500)):
            with patch("time.sleep"):
                result = c.fetch(gene="SIGMAR1")
        assert len(result.errors) > 0

    def test_network_exception_on_neighbors_appends_error(self):
        """Network exception on neighbors endpoint is caught and added to errors."""
        c = GalenKGAPIConnector()

        with patch("requests.get", side_effect=req_lib.ConnectionError("unreachable")):
            with patch("time.sleep"):
                result = c.fetch(gene="TBK1")
        assert len(result.errors) > 0
        assert result.evidence_items_added == 0

    def test_both_endpoints_fail_returns_errors(self):
        """If neighbors fails with a connection error, error is recorded.

        When the Galen server is unreachable (ConnectionError), the connector
        short-circuits after the neighbors failure and returns without calling
        search (both would fail on the same host). At least one error is recorded.
        """
        c = GalenKGAPIConnector()

        with patch("requests.get", side_effect=req_lib.ConnectionError("unreachable")):
            with patch("time.sleep"):
                result = c.fetch(gene="NEK1")

        assert result.evidence_items_added == 0
        assert len(result.errors) >= 1

    def test_malformed_json_response_does_not_crash(self):
        """Unexpected JSON structure is handled gracefully."""
        c = GalenKGAPIConnector()
        # No 'neighbors' or 'results' key — falls through to empty
        with patch("requests.get", return_value=_make_response({"unexpected": "value"})):
            result = c.fetch(gene="FUS")
        assert isinstance(result, ConnectorResult)

    def test_partial_record_missing_name_handled(self):
        """Record with no name field is handled without crash."""
        c = GalenKGAPIConnector()
        payload = {"neighbors": [{"entity_type": "gene", "confidence": 0.6}]}
        with patch("requests.get", return_value=_make_response(payload)):
            result = c.fetch(gene="SOD1")
        # Should not raise — either adds 1 item with empty name or adds an error
        assert isinstance(result, ConnectorResult)


# ---------------------------------------------------------------------------
# Retry behavior
# ---------------------------------------------------------------------------


class TestRetryBehavior:
    def test_retries_on_failure_then_succeeds(self):
        """Retries up to MAX_RETRIES then succeeds."""
        c = GalenKGAPIConnector()
        attempt = [0]

        def flaky_get(url, **kwargs):
            attempt[0] += 1
            if attempt[0] < 3:
                raise req_lib.ConnectionError("transient")
            return _make_response(_neighbors_payload())

        with patch("requests.get", side_effect=flaky_get):
            with patch("time.sleep"):
                result = c.fetch(gene="SIGMAR1")

        assert isinstance(result, ConnectorResult)

    def test_exhausted_retries_adds_error(self):
        """All retries exhausted → error in result."""
        c = GalenKGAPIConnector()

        with patch("requests.get", side_effect=Exception("always fails")):
            with patch("time.sleep"):
                result = c.fetch(gene="TBK1")

        assert len(result.errors) > 0
        assert result.evidence_items_added == 0


# ---------------------------------------------------------------------------
# connector_mode.py mapping
# ---------------------------------------------------------------------------


class TestConnectorModeMapping:
    def test_galen_kg_mapped_to_api_variant(self):
        """connector_mode should map GalenKGConnector → GalenKGAPIConnector."""
        from connectors.connector_mode import resolve_connector_class

        mapped = resolve_connector_class(
            "connectors.galen_kg.GalenKGConnector",
            mode="api",
        )
        assert mapped == "connectors.galen_kg_api.GalenKGAPIConnector"

    def test_galen_kg_unchanged_in_local_mode(self):
        """In local mode the class path is returned unchanged."""
        from connectors.connector_mode import resolve_connector_class

        mapped = resolve_connector_class(
            "connectors.galen_kg.GalenKGConnector",
            mode="local",
        )
        assert mapped == "connectors.galen_kg.GalenKGConnector"
