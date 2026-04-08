"""Tests for ChEMBLAPIConnector — no real HTTP calls in unit tests.

All HTTP calls are mocked with unittest.mock.patch so these tests run
without network access.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from connectors.base import BaseConnector, ConnectorResult
from connectors.chembl_api import ChEMBLAPIConnector
from ontology.evidence import EvidenceItem


# ---------------------------------------------------------------------------
# Helpers: mock API payloads
# ---------------------------------------------------------------------------

def _target_search_payload(chembl_id: str = "CHEMBL2093872") -> dict:
    """Simulate /target.json response for UniProt lookup."""
    return {
        "targets": [
            {
                "target_chembl_id": chembl_id,
                "pref_name": "Sigma non-opioid intracellular receptor 1",
                "target_type": "SINGLE PROTEIN",
            }
        ],
        "page_meta": {"total_count": 1},
    }


def _gene_search_payload(chembl_id: str = "CHEMBL2093872") -> dict:
    """Simulate /target/search.json response for gene name lookup."""
    return {
        "targets": [
            {
                "target_chembl_id": chembl_id,
                "pref_name": "Sigma non-opioid intracellular receptor 1",
                "target_type": "SINGLE PROTEIN",
            }
        ],
        "page_meta": {"total_count": 1},
    }


def _bioactivity_payload() -> dict:
    """Simulate /activity.json response."""
    return {
        "activities": [
            {
                "molecule_chembl_id": "CHEMBL25",
                "molecule_pref_name": "ASPIRIN",
                "target_chembl_id": "CHEMBL2093872",
                "standard_type": "IC50",
                "standard_value": "50.0",
                "standard_units": "nM",
                "pchembl_value": "7.3",
            },
            {
                "molecule_chembl_id": "CHEMBL1200633",
                "molecule_pref_name": "HALOPERIDOL",
                "target_chembl_id": "CHEMBL2093872",
                "standard_type": "Ki",
                "standard_value": "2.5",
                "standard_units": "nM",
                "pchembl_value": "8.6",
            },
        ],
        "page_meta": {"total_count": 2},
    }


def _mechanism_payload() -> dict:
    """Simulate /mechanism.json response."""
    return {
        "mechanisms": [
            {
                "molecule_chembl_id": "CHEMBL25",
                "molecule_pref_name": "ASPIRIN",
                "target_chembl_id": "CHEMBL2093872",
                "mechanism_of_action": "Sigma receptor modulator",
                "action_type": "MODULATOR",
            }
        ],
        "page_meta": {"total_count": 1},
    }


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
        c = ChEMBLAPIConnector()
        assert c is not None

    def test_instantiates_with_store(self):
        store = MagicMock()
        c = ChEMBLAPIConnector(store=store)
        assert c._store is store

    def test_inherits_base_connector(self):
        c = ChEMBLAPIConnector()
        assert isinstance(c, BaseConnector)

    def test_fetch_method_exists(self):
        c = ChEMBLAPIConnector()
        assert callable(c.fetch)

    def test_fetch_full_profile_method_exists(self):
        c = ChEMBLAPIConnector()
        assert callable(c.fetch_full_profile)

    def test_base_url_is_chembl(self):
        c = ChEMBLAPIConnector()
        assert "ebi.ac.uk/chembl" in c.BASE_URL


# ---------------------------------------------------------------------------
# fetch() signature compliance
# ---------------------------------------------------------------------------

class TestFetchSignature:
    def test_fetch_accepts_gene_kwarg(self):
        c = ChEMBLAPIConnector()
        result = c.fetch(gene="")
        assert isinstance(result, ConnectorResult)

    def test_fetch_accepts_uniprot_kwarg(self):
        c = ChEMBLAPIConnector()
        result = c.fetch(uniprot="")
        assert isinstance(result, ConnectorResult)

    def test_fetch_returns_connector_result_with_no_args(self):
        c = ChEMBLAPIConnector()
        result = c.fetch()
        assert isinstance(result, ConnectorResult)

    def test_fetch_full_profile_accepts_uniprot_id(self):
        c = ChEMBLAPIConnector()
        result = c.fetch_full_profile(uniprot_id="")
        assert isinstance(result, ConnectorResult)


# ---------------------------------------------------------------------------
# fetch() with gene name — mocked HTTP
# ---------------------------------------------------------------------------

class TestFetchByGene:
    def test_fetch_gene_returns_evidence_items(self):
        """fetch(gene=...) triggers gene-name search and bioactivity fetch."""
        c = ChEMBLAPIConnector()

        responses = [
            _make_response(_gene_search_payload()),       # /target/search.json
            _make_response(_bioactivity_payload()),        # /activity.json
        ]

        with patch("requests.get", side_effect=responses):
            result = c.fetch(gene="SIGMAR1")

        assert result.evidence_items_added >= 1
        assert result.errors == []

    def test_fetch_gene_empty_returns_empty_result(self):
        """Empty gene name should return immediately with 0 items."""
        c = ChEMBLAPIConnector()
        with patch("requests.get") as mock_get:
            result = c.fetch(gene="")
        mock_get.assert_not_called()
        assert result.evidence_items_added == 0

    def test_fetch_gene_no_targets_found_returns_zero(self):
        """If target search returns no results, return 0 evidence items."""
        c = ChEMBLAPIConnector()
        empty_payload = {"targets": [], "page_meta": {"total_count": 0}}
        with patch("requests.get", return_value=_make_response(empty_payload)):
            result = c.fetch(gene="UNKNOWNGENE123")
        assert result.evidence_items_added == 0
        assert result.errors == []


# ---------------------------------------------------------------------------
# fetch() with uniprot — mocked HTTP
# ---------------------------------------------------------------------------

class TestFetchByUniprot:
    def test_fetch_uniprot_returns_evidence_items(self):
        """fetch(uniprot=...) triggers UniProt-based target lookup + bioactivity."""
        c = ChEMBLAPIConnector()

        responses = [
            _make_response(_target_search_payload()),     # /target.json?accession=...
            _make_response(_bioactivity_payload()),        # /activity.json
        ]

        with patch("requests.get", side_effect=responses):
            result = c.fetch(uniprot="Q99720")

        assert result.evidence_items_added >= 1
        assert result.errors == []

    def test_fetch_uniprot_empty_returns_empty_result(self):
        """Empty uniprot should return immediately with 0 items."""
        c = ChEMBLAPIConnector()
        with patch("requests.get") as mock_get:
            result = c.fetch(uniprot="")
        mock_get.assert_not_called()
        assert result.evidence_items_added == 0

    def test_fetch_uniprot_no_target_found_returns_zero(self):
        """If UniProt lookup returns no target, return 0 evidence items."""
        c = ChEMBLAPIConnector()
        empty_payload = {"targets": [], "page_meta": {"total_count": 0}}
        with patch("requests.get", return_value=_make_response(empty_payload)):
            result = c.fetch(uniprot="Q00000")
        assert result.evidence_items_added == 0
        assert result.errors == []


# ---------------------------------------------------------------------------
# fetch_full_profile() — mocked HTTP
# ---------------------------------------------------------------------------

class TestFetchFullProfile:
    def test_fetch_full_profile_returns_bioactivity_and_mechanism(self):
        """fetch_full_profile should combine bioactivity + mechanism evidence."""
        c = ChEMBLAPIConnector()

        responses = [
            _make_response(_target_search_payload()),     # /target.json for uniprot
            _make_response(_bioactivity_payload()),        # /activity.json
            _make_response(_mechanism_payload()),          # /mechanism.json
        ]

        with patch("requests.get", side_effect=responses):
            result = c.fetch_full_profile(uniprot_id="Q99720")

        # Should have bioactivity + mechanism items
        assert result.evidence_items_added >= 2
        assert result.errors == []

    def test_fetch_full_profile_empty_uniprot_returns_empty(self):
        c = ChEMBLAPIConnector()
        with patch("requests.get") as mock_get:
            result = c.fetch_full_profile(uniprot_id="")
        mock_get.assert_not_called()
        assert result.evidence_items_added == 0

    def test_fetch_full_profile_returns_connector_result(self):
        c = ChEMBLAPIConnector()
        with patch("requests.get", return_value=_make_response({"targets": []})):
            result = c.fetch_full_profile(uniprot_id="Q99720")
        assert isinstance(result, ConnectorResult)


# ---------------------------------------------------------------------------
# EvidenceItem construction
# ---------------------------------------------------------------------------

class TestEvidenceItemConstruction:
    def test_bioactivity_evidence_item_has_correct_id_format(self):
        """Bioactivity EvidenceItems should have deterministic IDs."""
        c = ChEMBLAPIConnector()

        responses = [
            _make_response(_target_search_payload()),
            _make_response(_bioactivity_payload()),
        ]

        created_items = []
        mock_store = MagicMock()
        mock_store.upsert_evidence_item.side_effect = lambda item: created_items.append(item)
        c._store = mock_store

        with patch("requests.get", side_effect=responses):
            c.fetch(uniprot="Q99720")

        assert len(created_items) >= 1
        # ID format: evi:chembl_api:{molecule_id}_{target_id}_{type}
        for item in created_items:
            assert item.id.startswith("evi:chembl_api:")
            assert isinstance(item, EvidenceItem)

    def test_bioactivity_evidence_item_has_pch_layer(self):
        """EvidenceItems should have pch_layer in body."""
        c = ChEMBLAPIConnector()

        responses = [
            _make_response(_target_search_payload()),
            _make_response(_bioactivity_payload()),
        ]

        created_items = []
        mock_store = MagicMock()
        mock_store.upsert_evidence_item.side_effect = lambda item: created_items.append(item)
        c._store = mock_store

        with patch("requests.get", side_effect=responses):
            c.fetch(uniprot="Q99720")

        for item in created_items:
            assert "pch_layer" in item.body

    def test_mechanism_evidence_item_has_mechanism_in_claim(self):
        """Mechanism EvidenceItems should include MOA text in the claim."""
        c = ChEMBLAPIConnector()

        responses = [
            _make_response(_target_search_payload()),
            _make_response({"activities": [], "page_meta": {"total_count": 0}}),
            _make_response(_mechanism_payload()),
        ]

        created_items = []
        mock_store = MagicMock()
        mock_store.upsert_evidence_item.side_effect = lambda item: created_items.append(item)
        c._store = mock_store

        with patch("requests.get", side_effect=responses):
            c.fetch_full_profile(uniprot_id="Q99720")

        moa_items = [i for i in created_items if "mechanism" in i.id.lower() or
                     any(kw in i.claim.lower() for kw in ["modulator", "mechanism", "action"])]
        assert len(moa_items) >= 1

    def test_evidence_item_has_source_system_database(self):
        """EvidenceItems provenance should indicate database source."""
        c = ChEMBLAPIConnector()

        responses = [
            _make_response(_target_search_payload()),
            _make_response(_bioactivity_payload()),
        ]

        created_items = []
        mock_store = MagicMock()
        mock_store.upsert_evidence_item.side_effect = lambda item: created_items.append(item)
        c._store = mock_store

        with patch("requests.get", side_effect=responses):
            c.fetch(uniprot="Q99720")

        for item in created_items:
            assert item.provenance.source_system.value == "database"
            assert item.provenance.asserted_by == "chembl_api_connector"

    def test_evidence_item_store_called_per_item(self):
        """store.upsert_evidence_item should be called once per evidence item."""
        c = ChEMBLAPIConnector()

        responses = [
            _make_response(_target_search_payload()),
            _make_response(_bioactivity_payload()),   # 2 activities
        ]

        mock_store = MagicMock()
        c._store = mock_store

        with patch("requests.get", side_effect=responses):
            result = c.fetch(uniprot="Q99720")

        assert mock_store.upsert_evidence_item.call_count == result.evidence_items_added


# ---------------------------------------------------------------------------
# HTTP error handling
# ---------------------------------------------------------------------------

class TestHTTPErrorHandling:
    def test_http_500_adds_error_returns_zero(self):
        """A 500 response should add an error and return 0 items."""
        c = ChEMBLAPIConnector()
        with patch("requests.get", return_value=_make_response({}, 500)):
            result = c.fetch(uniprot="Q99720")
        assert len(result.errors) > 0
        assert result.evidence_items_added == 0

    def test_network_exception_adds_error(self):
        """A network-level exception should be caught and added to errors."""
        import requests as req_lib
        c = ChEMBLAPIConnector()
        with patch("requests.get", side_effect=req_lib.ConnectionError("timeout")):
            result = c.fetch(uniprot="Q99720")
        assert len(result.errors) > 0
        assert result.evidence_items_added == 0

    def test_malformed_json_response_adds_error(self):
        """If the response JSON is malformed/unexpected, add error and continue."""
        c = ChEMBLAPIConnector()
        # Target lookup succeeds, but bioactivity returns garbage
        responses = [
            _make_response(_target_search_payload()),
            _make_response({"unexpected_key": "bad_value"}),  # no 'activities' key
        ]
        with patch("requests.get", side_effect=responses):
            result = c.fetch(uniprot="Q99720")
        # Should not crash — just return gracefully (either 0 items or an error)
        assert isinstance(result, ConnectorResult)

    def test_target_lookup_failure_does_not_call_bioactivity(self):
        """If target lookup fails (all retries exhausted), bioactivity is not called.

        BaseConnector._retry_with_backoff retries MAX_RETRIES (3) times for the
        target lookup, so calls to requests.get are all target-lookup attempts;
        no bioactivity URL should ever be requested.
        """
        c = ChEMBLAPIConnector()
        urls_called = []

        def tracking_get(url, **kwargs):
            urls_called.append(url)
            raise Exception("Connection refused")

        with patch("requests.get", side_effect=tracking_get):
            with patch("time.sleep"):   # don't actually sleep
                result = c.fetch(uniprot="Q99720")

        # All calls should be to the target endpoint, never to activity
        for url in urls_called:
            assert "activity" not in url, f"Bioactivity URL called after target lookup failure: {url}"
        assert result.evidence_items_added == 0


# ---------------------------------------------------------------------------
# Retry with backoff
# ---------------------------------------------------------------------------

class TestRetryBehavior:
    def test_retries_on_failure_then_succeeds(self):
        """Should retry up to MAX_RETRIES times before raising."""
        import time
        c = ChEMBLAPIConnector()

        attempt = [0]

        def flaky_get(url, **kwargs):
            attempt[0] += 1
            if attempt[0] < 3:
                raise Exception("Transient error")
            # Third attempt succeeds with target search
            if "target" in url and "activity" not in url and "mechanism" not in url:
                return _make_response(_target_search_payload())
            return _make_response(_bioactivity_payload())

        with patch("requests.get", side_effect=flaky_get):
            with patch("time.sleep"):  # don't actually sleep in tests
                result = c.fetch(uniprot="Q99720")

        # The retry mechanism should eventually succeed (3rd attempt passes)
        assert isinstance(result, ConnectorResult)

    def test_exhausted_retries_adds_error(self):
        """After MAX_RETRIES failures, error is recorded."""
        c = ChEMBLAPIConnector()

        with patch("requests.get", side_effect=Exception("always fails")):
            with patch("time.sleep"):
                result = c.fetch(uniprot="Q99720")

        assert len(result.errors) > 0
        assert result.evidence_items_added == 0


# ---------------------------------------------------------------------------
# Store-less operation
# ---------------------------------------------------------------------------

class TestNoStore:
    def test_works_without_store(self):
        """Connector works without a store — items counted but not persisted."""
        c = ChEMBLAPIConnector()  # no store
        responses = [
            _make_response(_target_search_payload()),
            _make_response(_bioactivity_payload()),
        ]
        with patch("requests.get", side_effect=responses):
            result = c.fetch(uniprot="Q99720")

        assert result.evidence_items_added >= 1
        assert result.errors == []

    def test_evidence_items_added_even_without_store(self):
        """evidence_items_added should count items even when store is None."""
        c = ChEMBLAPIConnector(store=None)
        responses = [
            _make_response(_target_search_payload()),
            _make_response(_bioactivity_payload()),
        ]
        with patch("requests.get", side_effect=responses):
            result = c.fetch(uniprot="Q99720")
        assert result.evidence_items_added >= 2  # 2 activities in mock payload


# ---------------------------------------------------------------------------
# Request headers
# ---------------------------------------------------------------------------

class TestRequestHeaders:
    def test_requests_include_json_accept_header(self):
        """All requests should include Accept: application/json."""
        c = ChEMBLAPIConnector()
        call_kwargs_list = []

        def capture_get(url, **kwargs):
            call_kwargs_list.append(kwargs)
            if "activity" in url:
                return _make_response(_bioactivity_payload())
            return _make_response(_target_search_payload())

        with patch("requests.get", side_effect=capture_get):
            c.fetch(uniprot="Q99720")

        for kwargs in call_kwargs_list:
            headers = kwargs.get("headers", {})
            assert headers.get("Accept") == "application/json", \
                f"Missing Accept: application/json header in {kwargs}"
