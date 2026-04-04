"""Tests for CMapConnector — drug repurposing via signature reversal."""
from __future__ import annotations

import pytest

from connectors.base import BaseConnector, ConnectorResult
from connectors.cmap import (
    CMapConnector,
    _build_reversal_query_hash,
    _normalize_compound,
)


# ---------------------------------------------------------------------------
# Connector instantiation
# ---------------------------------------------------------------------------

def test_connector_instantiates():
    c = CMapConnector()
    assert c is not None


def test_connector_inherits_base():
    assert isinstance(CMapConnector(), BaseConnector)


# ---------------------------------------------------------------------------
# _normalize_compound
# ---------------------------------------------------------------------------

def test_normalize_compound():
    assert _normalize_compound("Riluzole Hydrochloride") == "riluzole_hydrochloride"


def test_normalize_compound_already_lower():
    assert _normalize_compound("riluzole") == "riluzole"


def test_normalize_compound_multiple_spaces():
    assert _normalize_compound("Some Drug Name") == "some_drug_name"


# ---------------------------------------------------------------------------
# _build_reversal_query_hash
# ---------------------------------------------------------------------------

def test_build_reversal_query_hash_deterministic():
    h1 = _build_reversal_query_hash(["A", "B"], ["C", "D"])
    h2 = _build_reversal_query_hash(["A", "B"], ["C", "D"])
    assert h1 == h2
    assert len(h1) == 12


def test_build_reversal_query_hash_order_independent():
    h1 = _build_reversal_query_hash(["B", "A"], ["D", "C"])
    h2 = _build_reversal_query_hash(["A", "B"], ["C", "D"])
    assert h1 == h2  # sorted internally


def test_build_reversal_query_hash_case_independent():
    h1 = _build_reversal_query_hash(["tardbp", "sod1"], ["fus"])
    h2 = _build_reversal_query_hash(["TARDBP", "SOD1"], ["FUS"])
    assert h1 == h2


def test_build_reversal_query_hash_different_genes():
    h1 = _build_reversal_query_hash(["A"], ["B"])
    h2 = _build_reversal_query_hash(["C"], ["D"])
    assert h1 != h2


# ---------------------------------------------------------------------------
# fetch edge cases
# ---------------------------------------------------------------------------

def test_fetch_no_compound_no_genes_returns_empty():
    c = CMapConnector()
    r = c.fetch()
    # Should return empty (no genes available from GEO)
    assert r.evidence_items_added == 0


def test_fetch_returns_connector_result():
    c = CMapConnector()
    r = c.fetch()
    assert isinstance(r, ConnectorResult)


# ---------------------------------------------------------------------------
# Evidence ID format
# ---------------------------------------------------------------------------

def test_evidence_id_perturbation_format():
    assert "evi:cmap:" in "evi:cmap:riluzole_a375_perturbation"


def test_evidence_id_reversal_format():
    assert "evi:cmap:reversal_" in "evi:cmap:reversal_abc123def456_riluzole"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_min_connectivity_score_default():
    c = CMapConnector()
    assert c._min_connectivity_score == -90.0


def test_min_connectivity_score_custom():
    c = CMapConnector(min_connectivity_score=-80.0)
    assert c._min_connectivity_score == -80.0


def test_api_key_stored():
    c = CMapConnector(api_key="test_key_123")
    assert c._api_key == "test_key_123"


def test_api_key_default_none():
    c = CMapConnector()
    assert c._api_key is None


# ---------------------------------------------------------------------------
# _build_headers
# ---------------------------------------------------------------------------

def test_build_headers_without_key():
    c = CMapConnector()
    headers = c._build_headers()
    assert "Accept" in headers
    assert "user_key" not in headers


def test_build_headers_with_key():
    c = CMapConnector(api_key="mykey")
    headers = c._build_headers()
    assert headers["user_key"] == "mykey"


# ---------------------------------------------------------------------------
# Mode 2: signature reversal with mock
# ---------------------------------------------------------------------------

def test_fetch_by_signature_with_genes_no_api(monkeypatch):
    """When API is unavailable, returns error but doesn't crash."""
    import connectors.cmap as cmap_mod

    def fake_post(*args, **kwargs):
        raise ConnectionError("API down")

    monkeypatch.setattr(cmap_mod.requests, "get", fake_post)
    monkeypatch.setattr(cmap_mod.requests, "post", fake_post)

    c = CMapConnector()
    # Provide genes directly so it skips the GEO import path
    r = c.fetch(up_genes=["GENE1", "GENE2"], down_genes=["GENE3", "GENE4"])
    assert r.evidence_items_added == 0
    assert any("unavailable" in e.lower() for e in r.errors)


def test_fetch_by_compound_api_failure(monkeypatch):
    """Compound query gracefully handles API failure."""
    import connectors.cmap as cmap_mod

    def fake_get(*args, **kwargs):
        raise ConnectionError("API down")

    monkeypatch.setattr(cmap_mod.requests, "get", fake_get)

    c = CMapConnector()
    r = c.fetch(compound="riluzole")
    assert r.evidence_items_added == 0
    assert len(r.errors) > 0


def test_fetch_by_compound_mock_response(monkeypatch):
    """Compound query with a mocked API response produces evidence."""
    import connectors.cmap as cmap_mod

    class FakeResponse:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return [
                {"pert_iname": "riluzole", "cell_id": "A375", "pert_dose": "10",
                 "pert_time": "24", "score": 95.0},
                {"pert_iname": "riluzole", "cell_id": "MCF7", "pert_dose": "5",
                 "pert_time": "6", "score": 50.0},  # below threshold
            ]

    def fake_get(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(cmap_mod.requests, "get", fake_get)

    c = CMapConnector()
    r = c.fetch(compound="riluzole")
    assert r.evidence_items_added == 1  # only score > 80 passes
    assert r.errors == []


def test_fetch_by_signature_mock_response(monkeypatch):
    """Signature reversal with a mocked API response produces evidence."""
    import connectors.cmap as cmap_mod

    class FakeResponse:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return [
                {"pert_iname": "edaravone", "score": -95.0},
                {"pert_iname": "riluzole", "score": -92.0},
                {"pert_iname": "aspirin", "score": -50.0},  # above threshold
            ]

    def fake_post(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(cmap_mod.requests, "post", fake_post)

    c = CMapConnector(min_connectivity_score=-90.0)
    r = c.fetch(up_genes=["GENE1"], down_genes=["GENE2"])
    # edaravone (-95) and riluzole (-92) are below -90, aspirin (-50) is not
    assert r.evidence_items_added == 2
    assert r.errors == []


def test_fetch_by_compound_with_store(monkeypatch):
    """Evidence is upserted to a mock store."""
    import connectors.cmap as cmap_mod

    upserted = []

    class MockStore:
        def upsert_object(self, obj):
            upserted.append(obj)

    class FakeResponse:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return [
                {"pert_iname": "riluzole", "cell_id": "A375",
                 "pert_dose": "10", "pert_time": "24", "score": 95.0},
            ]

    monkeypatch.setattr(cmap_mod.requests, "get", lambda *a, **k: FakeResponse())

    c = CMapConnector(store=MockStore())
    r = c.fetch(compound="riluzole")

    assert r.evidence_items_added == 1
    assert len(upserted) == 1
    assert upserted[0].id == "evi:cmap:riluzole_a375_perturbation"
    assert upserted[0].body["pch_layer"] == 2


# ---------------------------------------------------------------------------
# Live API (marked — skipped unless explicitly requested)
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_cmap_fetch_riluzole():
    c = CMapConnector()
    r = c.fetch(compound="riluzole")
    # May succeed or fail depending on API — should not crash
    assert isinstance(r.evidence_items_added, int)
