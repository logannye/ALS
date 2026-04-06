"""Tests for the genetic testing upload endpoint."""
import json
import pytest
from unittest.mock import patch, MagicMock


def _make_client():
    """Create a FastAPI test client with auth bypassed."""
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app, headers={"X-Session-Token": "test-token"})


@pytest.fixture(autouse=True)
def mock_auth():
    """Bypass session validation for all tests in this module."""
    with patch("api.main.validate_session", return_value="test_family_member"):
        yield


@pytest.fixture
def mock_db():
    """Mock the database connection."""
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    with patch("api.routers.genetics.get_connection", return_value=mock_conn):
        yield mock_conn


def test_upload_genetics_valid(mock_db):
    client = _make_client()
    payload = {
        "gene": "SOD1",
        "variant": "G93A",
        "subtype": "SOD1_familial",
        "test_date": "2026-04-01",
        "lab_name": "GeneDx",
    }
    resp = client.post("/api/upload/genetics", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "created"
    assert "SOD1" in data["genetic_profile"]["gene"]
    mock_db.execute.assert_called()
    mock_db.commit.assert_called()


def test_upload_genetics_missing_gene(mock_db):
    client = _make_client()
    payload = {"variant": "G93A", "subtype": "SOD1_familial", "test_date": "2026-04-01"}
    resp = client.post("/api/upload/genetics", json=payload)
    assert resp.status_code == 422


def test_upload_genetics_bad_date(mock_db):
    client = _make_client()
    payload = {
        "gene": "SOD1",
        "variant": "G93A",
        "subtype": "SOD1_familial",
        "test_date": "not-a-date",
    }
    resp = client.post("/api/upload/genetics", json=payload)
    assert resp.status_code == 400


def test_upload_genetics_updates_research_state(mock_db):
    """The endpoint should update the research state's genetic_profile."""
    # Mock fetchone to return existing state
    mock_db.execute.return_value.fetchone.return_value = (json.dumps({"step_count": 100}),)
    client = _make_client()
    payload = {
        "gene": "C9orf72",
        "variant": "repeat_expansion",
        "subtype": "C9orf72",
        "test_date": "2026-04-01",
    }
    resp = client.post("/api/upload/genetics", json=payload)
    assert resp.status_code == 200
    calls = [str(c) for c in mock_db.execute.call_args_list]
    state_update_calls = [c for c in calls if "research_state" in c]
    assert len(state_update_calls) >= 1, "Should update research_state with genetic_profile"
