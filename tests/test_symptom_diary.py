"""Tests for the family symptom-diary endpoints.

Validates POST rejects empty, accepts valid, extracts tags, and that
GET returns rows newest-first.
"""
from __future__ import annotations

import os
import uuid

import pytest


def _can_connect() -> bool:
    import psycopg
    user = os.environ.get("USER", "logannye")
    try:
        c = psycopg.connect(f"dbname=erik_kg user={user}", connect_timeout=3)
        c.close()
        return True
    except Exception:
        return False


pg = pytest.mark.skipif(not _can_connect(), reason="erik_kg PG not reachable")


@pytest.fixture(scope='module')
def client():
    os.environ["ERIK_RESEARCH_LOOP"] = "false"
    os.environ.setdefault("ERIK_SKIP_MIGRATIONS", "true")
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app, base_url="https://testserver") as c:
        yield c


# ─── Pure tag extractor ──────────────────────────────────────────────────────


def test_extract_tags_finds_common_symptoms():
    from api.routers.symptom_diary import _extract_tags
    tags = _extract_tags("Dad seemed really weak today and had trouble swallowing.")
    assert "weakness" in tags
    assert "swallowing" in tags


def test_extract_tags_empty_on_empty_note():
    from api.routers.symptom_diary import _extract_tags
    assert _extract_tags("") == []
    assert _extract_tags("   ") == []


def test_extract_tags_returns_sorted():
    """Callers should get deterministic tag order."""
    from api.routers.symptom_diary import _extract_tags
    tags = _extract_tags("Trouble walking and twitches in his arm.")
    assert tags == sorted(tags)


# ─── POST /api/symptom-report ────────────────────────────────────────────────


@pg
def test_post_rejects_empty_report(client):
    r = client.post("/api/symptom-report", json={})
    assert r.status_code == 400


@pg
def test_post_rejects_whitespace_only_note_with_no_mood(client):
    r = client.post("/api/symptom-report", json={"note": "   "})
    assert r.status_code == 400


@pg
def test_post_accepts_mood_only(client):
    r = client.post("/api/symptom-report", json={"mood": 3, "reporter_name": "pytest"})
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert isinstance(data["id"], int)


@pg
def test_post_accepts_note_only(client):
    r = client.post("/api/symptom-report", json={
        "note": f"Test {uuid.uuid4().hex[:6]} — had some cramping in his calves tonight.",
        "reporter_name": "pytest",
    })
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert "cramping" in data["symptoms_mentioned"]


@pg
def test_post_rejects_mood_out_of_range(client):
    r = client.post("/api/symptom-report", json={"mood": 6})
    assert r.status_code == 422


@pg
def test_post_enforces_note_length_cap(client):
    r = client.post("/api/symptom-report", json={
        "note": "x" * 5000,
        "reporter_name": "pytest",
    })
    assert r.status_code == 422


# ─── GET /api/symptom-report/recent ──────────────────────────────────────────


@pg
def test_recent_returns_newest_first(client):
    # Seed two reports.
    tag1, tag2 = uuid.uuid4().hex[:6], uuid.uuid4().hex[:6]
    client.post("/api/symptom-report", json={
        "mood": 2, "note": f"First {tag1}.", "reporter_name": "pytest",
    })
    client.post("/api/symptom-report", json={
        "mood": 4, "note": f"Second {tag2}.", "reporter_name": "pytest",
    })

    r = client.get("/api/symptom-report/recent?days=1&limit=10")
    assert r.status_code == 200
    reports = r.json()["reports"]
    # newest first — find our tags in correct order.
    notes = [rep["note"] for rep in reports]
    idx1 = next((i for i, n in enumerate(notes) if tag1 in n), None)
    idx2 = next((i for i, n in enumerate(notes) if tag2 in n), None)
    assert idx1 is not None and idx2 is not None
    assert idx2 < idx1, "second report should be newer and appear earlier"


@pg
def test_recent_tolerates_missing_params(client):
    r = client.get("/api/symptom-report/recent")
    assert r.status_code == 200
    assert "reports" in r.json()


# ─── Append-only schema trigger ──────────────────────────────────────────────


@pg
def test_delete_forbidden_by_trigger():
    import psycopg
    user = os.environ.get("USER", "logannye")
    c = psycopg.connect(f"dbname=erik_kg user={user}")
    with c.cursor() as cur:
        cur.execute(
            "INSERT INTO erik_ops.symptom_reports(mood, note, reporter_name) "
            "VALUES (3, 'trigger test', 'pytest') RETURNING id"
        )
        rid = cur.fetchone()[0]
        c.commit()
    with pytest.raises(psycopg.errors.RaiseException):
        with c.cursor() as cur:
            cur.execute("DELETE FROM erik_ops.symptom_reports WHERE id = %s", (rid,))
    c.rollback()
    c.close()
