"""Tests for the family-facing API endpoints added in the 2026-04-24 UX pass.

Covers /api/trajectory, /api/top-candidate, /api/current-hypothesis,
/api/galen-believes, the rewritten /api/trials, and the rewritten
/api/discoveries. DB-gated — skips if erik_kg isn't reachable.
"""
from __future__ import annotations

import os
from datetime import date

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
    os.environ["ERIK_INVITE_CODES"] = "TEST_CODE_1"
    os.environ["ERIK_RESEARCH_LOOP"] = "false"
    os.environ.setdefault("ERIK_SKIP_MIGRATIONS", "true")
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app, base_url="https://testserver") as c:
        yield c


# ─── /api/trajectory ─────────────────────────────────────────────────────────


@pg
def test_trajectory_returns_three_series(client):
    r = client.get("/api/trajectory")
    assert r.status_code == 200
    data = r.json()
    for key in ("alsfrs_r", "nfl_pg_ml", "fvc_pct"):
        assert key in data
        assert "current" in data[key]
        assert "points" in data[key]
        assert isinstance(data[key]["points"], list)


@pg
def test_trajectory_summary_string_when_data_present(client):
    r = client.get("/api/trajectory").json()
    for series in r.values():
        if series["current"] is not None:
            assert isinstance(series.get("summary"), str)
            assert len(series["summary"]) > 0


# ─── /api/top-candidate ──────────────────────────────────────────────────────


@pg
def test_top_candidate_returns_valid_shape(client):
    r = client.get("/api/top-candidate")
    assert r.status_code == 200
    data = r.json()
    # Must always have one of: candidate OR reason.
    assert data.get("candidate") is not None or data.get("reason") is not None
    if data.get("candidate"):
        c = data["candidate"]
        assert "name" in c
        assert "score" in c and isinstance(c["score"], (int, float))
        assert "layer" in c
        assert "rationale" in c


def test_parse_layer_notes_extracts_drugs_and_scores():
    from api.routers.family import _parse_layer_notes
    out = _parse_layer_notes("Pridopidine (score=0.80); Rapamycin (Sirolimus) (score=0.60)")
    assert len(out) == 2
    assert out[0][0] == "Pridopidine"
    assert abs(out[0][1] - 0.80) < 1e-9
    assert out[1][0] == "Rapamycin (Sirolimus)"
    assert abs(out[1][1] - 0.60) < 1e-9


def test_parse_layer_notes_handles_malformed():
    from api.routers.family import _parse_layer_notes
    assert _parse_layer_notes("") == []
    assert _parse_layer_notes("ABSTENTION: No eligible interventions") == []
    # Partial / missing score → skipped, not raised.
    out = _parse_layer_notes("DrugA; DrugB (score=0.5)")
    assert len(out) == 1
    assert out[0] == ("DrugB", 0.5)


# ─── /api/current-hypothesis ─────────────────────────────────────────────────


@pg
def test_current_hypothesis_returns_shape(client):
    r = client.get("/api/current-hypothesis")
    assert r.status_code == 200
    data = r.json()
    # Null when the DB has no hypotheses at all — acceptable; never 500.
    if data.get("hypothesis"):
        h = data["hypothesis"]
        assert "text" in h
        assert "confidence" in h
        assert "status" in h


# ─── /api/galen-believes ─────────────────────────────────────────────────────


@pg
def test_galen_believes_always_returns_a_belief(client):
    """Even with zero data, the endpoint must return a non-empty belief
    string so the top-of-dashboard card is never blank."""
    r = client.get("/api/galen-believes")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data.get("belief"), str) and len(data["belief"]) > 0
    assert "components" in data


@pg
def test_galen_believes_does_not_leak_placeholder_ids(client):
    """Belief text must not include raw entity ids like 'gene:SOD1' or
    'edge:complement->sod1_misfolding' — those are operator-facing."""
    r = client.get("/api/galen-believes").json()
    belief = r["belief"].lower()
    assert "edge:" not in belief
    assert not belief.startswith("none")


# ─── /api/discoveries (rewritten) ────────────────────────────────────────────


@pg
def test_discoveries_surfaces_real_activity(client):
    r = client.get("/api/discoveries?days=7")
    assert r.status_code == 200
    days = r.json()["days"]
    assert len(days) == 7
    # Invariant: across 7 days, at least ONE day should have a highlight
    # that isn't the quiet-day fallback. This catches the pre-2026-04-24
    # regression where every day said "no major new findings".
    quiet = 0
    for d in days:
        for h in d["highlights"]:
            if h["text"].startswith("Research ran quietly"):
                quiet += 1
                break
    assert quiet < len(days), (
        f"all {len(days)} days reported 'Research ran quietly' — the "
        "metric queries are probably regressed"
    )


@pg
def test_discoveries_highlight_cap(client):
    """Each day is capped at 4 highlights so the cards don't wall-of-text."""
    r = client.get("/api/discoveries?days=3").json()
    for d in r["days"]:
        assert len(d["highlights"]) <= 4


def test_discoveries_pluralise_helper():
    from api.routers.discoveries import _pluralise
    assert _pluralise(1, "trial") == "trial"
    assert _pluralise(2, "trial") == "trials"
    assert _pluralise(2, "entity", "entities") == "entities"


# ─── /api/trials (rewritten) ─────────────────────────────────────────────────


@pg
def test_trials_response_shape(client):
    r = client.get("/api/trials")
    assert r.status_code == 200
    data = r.json()
    assert "trials" in data
    assert "summary" in data
    s = data["summary"]
    assert "total_monitored" in s and isinstance(s["total_monitored"], int)
    assert "by_status" in s


@pg
def test_trials_empty_state_has_message(client):
    """When the list is empty, summary.empty_state_message must explain why."""
    r = client.get("/api/trials").json()
    if not r["trials"]:
        assert isinstance(r["summary"].get("empty_state_message"), str)
        assert len(r["summary"]["empty_state_message"]) > 0
