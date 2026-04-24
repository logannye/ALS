"""Tests for the auth middleware + health endpoints.

Uses FastAPI's TestClient. Exercises the middleware directly rather than
going through a real HTTP server so tests are hermetic and fast.
"""
from __future__ import annotations

import os

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
    """Session-scoped TestClient to amortize startup cost.

    FastAPI's lifespan runs migrations + startup hooks on every
    ``with TestClient(app)`` context entry. Running that per-test blows
    up wall time and occasionally deadlocks under contention. Scoping
    the client at module level means lifespan runs exactly once.

    https://testserver is required so starlette treats the connection as
    secure — our session cookie is ``secure=True`` (correct for prod)
    and would be dropped by TestClient over plain http://testserver.
    """
    os.environ["ERIK_INVITE_CODES"] = "TEST_CODE_1,TEST_CODE_2"
    os.environ["ERIK_RESEARCH_LOOP"] = "false"
    # Avoid running full migrations in test — the DB is already migrated
    # by CI / dev setup. See api/main.py lifespan.
    os.environ.setdefault("ERIK_SKIP_MIGRATIONS", "true")
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app, base_url="https://testserver") as c:
        yield c


@pytest.fixture(autouse=True)
def _isolate_cookies(client):
    """Clear per-test cookies so auth state doesn't leak across tests."""
    client.cookies.clear()
    yield


# ─── Public endpoints — no auth required ─────────────────────────────────────


@pg
def test_health_is_public(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] in ("ok", "degraded")


@pg
def test_health_research_is_public(client):
    """Railway healthcheck must be able to hit this without a cookie."""
    r = client.get("/health/research")
    # 200 (fresh) or 503 (stalled) both acceptable — the point is it's not 401.
    assert r.status_code in (200, 503)


@pg
def test_auth_redeem_is_public(client):
    """Unauthenticated POST to /api/auth/redeem must reach the handler."""
    r = client.post("/api/auth/redeem", json={"code": "WRONG_CODE"})
    # 403 = invalid code reached the handler (good); 401 would mean middleware
    # blocked us before the handler ran, which is the bug we're guarding against.
    assert r.status_code == 403


# ─── Protected endpoints — auth required ─────────────────────────────────────


_PROTECTED_CANDIDATES = [
    "/api/activity",
    "/api/state",
    "/api/evidence",
    "/api/discoveries",
    "/api/activity",
]


@pg
@pytest.mark.parametrize("path", _PROTECTED_CANDIDATES)
def test_protected_path_without_cookie_is_401(client, path):
    r = client.get(path)
    assert r.status_code == 401, f"{path} should be guarded; got {r.status_code}"


@pg
def test_protected_path_with_valid_session_is_not_401(client):
    """Redeem a code, then re-use the cookie to hit a protected endpoint."""
    r = client.post("/api/auth/redeem", json={"code": "TEST_CODE_1", "name": "pytest"})
    assert r.status_code == 200
    # TestClient persists cookies automatically.
    r2 = client.get("/api/activity")
    # 200 or 500 (upstream error) both prove the middleware let us through —
    # a 401 would fail this test, which is what we care about.
    assert r2.status_code != 401


@pg
def test_cors_preflight_never_blocked(client):
    """OPTIONS requests must never return 401 — browsers reject the
    subsequent real request if the preflight fails."""
    r = client.options(
        "/api/activity",
        headers={
            "Origin": "https://erik-website-eosin.vercel.app",
            "Access-Control-Request-Method": "GET",
        },
    )
    # FastAPI returns 200 for handled OPTIONS; 405 is also acceptable
    # (some routers don't register OPTIONS). Never 401.
    assert r.status_code != 401


@pg
def test_invalid_session_cookie_rejected(client):
    """A cookie with a bogus token value gets 401, not a silent bypass."""
    client.cookies.set("erik_session", "not_a_real_token")
    r = client.get("/api/activity")
    assert r.status_code == 401
