"""Tests for the auth middleware.

The middleware is currently a **pass-through** — auth is disabled by
product decision (see api/main.py block comment). These tests encode
that invariant so a future re-enablement can't land silently:
every /api/* endpoint must remain reachable without a session cookie.

If the product decision changes and auth is re-enabled, flip the
assertions (the "enabled" version lives in git history at the PR #2
merge commit).
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
    """Session-scoped TestClient to amortize FastAPI lifespan cost."""
    os.environ["ERIK_INVITE_CODES"] = "TEST_CODE_1,TEST_CODE_2"
    os.environ["ERIK_RESEARCH_LOOP"] = "false"
    os.environ.setdefault("ERIK_SKIP_MIGRATIONS", "true")
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app, base_url="https://testserver") as c:
        yield c


# ─── Public endpoints ────────────────────────────────────────────────────────
#
# Still public even when auth is re-enabled — these are the rails Railway's
# healthcheck and Vercel's edge rely on.


@pg
def test_health_is_public(client):
    r = client.get("/health")
    assert r.status_code == 200


@pg
def test_health_research_is_public(client):
    r = client.get("/health/research")
    assert r.status_code in (200, 503)


# ─── /api/* must be open (auth is OFF by product decision) ──────────────────


_API_SURFACE = [
    "/api/state",
    "/api/activity",
    "/api/evidence",
    "/api/discoveries",
]


@pg
@pytest.mark.parametrize("path", _API_SURFACE)
def test_api_endpoint_reachable_without_cookie(client, path):
    """Without a cookie, every /api/* endpoint must reach its handler.

    Acceptable status codes: 200 (happy path), 404 (route not wired in
    this test's context), or 500 (handler-level error). The only
    unacceptable code is 401/403 — those would mean the middleware is
    gating the route, which contradicts the "auth OFF" product decision.
    """
    r = client.get(path)
    assert r.status_code not in (401, 403), (
        f"{path} returned {r.status_code}; auth middleware should be pass-through"
    )


@pg
def test_api_endpoint_reachable_with_bogus_cookie(client):
    """A bogus session cookie must not trigger 401 — the middleware
    should not be validating cookies at all."""
    client.cookies.set("erik_session", "not_a_real_token")
    r = client.get("/api/activity")
    assert r.status_code not in (401, 403)
    client.cookies.clear()


@pg
def test_cors_preflight_never_blocked(client):
    """Preflight must succeed so the browser issues the real request."""
    r = client.options(
        "/api/activity",
        headers={
            "Origin": "https://erik-website-eosin.vercel.app",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r.status_code not in (401, 403)


# ─── Auth scaffolding still present (for future re-enablement) ──────────────


@pg
def test_auth_redeem_endpoint_still_exists(client):
    """/api/auth/redeem is retained so auth can be re-enabled with a
    one-line middleware flip. Unreachable via the middleware today, but
    the handler + invite-code validation logic is preserved."""
    r = client.post("/api/auth/redeem", json={"code": "WRONG_CODE"})
    # 403 = bogus code reached the handler, which means the handler
    # still exists and works. If this ever returned 404 it'd mean the
    # scaffolding was deleted.
    assert r.status_code == 403
