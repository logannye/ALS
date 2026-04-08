"""Invite-code authentication for the Erik family dashboard.

Flow: family member enters an invite code → POST /api/auth/redeem → session
token returned as httpOnly cookie.  All /api/* routes (except /health and
/api/auth/*) require a valid session.
"""
from __future__ import annotations

import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Cookie, HTTPException, Request

from db.pool import get_connection

# Invite codes are set via ERIK_INVITE_CODES env var (comma-separated)
SESSION_TTL_DAYS = 90


def _get_invite_codes() -> set[str]:
    raw = os.environ.get("ERIK_INVITE_CODES", "")
    return {c.strip() for c in raw.split(",") if c.strip()}


def _ensure_sessions_table() -> None:
    """Create the sessions table if it doesn't exist."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS erik_ops.sessions (
                token TEXT PRIMARY KEY,
                family_member TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                expires_at TIMESTAMPTZ NOT NULL
            )
        """)
        conn.commit()


def redeem_invite_code(code: str, family_member: str = "family") -> str:
    """Validate an invite code and create a session token.

    Returns the session token string on success.
    Raises HTTPException(403) on invalid code.
    """
    valid_codes = _get_invite_codes()
    if code.strip() not in valid_codes:
        raise HTTPException(status_code=403, detail="Invalid invite code")

    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(days=SESSION_TTL_DAYS)

    with get_connection() as conn:
        conn.execute(
            """INSERT INTO erik_ops.sessions (token, family_member, expires_at)
               VALUES (%s, %s, %s)""",
            (token, family_member, expires),
        )
        conn.commit()

    return token


def validate_session(token: Optional[str]) -> str:
    """Check that *token* is a valid, non-expired session.

    Returns the family_member name on success.
    Raises HTTPException(401) on failure.
    """
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    with get_connection() as conn:
        row = conn.execute(
            """SELECT family_member, expires_at FROM erik_ops.sessions
               WHERE token = %s""",
            (token,),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid session")

    family_member, expires_at = row
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) > expires_at:
        raise HTTPException(status_code=401, detail="Session expired")

    return family_member


def get_session_token(request: Request) -> Optional[str]:
    """Extract session token from cookie or header."""
    token = request.cookies.get("erik_session")
    if not token:
        token = request.headers.get("X-Session-Token")
    return token
