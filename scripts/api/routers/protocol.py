"""Protocol viewer endpoint — returns the latest cure protocol candidate."""
from __future__ import annotations

import json

from fastapi import APIRouter

from db.pool import get_connection

router = APIRouter(prefix="/api")


@router.get("/protocol")
def get_protocol():
    """Return the latest CureProtocolCandidate."""
    with get_connection() as conn:
        row = conn.execute(
            """SELECT id, body, confidence, created_at, updated_at
               FROM erik_core.objects
               WHERE type = 'CureProtocolCandidate' AND status = 'active'
               ORDER BY updated_at DESC
               LIMIT 1"""
        ).fetchone()

    if not row:
        return {"protocol": None}

    obj_id, body, confidence, created, updated = row
    if isinstance(body, str):
        body = json.loads(body)

    return {
        "protocol": {
            "id": obj_id,
            "body": body,
            "confidence": confidence,
            "created_at": str(created),
            "updated_at": str(updated),
        }
    }


@router.get("/protocol/history")
def protocol_history(limit: int = 10):
    """Return recent protocol versions."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, body->>'objective' as objective, confidence, updated_at
               FROM erik_core.objects
               WHERE type = 'CureProtocolCandidate'
               ORDER BY updated_at DESC
               LIMIT %s""",
            (limit,),
        ).fetchall()

    return {
        "versions": [
            {
                "id": r[0],
                "objective": r[1],
                "confidence": r[2],
                "updated_at": str(r[3]),
            }
            for r in rows
        ]
    }
