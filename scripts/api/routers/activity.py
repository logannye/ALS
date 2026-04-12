"""Activity feed endpoint — research loop events from activity_feed."""
from __future__ import annotations

from fastapi import APIRouter, Query

from db.pool import get_connection

router = APIRouter(prefix="/api")


@router.get("/activity")
def get_activity(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Return paginated activity feed events, most recent first."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT phase, event_type, summary, details, created_at
                FROM erik_ops.activity_feed
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """, (limit, offset))
            rows = cur.fetchall()
    events = [
        {"phase": r[0], "event_type": r[1], "summary": r[2],
         "details": r[3], "timestamp": r[4].isoformat() if r[4] else None}
        for r in rows
    ]
    return {"events": events, "limit": limit, "offset": offset}
