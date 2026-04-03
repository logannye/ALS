"""Activity feed endpoint — audit log of research events."""
from __future__ import annotations

import json

from fastapi import APIRouter, Query

from db.pool import get_connection

router = APIRouter(prefix="/api")


@router.get("/activity")
def list_activity(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    event_type: str | None = Query(None),
):
    """Return paginated audit events, most recent first."""
    conditions = ["1=1"]
    params: list = []

    if event_type:
        conditions.append("event_type = %s")
        params.append(event_type)

    where = " AND ".join(conditions)
    params.extend([limit, offset])

    with get_connection() as conn:
        rows = conn.execute(
            f"""SELECT id, event_type, object_id, object_type, actor, details, created_at
                FROM erik_ops.audit_events
                WHERE {where}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s""",
            params,
        ).fetchall()

    events = []
    for row in rows:
        eid, etype, obj_id, obj_type, actor, details, created = row
        if isinstance(details, str):
            details = json.loads(details)
        events.append({
            "id": eid,
            "event_type": etype,
            "object_id": obj_id,
            "object_type": obj_type,
            "actor": actor,
            "details": details,
            "created_at": str(created),
        })

    return {"events": events, "limit": limit, "offset": offset}
