"""Evidence explorer endpoints — list, filter, search evidence items."""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Query

from db.pool import get_connection

router = APIRouter(prefix="/api")


@router.get("/evidence")
def list_evidence(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    source: Optional[str] = Query(None, description="Filter by provenance source system"),
    direction: Optional[str] = Query(None, description="supports|refutes|mixed"),
    strength: Optional[str] = Query(None, description="strong|moderate|emerging|preclinical"),
    search: Optional[str] = Query(None, description="Full-text search on claim"),
):
    """Return paginated evidence items with optional filters."""
    conditions = ["type = 'EvidenceItem'", "status = 'active'"]
    params: list = []

    if source:
        conditions.append("provenance_source_system = %s")
        params.append(source)
    if direction:
        conditions.append("body->>'direction' = %s")
        params.append(direction)
    if strength:
        conditions.append("body->>'strength' = %s")
        params.append(strength)
    if search:
        conditions.append("body->>'claim' ILIKE %s")
        params.append(f"%{search}%")

    where = " AND ".join(conditions)
    params.extend([limit, offset])

    with get_connection() as conn:
        rows = conn.execute(
            f"""SELECT id, body, confidence, provenance_source_system, created_at
                FROM erik_core.objects
                WHERE {where}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s""",
            params,
        ).fetchall()

        count_row = conn.execute(
            f"SELECT COUNT(*) FROM erik_core.objects WHERE {where}",
            params[:-2],  # exclude limit/offset
        ).fetchone()

    items = []
    for row in rows:
        obj_id, body, confidence, prov, created = row
        if isinstance(body, str):
            body = json.loads(body)
        items.append({
            "id": obj_id,
            "claim": body.get("claim", ""),
            "direction": body.get("direction"),
            "strength": body.get("strength"),
            "source": prov,
            "confidence": confidence,
            "protocol_layer": body.get("protocol_layer"),
            "source_refs": body.get("source_refs", []),
            "created_at": str(created),
        })

    return {
        "items": items,
        "total": count_row[0] if count_row else 0,
        "limit": limit,
        "offset": offset,
    }


@router.get("/evidence/{item_id:path}")
def get_evidence_item(item_id: str):
    """Return a single evidence item by ID."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, type, body, confidence, provenance_source_system, created_at FROM erik_core.objects WHERE id = %s",
            (item_id,),
        ).fetchone()

    if not row:
        return {"error": "not found"}

    obj_id, obj_type, body, confidence, prov, created = row
    if isinstance(body, str):
        body = json.loads(body)

    return {
        "id": obj_id,
        "type": obj_type,
        "body": body,
        "confidence": confidence,
        "source": prov,
        "created_at": str(created),
    }
