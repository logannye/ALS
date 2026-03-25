"""Append-only audit event logger for the Erik ALS engine.

Writes to erik_ops.audit_events in PostgreSQL.
All writes are immutable — there is no update or delete (except the
test-cleanup helper `delete_test_events`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from db.pool import get_connection


@dataclass
class AuditEvent:
    event_type: str
    actor: str = "erik_system"
    object_id: str | None = None
    object_type: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


class AuditLogger:
    """Writes and queries audit events stored in erik_ops.audit_events."""

    def log(
        self,
        event_type: str,
        object_id: str | None = None,
        object_type: str | None = None,
        actor: str = "erik_system",
        details: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> AuditEvent:
        """Insert an audit event and return the resulting AuditEvent."""
        if details is None:
            details = {}

        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO erik_ops.audit_events
                    (event_type, object_id, object_type, actor, details, trace_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING created_at
                """,
                (event_type, object_id, object_type, actor, json.dumps(details), trace_id),
            )
            row = cur.fetchone()
            conn.commit()

        return AuditEvent(
            event_type=event_type,
            object_id=object_id,
            object_type=object_type,
            actor=actor,
            details=details,
            trace_id=trace_id,
            created_at=row[0],
        )

    def query(
        self,
        object_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Return audit events matching the given filters, newest first."""
        filters: list[str] = []
        params: list[Any] = []

        if object_id is not None:
            filters.append("object_id = %s")
            params.append(object_id)
        if event_type is not None:
            filters.append("event_type = %s")
            params.append(event_type)

        where_clause = ("WHERE " + " AND ".join(filters)) if filters else ""
        params.append(limit)

        sql = f"""
            SELECT event_type, object_id, object_type, actor, details, trace_id, created_at
            FROM erik_ops.audit_events
            {where_clause}
            ORDER BY created_at DESC
            LIMIT %s
        """

        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()

        events: list[AuditEvent] = []
        for row in rows:
            evt_type, obj_id, obj_type, actor, details_raw, tid, created = row
            details = details_raw if isinstance(details_raw, dict) else json.loads(details_raw)
            events.append(
                AuditEvent(
                    event_type=evt_type,
                    object_id=obj_id,
                    object_type=obj_type,
                    actor=actor,
                    details=details,
                    trace_id=tid,
                    created_at=created,
                )
            )
        return events

    def delete_test_events(self, object_id: str) -> int:
        """Delete audit events by object_id. For test cleanup only."""
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM erik_ops.audit_events WHERE object_id = %s",
                (object_id,),
            )
            deleted = cur.rowcount
            conn.commit()
        return deleted
