"""EvidenceStore — PostgreSQL CRUD for EvidenceItem and Intervention objects.

Persists canonical objects into the ``erik_core.objects`` table using
ON CONFLICT upsert semantics.  All top-level domain fields are merged
into the ``body`` JSONB column so that they survive a full roundtrip.
"""
from __future__ import annotations

import json
from typing import Optional

from ontology.evidence import EvidenceItem
from ontology.intervention import Intervention
from db.pool import get_connection


class EvidenceStore:
    """Thin persistence layer over ``erik_core.objects`` for evidence objects."""

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def upsert_evidence_item(self, item: EvidenceItem) -> None:
        """Upsert an EvidenceItem into erik_core.objects."""
        raw = item.model_dump(mode="json")
        body: dict = raw.get("body", {})

        # Merge top-level evidence fields into body so they round-trip
        for key in ("claim", "direction", "strength", "source_refs", "supersedes_ref", "notes"):
            body[key] = raw.get(key)

        self._upsert_object(
            obj_id=raw["id"],
            obj_type=raw["type"],
            status=raw["status"],
            body=body,
            provenance_source_system=raw.get("provenance", {}).get("source_system"),
            confidence=raw.get("uncertainty", {}).get("confidence"),
        )

    def upsert_intervention(self, intervention: Intervention) -> None:
        """Upsert an Intervention into erik_core.objects."""
        raw = intervention.model_dump(mode="json")
        body: dict = raw.get("body", {})

        # Merge top-level intervention fields into body so they round-trip
        for key in (
            "name",
            "intervention_class",
            "targets",
            "protocol_layer",
            "route",
            "intended_effects",
            "known_risks",
        ):
            body[key] = raw.get(key)

        self._upsert_object(
            obj_id=raw["id"],
            obj_type=raw["type"],
            status=raw["status"],
            body=body,
            provenance_source_system=raw.get("provenance", {}).get("source_system"),
            confidence=raw.get("uncertainty", {}).get("confidence"),
        )

    def _upsert_object(
        self,
        *,
        obj_id: str,
        obj_type: str,
        status: str,
        body: dict,
        provenance_source_system: Optional[str],
        confidence: Optional[float],
    ) -> None:
        """Low-level INSERT … ON CONFLICT DO UPDATE into erik_core.objects."""
        sql = """
            INSERT INTO erik_core.objects (
                id, type, status, body, provenance_source_system, confidence, updated_at
            )
            VALUES (%s, %s, %s, %s::jsonb, %s, %s, NOW())
            ON CONFLICT (id) DO UPDATE
                SET type                     = EXCLUDED.type,
                    status                   = EXCLUDED.status,
                    body                     = EXCLUDED.body,
                    provenance_source_system = EXCLUDED.provenance_source_system,
                    confidence               = EXCLUDED.confidence,
                    updated_at               = NOW()
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        obj_id,
                        obj_type,
                        str(status),
                        json.dumps(body),
                        str(provenance_source_system) if provenance_source_system else None,
                        confidence,
                    ),
                )
            conn.commit()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_evidence_item(self, item_id: str) -> Optional[dict]:
        """Return an EvidenceItem dict or None if not found."""
        sql = """
            SELECT id, type, status, body
            FROM erik_core.objects
            WHERE id = %s AND type = 'EvidenceItem'
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (item_id,))
                row = cur.fetchone()
        if row is None:
            return None
        obj_id, obj_type, status, body = row
        return {
            "id": obj_id,
            "type": obj_type,
            "status": status,
            "body": body,
            "claim": body.get("claim"),
            "direction": body.get("direction"),
            "strength": body.get("strength"),
        }

    def get_intervention(self, int_id: str) -> Optional[dict]:
        """Return an Intervention dict or None if not found."""
        sql = """
            SELECT id, type, status, body
            FROM erik_core.objects
            WHERE id = %s AND type = 'Intervention'
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (int_id,))
                row = cur.fetchone()
        if row is None:
            return None
        obj_id, obj_type, status, body = row
        return {
            "id": obj_id,
            "type": obj_type,
            "status": status,
            "body": body,
            "name": body.get("name"),
        }

    def query_by_protocol_layer(self, layer: str) -> list[dict]:
        """Return active EvidenceItems whose body.protocol_layer matches layer."""
        sql = """
            SELECT id, type, status, body
            FROM erik_core.objects
            WHERE type = 'EvidenceItem'
              AND status = 'active'
              AND body->>'protocol_layer' = %s
        """
        return self._run_query(sql, (layer,))

    def query_by_mechanism_target(self, target: str) -> list[dict]:
        """Return active EvidenceItems whose body.mechanism_target matches target."""
        sql = """
            SELECT id, type, status, body
            FROM erik_core.objects
            WHERE type = 'EvidenceItem'
              AND status = 'active'
              AND body->>'mechanism_target' = %s
        """
        return self._run_query(sql, (target,))

    def _run_query(self, sql: str, params: tuple) -> list[dict]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        results = []
        for row in rows:
            obj_id, obj_type, status, body = row
            results.append(
                {
                    "id": obj_id,
                    "type": obj_type,
                    "status": status,
                    "body": body,
                    "claim": body.get("claim"),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    def count_by_type(self, obj_type: str) -> int:
        """Return the count of active objects with the given type."""
        sql = """
            SELECT COUNT(*)
            FROM erik_core.objects
            WHERE type = %s AND status = 'active'
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (obj_type,))
                row = cur.fetchone()
        return int(row[0]) if row else 0

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    def delete(self, obj_id: str) -> None:
        """Hard DELETE by id — for test cleanup only."""
        sql = "DELETE FROM erik_core.objects WHERE id = %s"
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (obj_id,))
            conn.commit()
