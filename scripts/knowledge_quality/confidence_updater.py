"""Periodic confidence uplift for relationships backed by multiple evidence items.

Relationships supported by multiple evidence items should have higher
confidence than the 0.3 default.  This module scans relationships with
low confidence, counts their backing evidence, and upgrades confidence
based on evidence count and strength distribution.

Usage:
    from knowledge_quality.confidence_updater import update_relationship_confidences
    stats = update_relationship_confidences()
    # stats = {"relationships_scanned": N, "relationships_updated": M}
"""
from __future__ import annotations

import math
from typing import Any

from db.pool import get_connection

# Strength → numeric score mapping
_STRENGTH_SCORES: dict[str, float] = {
    "strong": 0.9,
    "moderate": 0.7,
    "emerging": 0.5,
    "preclinical": 0.4,
    "unknown": 0.3,
}


def compute_confidence(evidence_count: int, strength_counts: dict[str, int]) -> float:
    """Compute relationship confidence from evidence count and strength distribution.

    Args:
        evidence_count: Total number of evidence items backing the relationship.
        strength_counts: Mapping of evidence_strength label to count,
            e.g. ``{"strong": 2, "moderate": 1}``.

    Returns:
        Confidence score in [0.3, 0.95].
    """
    if evidence_count <= 0:
        return 0.3

    # Weighted average of strength scores
    total_weight = 0.0
    weighted_sum = 0.0
    for label, count in strength_counts.items():
        score = _STRENGTH_SCORES.get(label, 0.3)
        weighted_sum += score * count
        total_weight += count

    base = weighted_sum / total_weight if total_weight > 0 else 0.3

    # Logarithmic boost for accumulation
    boost = min(0.3, 0.1 * math.log2(evidence_count))

    return min(0.95, base + boost)


def update_relationship_confidences(dry_run: bool = False) -> dict[str, Any]:
    """Scan relationships and upgrade confidence based on evidence strength.

    Scans ``erik_core.relationships`` WHERE ``confidence < 0.9`` (LIMIT 500).
    For each relationship, reads its ``sources`` array (list of evidence IDs),
    queries ``erik_core.objects`` to count evidence items and group by
    ``body->>'evidence_strength'``, computes new confidence, and UPDATEs if
    the new value exceeds the current one.

    Args:
        dry_run: If True, compute but do not write updates.

    Returns:
        Dict with keys ``relationships_scanned`` and ``relationships_updated``.
    """
    scanned = 0
    updated = 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Fetch relationships with room for improvement
            cur.execute(
                """
                SELECT id, confidence, sources
                FROM erik_core.relationships
                WHERE confidence < 0.9
                ORDER BY confidence ASC
                LIMIT 500
                """
            )
            rows = cur.fetchall()

        for row in rows:
            rel_id, current_conf, sources = row
            scanned += 1

            # sources is a JSONB array of evidence IDs; may be None or empty
            if not sources:
                continue

            # Count evidence items grouped by evidence_strength
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COALESCE(body->>'evidence_strength', 'unknown') AS strength,
                           COUNT(*) AS cnt
                    FROM erik_core.objects
                    WHERE id = ANY(%s)
                    GROUP BY strength
                    """,
                    (list(sources),),
                )
                strength_rows = cur.fetchall()

            if not strength_rows:
                continue

            strength_counts: dict[str, int] = {}
            evidence_count = 0
            for strength_label, cnt in strength_rows:
                strength_counts[strength_label] = cnt
                evidence_count += cnt

            new_conf = compute_confidence(evidence_count, strength_counts)

            # Only upgrade — never downgrade
            if new_conf <= current_conf:
                continue

            if not dry_run:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE erik_core.relationships
                        SET confidence = GREATEST(confidence, %s)
                        WHERE id = %s
                        """,
                        (new_conf, rel_id),
                    )
            updated += 1

        if not dry_run:
            conn.commit()

    return {"relationships_scanned": scanned, "relationships_updated": updated}
