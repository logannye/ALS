"""GalenSCMConnector — queries Galen's causal knowledge graph (SCM layer).

Walks L2/L3 causal edges in galen_kg using recursive CTEs to surface
downstream/upstream causal chains relevant to ALS gene targets, and
measures the causal density of named biological pathways.

Connection pattern mirrors galen_kg.py: open per query, close in finally.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class CausalEdge:
    """A single directed causal edge from Galen's SCM."""
    source: str
    target: str
    relationship_type: str
    pch_layer: int
    confidence: float


@dataclass
class PathwayStrength:
    """Causal density metrics for a named biological pathway.

    confidence = l3_edges / (l2_edges + l3_edges), or 0.0 if no edges.
    """
    pathway: str
    l2_edges: int
    l3_edges: int
    total_entities: int
    confidence: float


class GalenSCMConnector:
    """Read-only queries against Galen's causal knowledge graph (galen_kg).

    Uses recursive CTEs to walk L2+ / L3 causal edges downstream or
    upstream from a seed gene, and counts pathway-level causal density.
    """

    def __init__(
        self,
        database: str = "galen_kg",
        enabled: bool = True,
        min_pch_layer: int = 2,
        max_depth: int = 3,
    ) -> None:
        self._database = database
        self._enabled = enabled
        self._min_pch_layer = min_pch_layer
        self._max_depth = max_depth
        self._user = os.environ.get("USER", "logannye")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self):
        """Open a raw psycopg connection with appropriate session settings."""
        import psycopg
        return psycopg.connect(
            f"dbname={self._database} user={self._user}",
            connect_timeout=10,
            options="-c statement_timeout=30000 -c work_mem=16MB",
        )

    def _zeroed_pathway(self, pathway_name: str) -> PathwayStrength:
        return PathwayStrength(
            pathway=pathway_name,
            l2_edges=0,
            l3_edges=0,
            total_entities=0,
            confidence=0.0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query_causal_downstream(
        self,
        target_gene: str,
        max_depth: int | None = None,
    ) -> list[CausalEdge]:
        """Walk L2+/L3 edges downstream from *target_gene*.

        Returns up to 100 CausalEdge objects.
        Returns [] if disabled or on any error.
        """
        if not self._enabled:
            return []

        depth = max_depth if max_depth is not None else self._max_depth
        conn = None
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH RECURSIVE chain AS (
                        SELECT r.target_id, e2.name AS target_name,
                               r.relationship_type, r.pch_layer, r.confidence, 1 AS depth
                        FROM entities e1
                        JOIN relationships r ON r.source_id = e1.id
                        JOIN entities e2 ON r.target_id = e2.id
                        WHERE e1.name = %s AND r.pch_layer >= %s
                        UNION ALL
                        SELECT r.target_id, e2.name,
                               r.relationship_type, r.pch_layer, r.confidence, c.depth + 1
                        FROM chain c
                        JOIN relationships r ON r.source_id = c.target_id
                        JOIN entities e2 ON r.target_id = e2.id
                        WHERE c.depth < %s AND r.pch_layer >= %s
                    )
                    SELECT %s AS source, target_name, relationship_type,
                           pch_layer, COALESCE(confidence, 0.5)
                    FROM chain LIMIT 100
                    """,
                    (target_gene, self._min_pch_layer, depth, self._min_pch_layer, target_gene),
                )
                rows = cur.fetchall()
        except Exception:
            return []
        finally:
            if conn is not None:
                conn.close()

        return [
            CausalEdge(
                source=row[0],
                target=row[1],
                relationship_type=row[2],
                pch_layer=int(row[3]),
                confidence=float(row[4]),
            )
            for row in rows
        ]

    def query_causal_upstream(
        self,
        effect: str,
        max_depth: int | None = None,
    ) -> list[CausalEdge]:
        """Walk L2+/L3 edges upstream to find causes of *effect*.

        Returns up to 100 CausalEdge objects.
        Returns [] if disabled or on any error.
        """
        if not self._enabled:
            return []

        depth = max_depth if max_depth is not None else self._max_depth
        conn = None
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH RECURSIVE chain AS (
                        SELECT r.source_id, e1.name AS source_name,
                               r.relationship_type, r.pch_layer, r.confidence, 1 AS depth
                        FROM entities e2
                        JOIN relationships r ON r.target_id = e2.id
                        JOIN entities e1 ON r.source_id = e1.id
                        WHERE e2.name = %s AND r.pch_layer >= %s
                        UNION ALL
                        SELECT r.source_id, e1.name,
                               r.relationship_type, r.pch_layer, r.confidence, c.depth + 1
                        FROM chain c
                        JOIN relationships r ON r.target_id = c.source_id
                        JOIN entities e1 ON r.source_id = e1.id
                        WHERE c.depth < %s AND r.pch_layer >= %s
                    )
                    SELECT source_name, %s AS target, relationship_type,
                           pch_layer, COALESCE(confidence, 0.5)
                    FROM chain LIMIT 100
                    """,
                    (effect, self._min_pch_layer, depth, self._min_pch_layer, effect),
                )
                rows = cur.fetchall()
        except Exception:
            return []
        finally:
            if conn is not None:
                conn.close()

        return [
            CausalEdge(
                source=row[0],
                target=row[1],
                relationship_type=row[2],
                pch_layer=int(row[3]),
                confidence=float(row[4]),
            )
            for row in rows
        ]

    def query_pathway_strength(self, pathway_name: str) -> PathwayStrength:
        """Count L2/L3 edges where either endpoint name ILIKE %pathway_name%.

        Returns PathwayStrength.
        Returns zeroed PathwayStrength if disabled or on any error.
        """
        if not self._enabled:
            return self._zeroed_pathway(pathway_name)

        pattern = f"%{pathway_name}%"
        conn = None
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        SUM(CASE WHEN r.pch_layer = 2 THEN 1 ELSE 0 END) AS l2_edges,
                        SUM(CASE WHEN r.pch_layer >= 3 THEN 1 ELSE 0 END) AS l3_edges,
                        COUNT(DISTINCT e1.id) + COUNT(DISTINCT e2.id) AS total_entities
                    FROM relationships r
                    JOIN entities e1 ON r.source_id = e1.id
                    JOIN entities e2 ON r.target_id = e2.id
                    WHERE r.pch_layer >= %s
                      AND (e1.name ILIKE %s OR e2.name ILIKE %s)
                    """,
                    (self._min_pch_layer, pattern, pattern),
                )
                row = cur.fetchone()
        except Exception:
            return self._zeroed_pathway(pathway_name)
        finally:
            if conn is not None:
                conn.close()

        if row is None:
            return self._zeroed_pathway(pathway_name)

        l2 = int(row[0] or 0)
        l3 = int(row[1] or 0)
        total = int(row[2] or 0)
        total_edges = l2 + l3
        confidence = l3 / total_edges if total_edges > 0 else 0.0
        return PathwayStrength(
            pathway=pathway_name,
            l2_edges=l2,
            l3_edges=l3,
            total_entities=total,
            confidence=confidence,
        )
