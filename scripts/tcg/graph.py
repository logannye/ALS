# scripts/tcg/graph.py
"""TCG read/write operations — the interface between daemons and the Therapeutic Causal Graph."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from db.pool import get_connection
from tcg.models import TCGNode, TCGEdge, TCGHypothesis, AcquisitionItem


class TCGraph:
    """Read/write interface for the Therapeutic Causal Graph."""

    # ── Nodes ──────────────────────────────────────────────

    def upsert_node(self, node: TCGNode) -> None:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO erik_core.tcg_nodes
                        (id, entity_type, name, description, pathway_cluster,
                         druggability_score, metadata, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        entity_type = EXCLUDED.entity_type,
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        pathway_cluster = EXCLUDED.pathway_cluster,
                        druggability_score = EXCLUDED.druggability_score,
                        metadata = EXCLUDED.metadata,
                        updated_at = EXCLUDED.updated_at
                """, (
                    node.id, node.entity_type, node.name, node.description,
                    node.pathway_cluster, node.druggability_score,
                    __import__("json").dumps(node.metadata),
                    node.created_at, node.updated_at,
                ))
            conn.commit()

    def get_node(self, node_id: str) -> Optional[TCGNode]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, entity_type, name, description, pathway_cluster,
                           druggability_score, metadata, created_at, updated_at
                    FROM erik_core.tcg_nodes WHERE id = %s
                """, (node_id,))
                row = cur.fetchone()
        if row is None:
            return None
        return TCGNode(
            id=row[0], entity_type=row[1], name=row[2], description=row[3],
            pathway_cluster=row[4], druggability_score=row[5] or 0.0,
            metadata=row[6] if isinstance(row[6], dict) else {},
            created_at=row[7], updated_at=row[8],
        )

    def list_nodes(self, pathway_cluster: Optional[str] = None) -> list[TCGNode]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                if pathway_cluster:
                    cur.execute("""
                        SELECT id, entity_type, name, description, pathway_cluster,
                               druggability_score, metadata, created_at, updated_at
                        FROM erik_core.tcg_nodes WHERE pathway_cluster = %s
                    """, (pathway_cluster,))
                else:
                    cur.execute("""
                        SELECT id, entity_type, name, description, pathway_cluster,
                               druggability_score, metadata, created_at, updated_at
                        FROM erik_core.tcg_nodes
                    """)
                rows = cur.fetchall()
        return [
            TCGNode(
                id=r[0], entity_type=r[1], name=r[2], description=r[3],
                pathway_cluster=r[4], druggability_score=r[5] or 0.0,
                metadata=r[6] if isinstance(r[6], dict) else {},
                created_at=r[7], updated_at=r[8],
            )
            for r in rows
        ]

    # ── Edges ──────────────────────────────────────────────

    def upsert_edge(self, edge: TCGEdge) -> None:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO erik_core.tcg_edges
                        (id, source_id, target_id, edge_type, confidence,
                         evidence_ids, contradiction_ids, open_questions,
                         intervention_potential, last_reasoned_at, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        edge_type = EXCLUDED.edge_type,
                        confidence = EXCLUDED.confidence,
                        evidence_ids = EXCLUDED.evidence_ids,
                        contradiction_ids = EXCLUDED.contradiction_ids,
                        open_questions = EXCLUDED.open_questions,
                        intervention_potential = EXCLUDED.intervention_potential,
                        last_reasoned_at = EXCLUDED.last_reasoned_at,
                        updated_at = EXCLUDED.updated_at
                """, (
                    edge.id, edge.source_id, edge.target_id, edge.edge_type,
                    edge.confidence, edge.evidence_ids, edge.contradiction_ids,
                    edge.open_questions,
                    __import__("json").dumps(edge.intervention_potential),
                    edge.last_reasoned_at, edge.created_at, edge.updated_at,
                ))
            conn.commit()

    def get_edge(self, edge_id: str) -> Optional[TCGEdge]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, source_id, target_id, edge_type, confidence,
                           evidence_ids, contradiction_ids, open_questions,
                           intervention_potential, last_reasoned_at, created_at, updated_at
                    FROM erik_core.tcg_edges WHERE id = %s
                """, (edge_id,))
                row = cur.fetchone()
        if row is None:
            return None
        return TCGEdge(
            id=row[0], source_id=row[1], target_id=row[2], edge_type=row[3],
            confidence=row[4], evidence_ids=list(row[5] or []),
            contradiction_ids=list(row[6] or []), open_questions=list(row[7] or []),
            intervention_potential=row[8] if isinstance(row[8], dict) else {},
            last_reasoned_at=row[9], created_at=row[10], updated_at=row[11],
        )

    def get_edges_from(self, node_id: str) -> list[TCGEdge]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, source_id, target_id, edge_type, confidence,
                           evidence_ids, contradiction_ids, open_questions,
                           intervention_potential, last_reasoned_at, created_at, updated_at
                    FROM erik_core.tcg_edges WHERE source_id = %s
                """, (node_id,))
                rows = cur.fetchall()
        return [self._edge_from_row(r) for r in rows]

    def get_weakest_edges(self, limit: int = 10) -> list[TCGEdge]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, source_id, target_id, edge_type, confidence,
                           evidence_ids, contradiction_ids, open_questions,
                           intervention_potential, last_reasoned_at, created_at, updated_at
                    FROM erik_core.tcg_edges
                    ORDER BY confidence ASC
                    LIMIT %s
                """, (limit,))
                rows = cur.fetchall()
        return [self._edge_from_row(r) for r in rows]

    def update_edge_confidence(
        self, edge_id: str, confidence: float, evidence_id: Optional[str] = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        with get_connection() as conn:
            with conn.cursor() as cur:
                if evidence_id:
                    cur.execute("""
                        UPDATE erik_core.tcg_edges
                        SET confidence = %s,
                            evidence_ids = array_append(evidence_ids, %s),
                            updated_at = %s
                        WHERE id = %s
                    """, (confidence, evidence_id, now, edge_id))
                else:
                    cur.execute("""
                        UPDATE erik_core.tcg_edges
                        SET confidence = %s, updated_at = %s
                        WHERE id = %s
                    """, (confidence, now, edge_id))
            conn.commit()

    def bayesian_update(
        self, edge_id: str, evidence_strength: float, evidence_id: str,
    ) -> None:
        """Bayesian confidence update with diminishing returns."""
        edge = self.get_edge(edge_id)
        if edge is None:
            return
        prior_strength = max(1.0, len(edge.evidence_ids))
        posterior = (edge.confidence * prior_strength + evidence_strength) / (prior_strength + 1)
        posterior = max(0.0, min(1.0, posterior))
        self.update_edge_confidence(edge_id, posterior, evidence_id)

    # ── Hypotheses ─────────────────────────────────────────

    def upsert_hypothesis(self, hyp: TCGHypothesis) -> None:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO erik_core.tcg_hypotheses
                        (id, hypothesis, supporting_path, confidence, status,
                         generated_by, evidence_for, evidence_against,
                         open_questions, therapeutic_relevance, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        hypothesis = EXCLUDED.hypothesis,
                        supporting_path = EXCLUDED.supporting_path,
                        confidence = EXCLUDED.confidence,
                        status = EXCLUDED.status,
                        evidence_for = EXCLUDED.evidence_for,
                        evidence_against = EXCLUDED.evidence_against,
                        open_questions = EXCLUDED.open_questions,
                        therapeutic_relevance = EXCLUDED.therapeutic_relevance,
                        updated_at = EXCLUDED.updated_at
                """, (
                    hyp.id, hyp.hypothesis, hyp.supporting_path, hyp.confidence,
                    hyp.status, hyp.generated_by, hyp.evidence_for,
                    hyp.evidence_against, hyp.open_questions,
                    hyp.therapeutic_relevance, hyp.created_at, hyp.updated_at,
                ))
            conn.commit()

    def get_hypotheses_by_status(self, status: str) -> list[TCGHypothesis]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, hypothesis, supporting_path, confidence, status,
                           generated_by, evidence_for, evidence_against,
                           open_questions, therapeutic_relevance, created_at, updated_at
                    FROM erik_core.tcg_hypotheses
                    WHERE status = %s
                    ORDER BY therapeutic_relevance DESC
                """, (status,))
                rows = cur.fetchall()
        return [
            TCGHypothesis(
                id=r[0], hypothesis=r[1], supporting_path=list(r[2] or []),
                confidence=r[3], status=r[4], generated_by=r[5],
                evidence_for=list(r[6] or []), evidence_against=list(r[7] or []),
                open_questions=list(r[8] or []), therapeutic_relevance=r[9],
                created_at=r[10], updated_at=r[11],
            )
            for r in rows
        ]

    # ── Acquisition Queue ──────────────────────────────────

    def push_acquisition(self, item: AcquisitionItem) -> None:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO erik_ops.acquisition_queue
                        (tcg_edge_id, open_question, suggested_sources,
                         exhausted_sources, priority, status, created_by)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    item.tcg_edge_id, item.open_question, item.suggested_sources,
                    item.exhausted_sources, item.priority, item.status, item.created_by,
                ))
            conn.commit()

    def pop_acquisition(self) -> Optional[AcquisitionItem]:
        """Atomically pop the highest-priority pending item."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE erik_ops.acquisition_queue
                    SET status = 'in_progress'
                    WHERE id = (
                        SELECT id FROM erik_ops.acquisition_queue
                        WHERE status = 'pending'
                        ORDER BY priority DESC
                        LIMIT 1
                        FOR UPDATE SKIP LOCKED
                    )
                    RETURNING id, tcg_edge_id, open_question, suggested_sources,
                              exhausted_sources, priority, status, created_by,
                              created_at, answered_at
                """)
                row = cur.fetchone()
            conn.commit()
        if row is None:
            return None
        return AcquisitionItem(
            id=row[0], tcg_edge_id=row[1], open_question=row[2],
            suggested_sources=list(row[3] or []), exhausted_sources=list(row[4] or []),
            priority=row[5], status=row[6], created_by=row[7],
            created_at=row[8], answered_at=row[9],
        )

    def mark_acquisition(self, item_id: int, status: str) -> None:
        now = datetime.now(timezone.utc)
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE erik_ops.acquisition_queue
                    SET status = %s, answered_at = %s
                    WHERE id = %s
                """, (status, now if status in ("answered", "unanswerable") else None, item_id))
            conn.commit()

    # ── Activity Feed ──────────────────────────────────────

    def log_activity(
        self, phase: str, event_type: str, summary: str, *,
        details: dict | None = None, tcg_edge_id: str | None = None,
        tcg_hypothesis_id: str | None = None,
    ) -> None:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO erik_ops.activity_feed
                        (phase, event_type, summary, details, tcg_edge_id, tcg_hypothesis_id)
                    VALUES (%s, %s, %s, %s::jsonb, %s, %s)
                """, (
                    phase, event_type, summary,
                    __import__("json").dumps(details or {}),
                    tcg_edge_id, tcg_hypothesis_id,
                ))
            conn.commit()

    # ── Summary / Metrics ──────────────────────────────────

    def summary(self) -> dict:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM erik_core.tcg_nodes")
                node_count = cur.fetchone()[0]
                cur.execute("SELECT count(*), coalesce(avg(confidence), 0) FROM erik_core.tcg_edges")
                row = cur.fetchone()
                edge_count, mean_confidence = row[0], float(row[1])
                cur.execute("""
                    SELECT pathway_cluster, count(*)
                    FROM erik_core.tcg_nodes
                    WHERE pathway_cluster IS NOT NULL
                    GROUP BY pathway_cluster
                """)
                clusters = {r[0]: r[1] for r in cur.fetchall()}
        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "mean_confidence": round(mean_confidence, 4),
            "clusters": clusters,
        }

    # ── Helpers ────────────────────────────────────────────

    def _edge_from_row(self, r: tuple) -> TCGEdge:
        return TCGEdge(
            id=r[0], source_id=r[1], target_id=r[2], edge_type=r[3],
            confidence=r[4], evidence_ids=list(r[5] or []),
            contradiction_ids=list(r[6] or []), open_questions=list(r[7] or []),
            intervention_potential=r[8] if isinstance(r[8], dict) else {},
            last_reasoned_at=r[9], created_at=r[10], updated_at=r[11],
        )
