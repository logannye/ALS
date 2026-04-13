"""Phase 2 — IntegrationDaemon: weave raw evidence into the Therapeutic Causal Graph.

Runs every 2-5 minutes. Reads unintegrated evidence from erik_core.objects,
extracts entities and relationships, updates TCG edge confidence via Bayesian
updates, and creates acquisition queue entries for new weak edges.
"""
from __future__ import annotations

import time
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from config.loader import ConfigLoader
from db.pool import get_connection
from tcg.graph import TCGraph
from tcg.models import AcquisitionItem


# Source mapping: question type -> suggested data sources
QUESTION_TYPE_SOURCES: dict[str, list[str]] = {
    "mechanistic": ["pubmed", "biorxiv", "galen_kg"],
    "binding": ["chembl", "bindingdb", "drugbank"],
    "expression": ["gtex", "hpa", "geo_als"],
    "genetic": ["clinvar", "gnomad", "gwas", "alsod"],
    "clinical": ["clinical_trials", "faers"],
    "pathway": ["reactome", "kegg", "string"],
    "structural": ["alphafold", "uniprot"],
}

_QUESTION_KEYWORDS: dict[str, list[str]] = {
    "binding": ["bind", "affinity", "ic50", "ki ", "kd ", "inhibit"],
    "expression": ["express", "transcri", "mrna", "rna level"],
    "genetic": ["variant", "mutation", "pathogenic", "snp", "polymorphism", "allele"],
    "clinical": ["trial", "clinical", "phase ", "fda", "approved"],
    "pathway": ["pathway", "signal", "cascade", "downstream"],
    "structural": ["structure", "3d", "crystal", "cryo-em", "fold"],
}


def _classify_question_type(question: str) -> str:
    """Classify an open question to determine which data sources to suggest."""
    q_lower = question.lower()
    for qtype, keywords in _QUESTION_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            return qtype
    return "mechanistic"  # default


def _build_node_name_index() -> dict[str, list[tuple[str, float]]]:
    """Build a lookup: lowercase node name -> list of (edge_id, edge_confidence).

    Each node maps to all edges where it appears as source or target.
    Called once per integrate_batch for efficiency.
    """
    index: dict[str, list[tuple[str, float]]] = {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT e.id, e.confidence, sn.name, tn.name
                FROM erik_core.tcg_edges e
                JOIN erik_core.tcg_nodes sn ON e.source_id = sn.id
                JOIN erik_core.tcg_nodes tn ON e.target_id = tn.id
            """)
            for edge_id, conf, src_name, tgt_name in cur.fetchall():
                entry = (edge_id, conf)
                for name in (src_name.lower(), tgt_name.lower()):
                    if name not in index:
                        index[name] = []
                    index[name].append(entry)
    return index


def _match_evidence_to_edges(
    evidence: dict,
    node_index: dict[str, list[tuple[str, float]]],
) -> list[tuple[str, float]]:
    """Match evidence to TCG edges by finding node names in evidence text.

    Returns deduplicated list of (edge_id, evidence_strength) tuples.
    """
    body = evidence.get("body", {})
    claim = body.get("claim", "")
    text = f"{claim} {body.get('notes', '')} {body.get('mechanism', '')}".lower()
    confidence = evidence.get("confidence") or 0.5

    seen_edges: set[str] = set()
    matches: list[tuple[str, float]] = []

    for node_name, edges in node_index.items():
        if node_name in text:
            for edge_id, _edge_conf in edges:
                if edge_id not in seen_edges:
                    seen_edges.add(edge_id)
                    matches.append((edge_id, confidence))

    return matches


class IntegrationDaemon:
    """Phase 2: Evidence -> TCG integration."""

    def __init__(self) -> None:
        cfg = ConfigLoader()
        self._interval_s = cfg.get("integration_interval_s", 180)
        self._batch_size = cfg.get("integration_batch_size", 50)
        self._prior_strength = cfg.get("integration_confidence_prior_strength", 2.0)
        self._graph = TCGraph()
        self._stop = threading.Event()

    def _get_unintegrated_evidence(self) -> list[dict]:
        """Fetch evidence items not yet integrated into TCG."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, type, body, confidence, provenance_source_system
                    FROM erik_core.objects
                    WHERE status = 'active'
                      AND tcg_integrated = FALSE
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (self._batch_size,))
                rows = cur.fetchall()
        return [
            {"id": r[0], "type": r[1], "body": r[2] if isinstance(r[2], dict) else {},
             "confidence": r[3], "source": r[4]}
            for r in rows
        ]

    def _mark_integrated(self, item_ids: list[str]) -> None:
        if not item_ids:
            return
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE erik_core.objects SET tcg_integrated = TRUE
                    WHERE id = ANY(%s)
                """, (item_ids,))
            conn.commit()

    def _extract_edge_matches(self, evidence: dict, node_index: dict[str, list[tuple[str, float]]]) -> list[tuple[str, float]]:
        """Match evidence to TCG edges by entity name overlap.

        Returns list of (edge_id, evidence_strength) tuples.
        """
        return _match_evidence_to_edges(evidence, node_index)

    def integrate_batch(self) -> dict[str, int]:
        """Process one batch of unintegrated evidence. Returns stats."""
        evidence_items = self._get_unintegrated_evidence()
        if not evidence_items:
            return {"items_processed": 0, "edges_updated": 0, "queue_items_created": 0}

        edges_updated = 0
        queue_items = 0
        processed_ids = []

        node_index = _build_node_name_index()
        for ev in evidence_items:
            matches = self._extract_edge_matches(ev, node_index)
            for edge_id, strength in matches:
                self._graph.bayesian_update(edge_id, strength, ev["id"])
                edges_updated += 1

                # Create acquisition items for edges that got updated but are still weak
                edge = self._graph.get_edge(edge_id)
                if edge and edge.confidence < 0.7 and edge.open_questions:
                    for q in edge.open_questions[:1]:  # First open question only
                        qtype = _classify_question_type(q)
                        sources = QUESTION_TYPE_SOURCES.get(qtype, ["pubmed"])
                        self._graph.push_acquisition(AcquisitionItem(
                            tcg_edge_id=edge_id,
                            open_question=q,
                            suggested_sources=sources,
                            priority=edge.therapeutic_priority(),
                            created_by="integration",
                        ))
                        queue_items += 1

            processed_ids.append(ev["id"])

        self._mark_integrated(processed_ids)

        if edges_updated > 0:
            self._graph.log_activity(
                phase="integration", event_type="batch_integrated",
                summary=f"Integrated {len(processed_ids)} evidence items, updated {edges_updated} edges",
            )

        return {
            "items_processed": len(processed_ids),
            "edges_updated": edges_updated,
            "queue_items_created": queue_items,
        }

    def run(self) -> None:
        """Run the daemon loop until stopped."""
        print("[INTEGRATION] Daemon started")
        while not self._stop.is_set():
            try:
                cfg = ConfigLoader()
                if not cfg.get("integration_enabled", True):
                    self._stop.wait(60)
                    continue

                self._interval_s = cfg.get("integration_interval_s", 180)
                stats = self.integrate_batch()
                if stats["items_processed"] > 0:
                    print(f"[INTEGRATION] Batch: {stats['items_processed']} items, "
                          f"{stats['edges_updated']} edges, {stats['queue_items_created']} queue items")
            except Exception as e:
                print(f"[INTEGRATION] Error: {e}")

            self._stop.wait(self._interval_s)

    def stop(self) -> None:
        self._stop.set()
