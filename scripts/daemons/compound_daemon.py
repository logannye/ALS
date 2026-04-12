"""Phase 4 — CompoundDaemon: evaluate drug candidates against TCG hypotheses.

Runs every 1-2 hours. Takes supported hypotheses, identifies druggable nodes,
queries compound databases, and scores candidates for the treatment protocol.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Optional

from config.loader import ConfigLoader
from db.pool import get_connection
from llm.claude_client import ClaudeClient
from tcg.graph import TCGraph
from tcg.models import TCGHypothesis


class CompoundDaemon:
    """Phase 4: Drug candidate evaluation against TCG hypotheses."""

    def __init__(self, claude_api_key: str) -> None:
        cfg = ConfigLoader()
        self._interval_s = cfg.get("compound_interval_s", 3600)
        self._min_confidence = cfg.get("compound_min_hypothesis_confidence", 0.6)
        self._graph = TCGraph()
        self._claude = ClaudeClient(
            api_key=claude_api_key,
            evaluation_model=cfg.get("claude_evaluation_model", "claude-sonnet-4-6"),
            monthly_budget_usd=cfg.get("claude_monthly_budget_usd", 100.0),
        )
        self._stop = threading.Event()

    def _get_druggable_nodes_for_hypothesis(self, hyp: TCGHypothesis) -> list[dict]:
        """Find druggable nodes along a hypothesis's causal path."""
        druggable = []
        for edge_id in hyp.supporting_path:
            edge = self._graph.get_edge(edge_id)
            if not edge:
                continue
            for node_id in (edge.source_id, edge.target_id):
                node = self._graph.get_node(node_id)
                if node and node.druggability_score > 0.3:
                    druggable.append({
                        "node_id": node.id,
                        "name": node.name,
                        "druggability": node.druggability_score,
                        "cluster": node.pathway_cluster,
                    })
        return druggable

    def _query_existing_compounds(self, target_name: str) -> list[dict]:
        """Query evidence store for known compounds targeting this entity."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, body->>'name' as name, body->>'intervention_class' as cls,
                           confidence
                    FROM erik_core.objects
                    WHERE type = 'Intervention' AND status = 'active'
                    AND (body->>'targets')::text ILIKE %s
                    LIMIT 10
                """, (f"%{target_name}%",))
                rows = cur.fetchall()
        return [{"id": r[0], "name": r[1], "class": r[2], "confidence": r[3]} for r in rows]

    def evaluate_once(self) -> dict:
        """Run one compound evaluation cycle."""
        # Get supported hypotheses above confidence threshold
        supported = self._graph.get_hypotheses_by_status("supported")
        actionable = [h for h in supported if h.confidence >= self._min_confidence]

        if not actionable:
            return {"action": "no_actionable_hypotheses"}

        hyp = actionable[0]
        druggable = self._get_druggable_nodes_for_hypothesis(hyp)
        if not druggable:
            return {"action": "no_druggable_nodes", "hypothesis": hyp.id}

        # Collect existing compounds for each druggable target
        target_compounds: dict[str, list[dict]] = {}
        for node in druggable:
            compounds = self._query_existing_compounds(node["name"])
            if compounds:
                target_compounds[node["name"]] = compounds

        # Build edge descriptions for Claude
        edge_descriptions = []
        for eid in hyp.supporting_path:
            edge = self._graph.get_edge(eid)
            if edge:
                src = self._graph.get_node(edge.source_id)
                tgt = self._graph.get_node(edge.target_id)
                edge_descriptions.append(
                    f"{src.name if src else edge.source_id} --[{edge.edge_type}]--> "
                    f"{tgt.name if tgt else edge.target_id}"
                )

        # Get current protocol summary
        protocol_summary = self._get_protocol_summary()

        # Ask Claude to evaluate
        compound_text = ""
        for target, compounds in target_compounds.items():
            compound_text += f"\nTarget: {target}\n"
            for c in compounds:
                compound_text += f"  - {c['name']} ({c['class']}, confidence: {c['confidence']})\n"

        result = self._claude.evaluate_compound(
            compound=compound_text or "No existing compounds found — novel design needed",
            target_edges=edge_descriptions,
            current_protocol=protocol_summary,
        )

        if result and not result.get("budget_exceeded"):
            # Log the evaluation
            self._graph.log_activity(
                phase="compound", event_type="compound_evaluated",
                summary=f"Evaluated compounds for hypothesis '{hyp.id}': "
                        f"{result.get('recommendation', 'pending')}",
                tcg_hypothesis_id=hyp.id,
                details=result,
            )

            # If highly recommended, promote hypothesis to actionable
            score = result.get("suitability_score", 0)
            if score >= 0.7 and hyp.status != "actionable":
                hyp.status = "actionable"
                self._graph.upsert_hypothesis(hyp)

        return {
            "hypothesis": hyp.id,
            "druggable_nodes": len(druggable),
            "compounds_found": sum(len(v) for v in target_compounds.values()),
            "evaluation": result,
        }

    def _get_protocol_summary(self) -> str:
        """Get current protocol as a text summary."""
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT body FROM erik_core.objects
                        WHERE type = 'Protocol' AND status = 'active'
                        ORDER BY updated_at DESC LIMIT 1
                    """)
                    row = cur.fetchone()
            if row and isinstance(row[0], dict):
                layers = row[0].get("layers", [])
                return "\n".join(
                    f"Layer {l.get('name', '?')}: "
                    + ", ".join(i.get("name", "?") for i in l.get("interventions", []))
                    for l in layers
                )
        except Exception:
            pass
        return "No protocol available"

    def run(self) -> None:
        """Run the daemon loop until stopped."""
        print("[COMPOUND] Daemon started")
        while not self._stop.is_set():
            try:
                cfg = ConfigLoader()
                if not cfg.get("compound_enabled", True):
                    self._stop.wait(60)
                    continue

                self._interval_s = cfg.get("compound_interval_s", 3600)
                result = self.evaluate_once()
                print(f"[COMPOUND] {result}")
            except Exception as e:
                print(f"[COMPOUND] Error: {e}")

            self._stop.wait(self._interval_s)

    def stop(self) -> None:
        self._stop.set()
