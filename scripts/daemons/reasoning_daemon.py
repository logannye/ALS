# scripts/daemons/reasoning_daemon.py
"""Phase 3 — ReasoningDaemon: Claude-powered deep reasoning over the TCG.

Three modes rotated by configurable weights:
A) Edge Deepening — strengthen/refute individual edges (Sonnet)
B) Counterfactual Analysis — trace intervention consequences (Opus)
C) Cross-Pathway Synthesis — discover inter-cluster connections (Opus)
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Optional

from config.loader import ConfigLoader
from db.pool import get_connection
from llm.claude_client import ClaudeClient
from tcg.graph import TCGraph
from tcg.models import TCGEdge, TCGHypothesis


def _select_mode(step: int, weights: list[float]) -> str:
    """Deterministic mode selection from weights using golden-ratio stride."""
    modes = ["edge_deepening", "counterfactual", "cross_pathway"]
    total = sum(weights)
    target = ((step * 61) % 100) / 100.0 * total
    cumulative = 0.0
    for mode, w in zip(modes, weights):
        cumulative += w
        if cumulative >= target:
            return mode
    return modes[-1]


def _filter_cooled_down(
    edges: list[TCGEdge],
    cooldown_s: int = 3600,
) -> list[TCGEdge]:
    """Exclude edges that were reasoned about within the cooldown window.

    If ALL edges are on cooldown, returns the full list to prevent starvation.
    """
    now = datetime.now(timezone.utc)
    available = [
        e for e in edges
        if e.last_reasoned_at is None
        or (now - e.last_reasoned_at).total_seconds() > cooldown_s
    ]
    return available if available else edges


class ReasoningDaemon:
    """Phase 3: Claude-powered deep reasoning over the TCG."""

    def __init__(self, claude_api_key: str) -> None:
        cfg = ConfigLoader()
        self._interval_s = cfg.get("reasoning_interval_s", 900)
        self._mode_weights = cfg.get("reasoning_mode_weights", [0.5, 0.3, 0.2])
        self._max_evidence_per_prompt = cfg.get("reasoning_max_evidence_per_prompt", 30)
        self._graph = TCGraph()
        self._claude = ClaudeClient(
            api_key=claude_api_key,
            reasoning_model=cfg.get("claude_reasoning_model", "claude-opus-4-6"),
            evaluation_model=cfg.get("claude_evaluation_model", "claude-sonnet-4-6"),
            max_opus_per_hour=cfg.get("claude_max_opus_calls_per_hour", 30),
            max_sonnet_per_hour=cfg.get("claude_max_sonnet_calls_per_hour", 60),
            monthly_budget_usd=cfg.get("claude_monthly_budget_usd", 100.0),
        )
        self._step = 0
        self._stop = threading.Event()

    def _get_evidence_text(self, evidence_ids: list[str], limit: int = 30) -> list[str]:
        """Fetch evidence claim text by IDs."""
        if not evidence_ids:
            return []
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, body->>'claim' as claim
                    FROM erik_core.objects
                    WHERE id = ANY(%s) AND status = 'active'
                    LIMIT %s
                """, (evidence_ids[:limit], limit))
                rows = cur.fetchall()
        return [f"[{r[0]}] {r[1]}" for r in rows if r[1]]

    def _run_edge_deepening(self) -> dict:
        """Mode A: Select weakest high-relevance edge, reason about it with Claude Sonnet."""
        edges = self._graph.get_weakest_edges(limit=20)
        if not edges:
            return {"mode": "edge_deepening", "action": "no_weak_edges"}

        # Filter out recently-reasoned edges (cooldown)
        cfg = ConfigLoader()
        cooldown_s = cfg.get("reasoning_edge_cooldown_s", 3600)
        candidates = _filter_cooled_down(edges, cooldown_s=cooldown_s)

        # Pick the edge with highest therapeutic priority from available candidates
        edge = max(candidates, key=lambda e: e.therapeutic_priority())
        supporting = self._get_evidence_text(edge.evidence_ids)
        contradicting = self._get_evidence_text(edge.contradiction_ids)

        source_node = self._graph.get_node(edge.source_id)
        target_node = self._graph.get_node(edge.target_id)
        edge_context = (
            f"{source_node.name if source_node else edge.source_id} "
            f"--[{edge.edge_type}]--> "
            f"{target_node.name if target_node else edge.target_id} "
            f"(current confidence: {edge.confidence})"
        )

        result = self._claude.reason_about_edge(edge_context, supporting, contradicting)
        if not result or result.get("budget_exceeded"):
            return {"mode": "edge_deepening", "action": "api_unavailable"}

        # Update edge based on Claude's analysis
        new_confidence = result.get("confidence_assessment", edge.confidence)
        new_questions = result.get("open_questions", [])
        now = datetime.now(timezone.utc)

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE erik_core.tcg_edges
                    SET confidence = %s, open_questions = %s, last_reasoned_at = %s, updated_at = %s
                    WHERE id = %s
                """, (new_confidence, new_questions, now, now, edge.id))
            conn.commit()

        self._graph.log_activity(
            phase="reasoning", event_type="edge_deepened",
            summary=f"Edge {edge.id}: {edge.confidence:.2f} -> {new_confidence:.2f}",
            tcg_edge_id=edge.id,
        )

        return {"mode": "edge_deepening", "edge": edge.id,
                "confidence_before": edge.confidence, "confidence_after": new_confidence}

    def _run_counterfactual(self) -> dict:
        """Mode B: Counterfactual analysis on a hypothesis with Claude Opus."""
        hypotheses = self._graph.get_hypotheses_by_status("under_investigation")
        if not hypotheses:
            # Promote a proposed hypothesis
            proposed = self._graph.get_hypotheses_by_status("proposed")
            if proposed:
                hyp = proposed[0]
                hyp.status = "under_investigation"
                self._graph.upsert_hypothesis(hyp)
                hypotheses = [hyp]
            else:
                return {"mode": "counterfactual", "action": "no_hypotheses"}

        hyp = hypotheses[0]
        # Build causal path context
        path_descriptions = []
        for edge_id in hyp.supporting_path:
            edge = self._graph.get_edge(edge_id)
            if edge:
                src = self._graph.get_node(edge.source_id)
                tgt = self._graph.get_node(edge.target_id)
                path_descriptions.append(
                    f"{src.name if src else edge.source_id} --[{edge.edge_type}]--> "
                    f"{tgt.name if tgt else edge.target_id} (conf: {edge.confidence:.2f})"
                )

        result = self._claude.counterfactual_analysis(
            hypothesis=hyp.hypothesis,
            causal_path=path_descriptions,
            tcg_context=f"Hypothesis: {hyp.hypothesis}\nTherapeutic relevance: {hyp.therapeutic_relevance}",
        )
        if not result or result.get("budget_exceeded"):
            return {"mode": "counterfactual", "action": "api_unavailable"}

        # Update hypothesis
        new_conf = result.get("confidence", hyp.confidence)
        now = datetime.now(timezone.utc)
        hyp.confidence = new_conf
        hyp.updated_at = now

        # Create new edges from counterfactual analysis
        new_edges_data = result.get("new_edges", [])
        new_edges_created = 0
        for ne in new_edges_data:
            if all(k in ne for k in ("source", "target", "edge_type")):
                edge_id = f"edge:cf_{ne['source'].split(':')[-1]}_{ne['target'].split(':')[-1]}"
                new_edge = TCGEdge(
                    id=edge_id, source_id=ne["source"], target_id=ne["target"],
                    edge_type=ne["edge_type"], confidence=ne.get("confidence", 0.2),
                    open_questions=[ne.get("rationale", "Discovered via counterfactual analysis")],
                )
                try:
                    self._graph.upsert_edge(new_edge)
                    new_edges_created += 1
                except Exception:
                    pass  # Node may not exist

        if new_conf >= 0.7:
            hyp.status = "supported"
        elif new_conf < 0.2:
            hyp.status = "refuted"
        self._graph.upsert_hypothesis(hyp)

        self._graph.log_activity(
            phase="reasoning", event_type="counterfactual_analyzed",
            summary=f"Hypothesis '{hyp.id}': conf {new_conf:.2f}, {new_edges_created} new edges",
            tcg_hypothesis_id=hyp.id,
        )

        return {"mode": "counterfactual", "hypothesis": hyp.id,
                "new_edges": new_edges_created, "confidence": new_conf}

    def _run_cross_pathway(self) -> dict:
        """Mode C: Cross-pathway synthesis between two clusters with Claude Opus."""
        # Find the two clusters with fewest inter-cluster edges
        clusters = list(self._graph.summary().get("clusters", {}).keys())
        if len(clusters) < 2:
            return {"mode": "cross_pathway", "action": "insufficient_clusters"}

        # Pick two clusters with least connection
        # Simple heuristic: rotate through pairs using step count
        import itertools
        pairs = list(itertools.combinations(sorted(clusters), 2))
        if not pairs:
            return {"mode": "cross_pathway", "action": "no_pairs"}
        pair = pairs[self._step % len(pairs)]
        cluster_a, cluster_b = pair

        # Get evidence summaries from each cluster
        nodes_a = self._graph.list_nodes(pathway_cluster=cluster_a)
        nodes_b = self._graph.list_nodes(pathway_cluster=cluster_b)
        evidence_a = [f"{n.name}: {n.description or n.entity_type}" for n in nodes_a[:15]]
        evidence_b = [f"{n.name}: {n.description or n.entity_type}" for n in nodes_b[:15]]

        result = self._claude.cross_pathway_synthesis(cluster_a, evidence_a, cluster_b, evidence_b)
        if not result or result.get("budget_exceeded"):
            return {"mode": "cross_pathway", "action": "api_unavailable"}

        proposed = result.get("proposed_edges", [])
        created = 0
        for pe in proposed:
            if all(k in pe for k in ("source", "target", "edge_type")):
                edge_id = f"edge:xp_{pe['source'].split(':')[-1]}_{pe['target'].split(':')[-1]}"
                new_edge = TCGEdge(
                    id=edge_id, source_id=pe["source"], target_id=pe["target"],
                    edge_type=pe["edge_type"], confidence=pe.get("confidence", 0.2),
                    open_questions=[pe.get("rationale", "Cross-pathway synthesis discovery")],
                )
                try:
                    self._graph.upsert_edge(new_edge)
                    created += 1
                except Exception:
                    pass

        if created > 0:
            self._graph.log_activity(
                phase="reasoning", event_type="cross_pathway_discovery",
                summary=f"Found {created} new edges between {cluster_a} and {cluster_b}",
            )

        return {"mode": "cross_pathway", "clusters": [cluster_a, cluster_b],
                "new_edges": created}

    def reason_once(self) -> dict:
        """Execute one reasoning cycle. Returns stats dict."""
        cfg = ConfigLoader()
        self._mode_weights = cfg.get("reasoning_mode_weights", self._mode_weights)
        mode = _select_mode(self._step, self._mode_weights)
        self._step += 1

        if mode == "edge_deepening":
            return self._run_edge_deepening()
        elif mode == "counterfactual":
            return self._run_counterfactual()
        else:
            return self._run_cross_pathway()

    def run(self) -> None:
        """Run the daemon loop until stopped."""
        print("[REASONING] Daemon started")
        while not self._stop.is_set():
            try:
                cfg = ConfigLoader()
                if not cfg.get("reasoning_enabled", True):
                    self._stop.wait(60)
                    continue

                self._interval_s = cfg.get("reasoning_interval_s", 900)
                result = self.reason_once()
                print(f"[REASONING] {result.get('mode', '?')}: {result}")
            except Exception as e:
                print(f"[REASONING] Error: {e}")

            self._stop.wait(self._interval_s)

    def stop(self) -> None:
        self._stop.set()
