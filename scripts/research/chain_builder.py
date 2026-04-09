"""Mechanism→mechanism chain builder for the ALS knowledge graph.

The KG has gene→mechanism ``contributes_to`` edges but no
mechanism→mechanism chains.  This module discovers implicit chains:
when two mechanisms share a common upstream gene they are connected
with an inferred ``contributes_to`` edge at PCH Layer 2 (inferred
causal).

Usage:
    from research.chain_builder import build_mechanism_chains
    stats = build_mechanism_chains()
    # stats = {"chains_found": N, "edges_created": M}
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from itertools import combinations
from typing import Any

from db.pool import get_connection


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_mechanism_chains(edges: list[dict]) -> list[dict]:
    """Discover mechanism→mechanism chain links from gene→mechanism edges.

    Groups mechanisms by their source gene.  For every pair of distinct
    mechanisms that share a gene, creates a chain link dict.

    Returns a list of chain-link dicts with keys:
        source_mechanism, target_mechanism, via_gene, confidence, evidence_ids
    """
    # Group: gene -> list of (mechanism, confidence, evidence_id)
    gene_to_mechs: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
    for e in edges:
        gene_to_mechs[e["source_id"]].append(
            (e["target_id"], e["confidence"], e["evidence"])
        )

    chains: list[dict] = []
    for gene, mechs in gene_to_mechs.items():
        # Deduplicate mechanisms for this gene (keep highest confidence per mech)
        best: dict[str, tuple[float, str]] = {}
        for mech_id, conf, evi in mechs:
            if mech_id not in best or conf > best[mech_id][0]:
                best[mech_id] = (conf, evi)

        unique_mechs = list(best.keys())
        if len(unique_mechs) < 2:
            continue

        for mech_a, mech_b in combinations(sorted(unique_mechs), 2):
            # No self-loops (shouldn't happen after dedup, but guard anyway)
            if mech_a == mech_b:
                continue

            conf_a, evi_a = best[mech_a]
            conf_b, evi_b = best[mech_b]

            chains.append({
                "source_mechanism": mech_a,
                "target_mechanism": mech_b,
                "via_gene": gene,
                "confidence": min(conf_a, conf_b),
                "evidence_ids": sorted({evi_a, evi_b}),
            })

    return chains


def _build_chain_relationship(
    source_mechanism: str,
    target_mechanism: str,
    via_gene: str,
    confidence: float,
    evidence_ids: list[str],
) -> dict:
    """Create a relationship dict for a mechanism→mechanism chain edge.

    The relationship is ``contributes_to`` at PCH Layer 2 (inferred causal,
    not observational and not yet counterfactually validated).
    """
    raw = f"{source_mechanism}|contributes_to|{target_mechanism}"
    rel_id = f"rel:{hashlib.sha256(raw.encode()).hexdigest()[:12]}"

    return {
        "id": rel_id,
        "source_id": source_mechanism,
        "target_id": target_mechanism,
        "relationship_type": "contributes_to",
        "confidence": confidence,
        "evidence": f"Inferred chain via {via_gene}: {' + '.join(evidence_ids)}",
        "sources": [{"type": "inferred_chain", "via_gene": via_gene, "evidence_ids": evidence_ids}],
        "pch_layer": 2,
        "evidence_type": "inferred_chain",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_mechanism_chains(dry_run: bool = False) -> dict[str, int]:
    """Query the KG for gene→mechanism edges and create mechanism→mechanism chains.

    Args:
        dry_run: If True, discover chains but do not write to the database.

    Returns:
        ``{"chains_found": N, "edges_created": M}``
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT source_id, target_id, confidence, evidence
                FROM erik_core.relationships
                WHERE relationship_type = 'contributes_to'
                  AND source_id LIKE 'gene:%%'
                  AND target_id LIKE 'mechanism:%%'
            """)
            rows = cur.fetchall()

    # Convert rows to edge dicts
    edges: list[dict] = [
        {
            "source_id": r[0],
            "target_id": r[1],
            "confidence": float(r[2]) if r[2] is not None else 0.5,
            "evidence": r[3] or "",
        }
        for r in rows
    ]

    chains = _find_mechanism_chains(edges)
    stats: dict[str, int] = {"chains_found": len(chains), "edges_created": 0}

    if dry_run or not chains:
        return stats

    # Build relationship dicts and upsert
    with get_connection() as conn:
        with conn.cursor() as cur:
            for link in chains:
                rel = _build_chain_relationship(
                    source_mechanism=link["source_mechanism"],
                    target_mechanism=link["target_mechanism"],
                    via_gene=link["via_gene"],
                    confidence=link["confidence"],
                    evidence_ids=link["evidence_ids"],
                )
                try:
                    cur.execute("""
                        INSERT INTO erik_core.relationships
                            (id, source_id, target_id, relationship_type,
                             confidence, evidence, sources, pch_layer, evidence_type)
                        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            confidence = GREATEST(erik_core.relationships.confidence, EXCLUDED.confidence),
                            sources = EXCLUDED.sources,
                            updated_at = NOW()
                    """, (
                        rel["id"],
                        rel["source_id"],
                        rel["target_id"],
                        rel["relationship_type"],
                        rel["confidence"],
                        rel["evidence"][:500],
                        json.dumps(rel["sources"]),
                        rel["pch_layer"],
                        rel["evidence_type"],
                    ))
                    stats["edges_created"] += 1
                except Exception:
                    continue
        conn.commit()

    return stats
