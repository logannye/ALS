"""Query Expansion Engine — breaks evidence saturation by expanding to KG neighbors.

When a gene x database combination is exhausted (consecutive zero-delta queries),
this module expands to second-order targets derived from the knowledge graph's
entities and relationships — interacting genes, shared pathways, mechanism neighbors.

Transparent to the RL loop: expansion produces different params for the same actions.
"""
from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

import os as _os

import psycopg

if TYPE_CHECKING:
    from research.state import ResearchState


# ---------------------------------------------------------------------------
# Exhaustion detection
# ---------------------------------------------------------------------------

def should_expand(
    target_key: str,
    state: ResearchState,
    threshold: int = 3,
) -> bool:
    """Return True if the target has been exhausted (consecutive zero-delta >= threshold)."""
    return state.target_exhaustion.get(target_key, 0) >= threshold


# ---------------------------------------------------------------------------
# KG neighbor lookup
# ---------------------------------------------------------------------------

def get_gene_neighbors(
    gene: str,
    max_neighbors: int = 10,
    min_confidence: float = 0.4,
) -> list[dict]:
    """Query erik_core.entities + relationships for genes related to the given gene.

    Returns list of dicts: [{"gene": "OPTN", "relationship": "associated_with", "confidence": 0.7}, ...]
    """
    try:
        from db.pool import get_connection
    except Exception:
        return []

    neighbors: list[dict] = []
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT e2.name, r.relationship_type, r.confidence
                    FROM erik_core.entities e1
                    JOIN erik_core.relationships r
                        ON (r.source_id = e1.id OR r.target_id = e1.id)
                    JOIN erik_core.entities e2
                        ON (CASE WHEN r.source_id = e1.id
                                 THEN r.target_id
                                 ELSE r.source_id END = e2.id)
                    WHERE e1.name ILIKE %s
                      AND e1.entity_type IN ('gene', 'protein')
                      AND e2.entity_type IN ('gene', 'protein')
                      AND e2.name NOT ILIKE %s
                      AND r.confidence >= %s
                    ORDER BY r.confidence DESC
                    LIMIT %s
                """, (gene, gene, min_confidence, max_neighbors))
                for row in cur.fetchall():
                    neighbors.append({
                        "gene": row[0],
                        "relationship": row[1],
                        "confidence": float(row[2]) if row[2] is not None else 0.5,
                    })
    except Exception:
        pass

    return neighbors


# ---------------------------------------------------------------------------
# Galen KG neighbor lookup (731K entities, 6.5M relationships)
# ---------------------------------------------------------------------------

def get_gene_neighbors_galen(
    gene: str,
    max_neighbors: int = 10,
    min_confidence: float = 0.4,
) -> list[dict]:
    """Query Galen's cancer KG for gene/protein neighbors of *gene*.

    Returns list of dicts: [{"gene": "BECN1", "relationship": "regulates", "confidence": 0.8}, ...]
    Returns [] if galen_kg is unreachable or on any error.
    """
    try:
        user = _os.environ.get("USER", "logannye")
        conn = psycopg.connect(
            f"dbname=galen_kg user={user}",
            connect_timeout=10,
            options="-c statement_timeout=15000 -c work_mem=16MB",
        )
    except Exception:
        return []

    neighbors: list[dict] = []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT sub.name, sub.rel, sub.conf FROM (
                    (SELECT e2.name, r.relationship_type AS rel,
                            COALESCE(r.confidence, 0.5) AS conf
                     FROM entities e1
                     JOIN relationships r ON r.source_id = e1.id
                     JOIN entities e2 ON r.target_id = e2.id
                     WHERE e1.name = %s AND e2.entity_type IN ('gene', 'protein')
                       AND e2.name != %s
                     ORDER BY r.confidence DESC NULLS LAST LIMIT %s)
                    UNION
                    (SELECT e1.name, r.relationship_type,
                            COALESCE(r.confidence, 0.5)
                     FROM entities e2
                     JOIN relationships r ON r.target_id = e2.id
                     JOIN entities e1 ON r.source_id = e1.id
                     WHERE e2.name = %s AND e1.entity_type IN ('gene', 'protein')
                       AND e1.name != %s
                     ORDER BY r.confidence DESC NULLS LAST LIMIT %s)
                ) sub WHERE sub.conf >= %s
                ORDER BY sub.conf DESC LIMIT %s
            """, (gene, gene, max_neighbors, gene, gene, max_neighbors,
                  min_confidence, max_neighbors))
            for row in cur.fetchall():
                neighbors.append({
                    "gene": row[0],
                    "relationship": row[1],
                    "confidence": float(row[2]),
                })
    except Exception:
        pass
    finally:
        conn.close()

    return neighbors


# ---------------------------------------------------------------------------
# Expanded gene selection
# ---------------------------------------------------------------------------

def get_expanded_gene(
    original_gene: str,
    action_type: str,
    state: ResearchState,
    max_neighbors: int = 10,
    min_confidence: float = 0.4,
) -> str:
    """Pick the highest-confidence unexplored neighbor gene.

    Tries Galen's 731K-entity cancer KG first for cross-disease neighbors,
    then falls back to erik_core KG, then to original_gene.
    """
    neighbors = get_gene_neighbors_galen(original_gene, max_neighbors, min_confidence)
    if not neighbors:
        neighbors = get_gene_neighbors(original_gene, max_neighbors, min_confidence)
    if not neighbors:
        return original_gene

    # Filter out genes already expanded for this action type
    already_expanded = set(state.expansion_gene_history.get(action_type, []))

    for neighbor in neighbors:
        candidate = neighbor["gene"]
        if candidate not in already_expanded:
            return candidate

    # All neighbors exhausted — fall back to original
    return original_gene


# ---------------------------------------------------------------------------
# Expanded query generation (template-based)
# ---------------------------------------------------------------------------

_EXPANSION_TEMPLATES = [
    "ALS {neighbor} {relationship} {original} motor neuron {year}",
    "{neighbor} neuroprotection ALS therapeutic {year}",
    "ALS {neighbor} pathway mechanism {original} interaction {year}",
    "{neighbor} ALS clinical trial drug target {year}",
]


def get_expanded_queries(
    original_gene: str,
    neighbors: list[dict],
    state: ResearchState,
    year: int = 2026,
) -> list[str]:
    """Generate novel PubMed queries using KG neighbor context.

    Uses template-based expansion (no LLM) for speed and reliability.
    Filters against expansion_query_history to avoid repeats.
    """
    if not neighbors:
        return []

    history_set = set(state.expansion_query_history)
    queries: list[str] = []

    for neighbor in neighbors:
        for template in _EXPANSION_TEMPLATES:
            query = template.format(
                neighbor=neighbor["gene"],
                original=original_gene,
                relationship=neighbor["relationship"].replace("_", " "),
                year=year,
            )
            normalized = _normalize_query(query)
            if normalized not in history_set:
                queries.append(query)
                history_set.add(normalized)
            if len(queries) >= 3:
                return queries

    return queries


# ---------------------------------------------------------------------------
# Query normalization for dedup
# ---------------------------------------------------------------------------

_YEAR_PATTERN = re.compile(r"\b20\d{2}\b")


def _normalize_query(query: str) -> str:
    """Normalize a query for dedup: lowercase, strip year, sort terms."""
    q = query.lower().strip()
    q = _YEAR_PATTERN.sub("", q)
    words = sorted(q.split())
    return " ".join(w for w in words if w)


# ---------------------------------------------------------------------------
# History management
# ---------------------------------------------------------------------------

def _cap_history(history: list[str], max_size: int = 500) -> list[str]:
    """Cap history to max_size, keeping most recent entries."""
    if len(history) <= max_size:
        return history
    return history[-max_size:]


# ---------------------------------------------------------------------------
# Exhaustion key extraction
# ---------------------------------------------------------------------------

def get_exhaustion_key(action_value: str, params: dict) -> str | None:
    """Extract a target:action exhaustion key from action params.

    Returns e.g. "SOD1:query_clinvar" or None if no target identifiable.
    """
    gene = (
        params.get("gene")
        or params.get("gene_symbol")
        or params.get("target_gene")
        or params.get("target_name")
    )
    if gene:
        if isinstance(gene, list):
            gene = gene[0] if gene else None
        if gene:
            return f"{gene}:{action_value}"
    return None
