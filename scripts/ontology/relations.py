"""Typed relation vocabulary for the Erik ALS knowledge graph.

Defines the complete set of typed relations used as edges in the knowledge
graph, organised by category.  The ``OBSERVATIONAL_RELATION_TYPES`` frozenset
enforces the Pearl Causal Hierarchy guard: these relations can NEVER be
upgraded to L3 (counterfactual) status.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# RELATION_TYPES
# ---------------------------------------------------------------------------

RELATION_TYPES: dict[str, dict[str, str]] = {
    # Structural
    "has_part":         {"category": "structural"},
    "part_of":          {"category": "structural"},
    "subtype_of":       {"category": "structural"},
    "instance_of":      {"category": "structural"},
    "located_in":       {"category": "structural"},
    "member_of":        {"category": "structural"},
    "expressed_in":     {"category": "structural"},
    # Causal
    "causes":           {"category": "causal"},
    "contributes_to":   {"category": "causal"},
    "amplifies":        {"category": "causal"},
    "suppresses":       {"category": "causal"},
    "confounds":        {"category": "causal"},
    "modifies_risk_of": {"category": "causal"},
    # Observational
    "observed_in":      {"category": "observational"},
    "measures":         {"category": "observational"},
    "derived_from":     {"category": "observational"},
    "associated_with":  {"category": "observational"},
    "variant_in_gene":  {"category": "observational"},
    # Temporal
    "precedes":         {"category": "temporal"},
    "follows":          {"category": "temporal"},
    # Evidential
    "asserts":          {"category": "evidential"},
    "inferred_from":    {"category": "evidential"},
    "supports":         {"category": "evidential"},
    "refutes":          {"category": "evidential"},
    "learned_from":     {"category": "evidential"},
    "counterfactual_of":{"category": "evidential"},
    "supersedes":       {"category": "evidential"},
    # Therapeutic
    "targets":          {"category": "therapeutic"},
    "treats":           {"category": "therapeutic"},
    "contraindicates":  {"category": "therapeutic"},
    "eligible_for":     {"category": "therapeutic"},
    "ineligible_for":   {"category": "therapeutic"},
    "resulted_in":      {"category": "therapeutic"},
    # Governance
    "constrained_by":         {"category": "governance"},
    "optimizes_for":          {"category": "governance"},
    "requires_approval_from": {"category": "governance"},
    "executed_by":            {"category": "governance"},
    "evaluated_by":           {"category": "governance"},
}


# ---------------------------------------------------------------------------
# OBSERVATIONAL_RELATION_TYPES
# ---------------------------------------------------------------------------
# These relation types describe associations, co-occurrence, or structural
# membership and MUST NEVER be promoted to L3 (counterfactual) in the Pearl
# Causal Hierarchy.  Any code that upgrades PCH layers MUST check this set
# before proceeding.

OBSERVATIONAL_RELATION_TYPES: frozenset[str] = frozenset({
    "variant_in_gene",
    "located_in",
    "member_of",
    "subtype_of",
    "instance_of",
    "observed_in",
    "has_part",
    "part_of",
    "expressed_in",
    "associated_with",
    "derived_from",
    "measures",
})


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def is_observational(relation_type: str) -> bool:
    """Return True if *relation_type* is in ``OBSERVATIONAL_RELATION_TYPES``.

    These relations can never be upgraded to L3 (counterfactual) in the
    Pearl Causal Hierarchy.  Unknown relation types return False.
    """
    return relation_type in OBSERVATIONAL_RELATION_TYPES


def get_relation_category(relation_type: str) -> str:
    """Return the category string for *relation_type*.

    Returns ``"unknown"`` for relation types that are not in
    ``RELATION_TYPES``.
    """
    entry = RELATION_TYPES.get(relation_type)
    if entry is None:
        return "unknown"
    return entry["category"]
