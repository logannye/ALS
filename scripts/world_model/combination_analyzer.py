"""Drug combination synergy modeling for the Erik ALS research engine.

Analyzes pairwise pathway overlap between causal chains to detect redundancy,
antagonism, or synergy among proposed protocol interventions.
"""
from __future__ import annotations

import copy
from typing import Literal, Optional

from pydantic import BaseModel, Field

from research.causal_chains import CausalChain


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class InteractionFlag(BaseModel):
    """A detected interaction between two protocol interventions."""

    intervention_a: str
    intervention_b: str
    interaction_type: Literal["synergy", "antagonism", "redundancy"]
    mechanism: str
    confidence: float = 0.0
    cited_evidence: list[str] = Field(default_factory=list)


class CombinationAnalysis(BaseModel):
    """Aggregate result of analyzing all intervention pair interactions."""

    flags: list[InteractionFlag] = Field(default_factory=list)
    overall_coherence: float = 1.0
    suggested_substitutions: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_pathway_overlap(chain_a: CausalChain, chain_b: CausalChain) -> float:
    """Compute Jaccard similarity of intermediate pathway nodes between two chains.

    Intermediate nodes are the *target* endpoints of each link, lowercased.
    Returns 0.0 if either chain has no links.
    """
    nodes_a = {link.target.lower() for link in chain_a.links}
    nodes_b = {link.target.lower() for link in chain_b.links}

    if not nodes_a or not nodes_b:
        return 0.0

    intersection = nodes_a & nodes_b
    union = nodes_a | nodes_b

    return len(intersection) / len(union)


def analyze_combinations(
    chains: dict[str, CausalChain],
    overlap_threshold: float = 0.6,
) -> CombinationAnalysis:
    """Analyze all pairwise chain overlaps and flag redundant intervention pairs.

    Pairs whose Jaccard pathway overlap meets or exceeds *overlap_threshold* are
    flagged as "redundancy". Overall coherence is reduced by 0.2 per flag,
    clamped to [0.0, 1.0].
    """
    ids = list(chains.keys())
    flags: list[InteractionFlag] = []

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id_a, id_b = ids[i], ids[j]
            overlap = compute_pathway_overlap(chains[id_a], chains[id_b])
            if overlap >= overlap_threshold:
                flags.append(InteractionFlag(
                    intervention_a=id_a,
                    intervention_b=id_b,
                    interaction_type="redundancy",
                    mechanism="high_pathway_overlap",
                    confidence=overlap,
                ))

    coherence = max(0.0, min(1.0, 1.0 - len(flags) * 0.2))
    return CombinationAnalysis(flags=flags, overall_coherence=coherence)


def apply_interaction_flags(
    scores_by_layer: dict[str, list[dict]],
    flags: list[InteractionFlag],
    threshold: float = 0.7,
) -> dict[str, list[dict]]:
    """Apply interaction flags to scored intervention lists.

    For each antagonism or redundancy flag whose confidence meets *threshold*:
    - Locate both interventions across all layers.
    - Identify the lower-scoring one and remove it from its layer list.

    Synergy flags are ignored (no swap needed).
    Returns a deep copy — the original *scores_by_layer* is not mutated.
    """
    result = copy.deepcopy(scores_by_layer)

    for flag in flags:
        if flag.interaction_type == "synergy":
            continue
        if flag.confidence < threshold:
            continue

        # Locate each intervention and its score
        loc_a: Optional[tuple[str, int, float]] = None  # (layer_key, list_idx, score)
        loc_b: Optional[tuple[str, int, float]] = None

        for layer_key, entries in result.items():
            for idx, entry in enumerate(entries):
                name = entry.get("intervention", "")
                score = entry.get("score", 0.0)
                if name == flag.intervention_a and loc_a is None:
                    loc_a = (layer_key, idx, score)
                elif name == flag.intervention_b and loc_b is None:
                    loc_b = (layer_key, idx, score)

        if loc_a is None or loc_b is None:
            # One or both interventions not found — skip
            continue

        # Remove the lower-scoring one
        if loc_a[2] <= loc_b[2]:
            layer_key, idx, _ = loc_a
        else:
            layer_key, idx, _ = loc_b

        result[layer_key].pop(idx)

    return result
