"""Action selection policy — uncertainty-directed + hypothesis-guided.

Uses a round-robin interleaving strategy to ensure action diversity:
- Every 3rd step: evidence acquisition (PubMed, trials, pathways, PPI)
- Between acquisitions: chain deepening, hypothesis generation/validation
- Protocol regeneration triggers when evidence threshold is reached

Priority order within each cycle:
1. REGENERATE_PROTOCOL when enough new evidence accumulated
2. VALIDATE_HYPOTHESIS when pending hypotheses exist
3. DEEPEN_CAUSAL_CHAIN (one chain per cycle, round-robin)
4. GENERATE_HYPOTHESIS when uncertainties exist
5. Evidence acquisition (PubMed / pathways / PPI — rotated)
"""
from __future__ import annotations

from typing import Any

from research.actions import ActionType, build_action_params
from research.state import ResearchState, ALL_LAYERS


LAYER_SEARCH_QUERIES: dict[str, str] = {
    "root_cause_suppression": "ALS TDP-43 loss-of-function therapy 2024 2025 2026",
    "pathology_reversal": "ALS sigma-1R proteostasis aggregation reversal therapy",
    "circuit_stabilization": "ALS neuroprotection glutamate excitotoxicity riluzole combination",
    "regeneration_reinnervation": "ALS motor neuron regeneration NMJ reinnervation neurotrophic",
    "adaptive_maintenance": "ALS biomarker neurofilament monitoring disease progression",
}

# Evidence acquisition actions rotated through on acquisition steps
_ACQUISITION_ROTATION = [
    ActionType.SEARCH_PUBMED,
    ActionType.SEARCH_TRIALS,
    ActionType.QUERY_PATHWAYS,
    ActionType.QUERY_PPI_NETWORK,
    ActionType.SEARCH_PUBMED,  # PubMed gets double weight
]

# How often to force an evidence acquisition step (every N steps)
_ACQUISITION_INTERVAL = 3


def select_action(
    state: ResearchState,
    regen_threshold: int = 10,
    target_depth: int = 5,
    exploration_fraction: float = 0.15,
) -> tuple[ActionType, dict[str, Any]]:
    """Select the next research action.

    Returns (action_type, params_dict).
    """
    step = state.step_count

    # 1. Protocol regeneration — highest priority
    if state.new_evidence_since_regen >= regen_threshold and state.protocol_version >= 1:
        return ActionType.REGENERATE_PROTOCOL, build_action_params(ActionType.REGENERATE_PROTOCOL)

    # 2. Forced evidence acquisition every N steps to ensure diversity
    if step > 0 and step % _ACQUISITION_INTERVAL == 0:
        return _select_acquisition_action(state)

    # 3. Validate pending hypotheses
    if state.active_hypotheses:
        hyp_id = state.active_hypotheses[0]
        return ActionType.VALIDATE_HYPOTHESIS, build_action_params(
            ActionType.VALIDATE_HYPOTHESIS, hypothesis_id=hyp_id,
        )

    # 4. Deepen ONE shallow causal chain (round-robin by step count)
    shallow_chains = [(k, v) for k, v in state.causal_chains.items() if v < target_depth]
    if shallow_chains:
        idx = step % len(shallow_chains)
        int_id, _depth = shallow_chains[idx]
        return ActionType.DEEPEN_CAUSAL_CHAIN, build_action_params(
            ActionType.DEEPEN_CAUSAL_CHAIN, intervention_id=int_id,
        )

    # 5. Generate hypothesis — the intelligence module picks the best gap
    #    (top_uncertainties is checked by the intelligence module internally)
    if state.top_uncertainties or state.protocol_version > 0:
        return ActionType.GENERATE_HYPOTHESIS, build_action_params(
            ActionType.GENERATE_HYPOTHESIS,
        )

    # 6. Fallback: evidence acquisition
    return _select_acquisition_action(state)


def _select_acquisition_action(
    state: ResearchState,
) -> tuple[ActionType, dict[str, Any]]:
    """Select an evidence acquisition action using rotation."""
    step = state.step_count
    rotation_idx = (step // _ACQUISITION_INTERVAL) % len(_ACQUISITION_ROTATION)
    action = _ACQUISITION_ROTATION[rotation_idx]

    if action == ActionType.SEARCH_PUBMED:
        layer_idx = step % len(ALL_LAYERS)
        layer = ALL_LAYERS[layer_idx]
        query = LAYER_SEARCH_QUERIES.get(layer, f"ALS {layer.replace('_', ' ')} treatment")
        return action, build_action_params(action, query=query)

    elif action == ActionType.SEARCH_TRIALS:
        return action, build_action_params(action)

    elif action == ActionType.QUERY_PATHWAYS:
        # Rotate through intervention targets
        targets = list(state.causal_chains.keys())
        if targets:
            target_idx = step % len(targets)
            return action, build_action_params(action, target_name=targets[target_idx].replace("int:", ""))
        # Fallback to PubMed if no targets
        return ActionType.SEARCH_PUBMED, build_action_params(
            ActionType.SEARCH_PUBMED, query="ALS treatment 2025 2026",
        )

    elif action == ActionType.QUERY_PPI_NETWORK:
        # Query STRING for a protocol intervention's gene target
        from targets.als_targets import ALS_TARGETS
        target_genes = [t["gene"] for t in ALS_TARGETS.values() if t.get("gene")]
        if target_genes:
            gene_idx = step % len(target_genes)
            return action, build_action_params(action, gene_symbol=target_genes[gene_idx])
        return ActionType.SEARCH_PUBMED, build_action_params(
            ActionType.SEARCH_PUBMED, query="ALS protein interaction network",
        )

    # Shouldn't reach here, but fallback
    return ActionType.SEARCH_PUBMED, build_action_params(
        ActionType.SEARCH_PUBMED, query="ALS treatment",
    )


def _uncertainty_to_layer(uncertainty: str) -> str:
    """Map an uncertainty description to the most relevant protocol layer."""
    uncertainty_lower = uncertainty.lower()
    if any(kw in uncertainty_lower for kw in ("genetic", "subtype", "root cause", "tdp-43", "sod1")):
        return "root_cause_suppression"
    if any(kw in uncertainty_lower for kw in ("pathology", "aggregation", "proteostasis")):
        return "pathology_reversal"
    if any(kw in uncertainty_lower for kw in ("circuit", "glutamate", "excitotox")):
        return "circuit_stabilization"
    if any(kw in uncertainty_lower for kw in ("regenerat", "reinnervat", "neurotrophic")):
        return "regeneration_reinnervation"
    return "adaptive_maintenance"
