"""Action selection policy — uncertainty-directed + hypothesis-guided.

Priority order:
1. REGENERATE_PROTOCOL when enough new evidence accumulated
2. VALIDATE_HYPOTHESIS when pending hypotheses exist
3. DEEPEN_CAUSAL_CHAIN when top interventions have shallow chains
4. GENERATE_HYPOTHESIS when top uncertainties exist but no hypotheses pending
5. SEARCH_PUBMED as systematic layer-rotation fallback
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

def select_action(
    state: ResearchState,
    regen_threshold: int = 10,
    target_depth: int = 5,
    exploration_fraction: float = 0.15,
) -> tuple[ActionType, dict[str, Any]]:
    # 1. Protocol regeneration if enough new evidence (or first protocol if version 0)
    if state.new_evidence_since_regen >= regen_threshold and state.protocol_version >= 1:
        return ActionType.REGENERATE_PROTOCOL, build_action_params(ActionType.REGENERATE_PROTOCOL)

    # 2. Validate pending hypotheses
    if state.active_hypotheses:
        hyp_id = state.active_hypotheses[0]
        return ActionType.VALIDATE_HYPOTHESIS, build_action_params(
            ActionType.VALIDATE_HYPOTHESIS, hypothesis_id=hyp_id,
        )

    # 3. Deepen shallow causal chains (skip if last 3 actions were all chain deepening with 0 reward — stall detection)
    chain_stalled = (
        state.action_counts.get("deepen_causal_chain", 0) >= 3
        and state.last_action == "deepen_causal_chain"
        and state.last_reward == 0.0
        and state.action_counts.get("deepen_causal_chain", 0) > state.action_counts.get("generate_hypothesis", 0) + state.action_counts.get("search_pubmed", 0)
    )
    if not chain_stalled:
        for int_id, depth in state.causal_chains.items():
            if depth < target_depth:
                return ActionType.DEEPEN_CAUSAL_CHAIN, build_action_params(
                    ActionType.DEEPEN_CAUSAL_CHAIN, intervention_id=int_id,
                )

    # 4. Generate hypothesis targeting top uncertainty
    if state.top_uncertainties:
        uncertainty = state.top_uncertainties[0]
        return ActionType.GENERATE_HYPOTHESIS, build_action_params(
            ActionType.GENERATE_HYPOTHESIS,
            topic=_uncertainty_to_layer(uncertainty),
            uncertainty=uncertainty,
        )

    # 5. Systematic layer-rotation PubMed search
    layer_idx = state.step_count % len(ALL_LAYERS)
    layer = ALL_LAYERS[layer_idx]
    query = LAYER_SEARCH_QUERIES.get(layer, f"ALS {layer.replace('_', ' ')} treatment")
    return ActionType.SEARCH_PUBMED, build_action_params(
        ActionType.SEARCH_PUBMED, query=query,
    )

def _uncertainty_to_layer(uncertainty: str) -> str:
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
