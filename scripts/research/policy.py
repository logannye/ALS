"""Action selection policy — balanced cycle with validation budget.

Enforces a repeating 5-step cycle to ensure action diversity:

  Step 0: Evidence acquisition (PubMed/trials/pathways/PPI — rotated)
  Step 1: Reasoning (chain deepening OR hypothesis generation)
  Step 2: Evidence acquisition
  Step 3: Hypothesis validation (if pending, max 2 consecutive)
  Step 4: Evidence acquisition

Protocol regeneration preempts the cycle when the evidence threshold
is reached. Hypotheses that fail validation 3 times are expired.
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

_ACQUISITION_ROTATION = [
    ActionType.SEARCH_PUBMED,
    ActionType.SEARCH_TRIALS,
    ActionType.QUERY_PATHWAYS,
    ActionType.QUERY_PPI_NETWORK,
    ActionType.CHECK_PHARMACOGENOMICS,
    ActionType.SEARCH_PUBMED,
]

# The balanced 5-step cycle
_CYCLE_LENGTH = 5
_ACQUIRE_STEPS = {0, 2, 4}   # 3 of 5 steps are acquisition
_REASON_STEP = 1              # 1 of 5 is reasoning (chain/hypothesis gen)
_VALIDATE_STEP = 3            # 1 of 5 is validation

# Max consecutive validation attempts before forcing something else
_MAX_CONSECUTIVE_VALIDATIONS = 2

# Max validation attempts per hypothesis before expiry
_MAX_VALIDATION_ATTEMPTS_PER_HYP = 3


def select_action(
    state: ResearchState,
    regen_threshold: int = 10,
    target_depth: int = 5,
    exploration_fraction: float = 0.15,
) -> tuple[ActionType, dict[str, Any]]:
    """Select the next research action using a balanced cycle.

    Returns (action_type, params_dict).
    """
    step = state.step_count
    cycle_pos = step % _CYCLE_LENGTH

    # 0. Protocol regeneration — always highest priority
    if state.new_evidence_since_regen >= regen_threshold and state.protocol_version >= 1:
        return ActionType.REGENERATE_PROTOCOL, build_action_params(ActionType.REGENERATE_PROTOCOL)

    # Expire stale hypotheses (validated too many times without resolution)
    _maybe_expire_hypotheses(state)

    # 1. Acquisition steps (positions 0, 2, 4 in the cycle)
    if cycle_pos in _ACQUIRE_STEPS:
        return _select_acquisition_action(state)

    # 2. Reasoning step (position 1): chain deepening or hypothesis generation
    if cycle_pos == _REASON_STEP:
        return _select_reasoning_action(state, target_depth)

    # 3. Validation step (position 3): validate if pending, otherwise acquire
    if cycle_pos == _VALIDATE_STEP:
        # Check if we've been validating too much consecutively
        consecutive_validations = _count_consecutive_validations(state)
        if state.active_hypotheses and consecutive_validations < _MAX_CONSECUTIVE_VALIDATIONS:
            hyp_id = state.active_hypotheses[0]
            return ActionType.VALIDATE_HYPOTHESIS, build_action_params(
                ActionType.VALIDATE_HYPOTHESIS, hypothesis_id=hyp_id,
            )
        # No valid hypotheses or budget exhausted — acquire instead
        return _select_acquisition_action(state)

    # Fallback
    return _select_acquisition_action(state)


def _select_reasoning_action(
    state: ResearchState,
    target_depth: int,
) -> tuple[ActionType, dict[str, Any]]:
    """Pick between chain deepening and hypothesis generation."""
    step = state.step_count

    # Alternate: even reasoning steps deepen chains, odd generate hypotheses
    if (step // _CYCLE_LENGTH) % 2 == 0:
        # Try chain deepening
        shallow = [(k, v) for k, v in state.causal_chains.items() if v < target_depth]
        if shallow:
            idx = step % len(shallow)
            int_id, _ = shallow[idx]
            return ActionType.DEEPEN_CAUSAL_CHAIN, build_action_params(
                ActionType.DEEPEN_CAUSAL_CHAIN, intervention_id=int_id,
            )

    # Generate hypothesis (intelligence module picks the gap)
    if state.top_uncertainties or state.protocol_version > 0:
        return ActionType.GENERATE_HYPOTHESIS, build_action_params(
            ActionType.GENERATE_HYPOTHESIS,
        )

    # Fallback to acquisition
    return _select_acquisition_action(state)


def _select_acquisition_action(
    state: ResearchState,
) -> tuple[ActionType, dict[str, Any]]:
    """Select an evidence acquisition action using rotation."""
    step = state.step_count
    rotation_idx = step % len(_ACQUISITION_ROTATION)
    action = _ACQUISITION_ROTATION[rotation_idx]

    if action == ActionType.SEARCH_PUBMED:
        layer_idx = (step // _CYCLE_LENGTH) % len(ALL_LAYERS)
        layer = ALL_LAYERS[layer_idx]
        query = LAYER_SEARCH_QUERIES.get(layer, f"ALS {layer.replace('_', ' ')} treatment")
        return action, build_action_params(action, query=query)

    elif action == ActionType.SEARCH_TRIALS:
        return action, build_action_params(action)

    elif action == ActionType.QUERY_PATHWAYS:
        targets = list(state.causal_chains.keys())
        if targets:
            target_idx = step % len(targets)
            return action, build_action_params(action, target_name=targets[target_idx].replace("int:", ""))
        return ActionType.SEARCH_PUBMED, build_action_params(
            ActionType.SEARCH_PUBMED, query="ALS treatment 2025 2026",
        )

    elif action == ActionType.QUERY_PPI_NETWORK:
        from targets.als_targets import ALS_TARGETS
        target_genes = [t["gene"] for t in ALS_TARGETS.values() if t.get("gene")]
        if target_genes:
            gene_idx = step % len(target_genes)
            return action, build_action_params(action, gene_symbol=target_genes[gene_idx])
        return ActionType.SEARCH_PUBMED, build_action_params(
            ActionType.SEARCH_PUBMED, query="ALS protein interaction 2025",
        )

    elif action == ActionType.CHECK_PHARMACOGENOMICS:
        drugs = ["riluzole", "edaravone", "rapamycin", "pridopidine", "masitinib"]
        drug_idx = step % len(drugs)
        return action, build_action_params(action, drug_name=drugs[drug_idx])

    return ActionType.SEARCH_PUBMED, build_action_params(
        ActionType.SEARCH_PUBMED, query="ALS treatment",
    )


def _count_consecutive_validations(state: ResearchState) -> int:
    """Count how many of the last N actions were validate_hypothesis."""
    if state.last_action != "validate_hypothesis":
        return 0
    # Approximate: check action counts ratio
    val_count = state.action_counts.get("validate_hypothesis", 0)
    total = sum(state.action_counts.values())
    if total == 0:
        return 0
    # If more than 40% of all actions are validation, we're over-validating
    if val_count / total > 0.4:
        return _MAX_CONSECUTIVE_VALIDATIONS  # Force budget exhaustion
    return 1 if state.last_action == "validate_hypothesis" else 0


def _maybe_expire_hypotheses(state: ResearchState) -> None:
    """Remove hypotheses that have been validated too many times without resolution.

    Modifies state.active_hypotheses in place (mutable list).
    """
    val_count = state.action_counts.get("validate_hypothesis", 0)
    hyp_count = len(state.active_hypotheses)

    if hyp_count > 0 and val_count > 0:
        # Average validations per hypothesis
        avg_validations = val_count / max(hyp_count + state.resolved_hypotheses, 1)
        if avg_validations > _MAX_VALIDATION_ATTEMPTS_PER_HYP:
            # Expire the oldest hypothesis
            if state.active_hypotheses:
                expired = state.active_hypotheses.pop(0)
                state.resolved_hypotheses += 1  # Count as resolved (inconclusive)
