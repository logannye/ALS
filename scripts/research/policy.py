"""Action selection policy — balanced cycle with validation budget.

Enforces a repeating 5-step cycle to ensure action diversity:

  Step 0: Evidence acquisition (PubMed/trials/pathways/PPI — rotated)
  Step 1: Reasoning (chain deepening OR hypothesis generation)
  Step 2: Evidence acquisition
  Step 3: Hypothesis validation (if pending, max 2 consecutive)
  Step 4: Evidence acquisition

Protocol regeneration preempts the cycle when the evidence threshold
is reached. Hypotheses that fail validation 3 times are expired.

Thompson sampling policy (thompson_policy_enabled=true in config) uses
Beta posteriors over action types to adapt selection based on outcomes.
"""
from __future__ import annotations

import random
from typing import Any

from research.actions import ActionType, build_action_params
from research.state import ResearchState, ALL_LAYERS


import datetime as _dt

# Base query bank per layer — rotated by step count for diversity
_BASE_LAYER_QUERIES: dict[str, list[str]] = {
    "root_cause_suppression": [
        "ALS TDP-43 loss-of-function therapy",
        "ALS gene therapy ASO intrabody clinical trial",
        "ALS C9orf72 repeat expansion treatment",
        "ALS SOD1 silencing tofersen long-term outcome",
    ],
    "pathology_reversal": [
        "ALS sigma-1R proteostasis aggregation reversal therapy",
        "ALS TDP-43 aggregation clearance autophagy therapeutic",
        "ALS cryptic exon splicing UNC13A STMN2 rescue",
        "ALS protein misfolding chaperone therapy",
    ],
    "circuit_stabilization": [
        "ALS neuroprotection glutamate excitotoxicity riluzole combination",
        "ALS GABA interneuron inhibitory circuit therapy",
        "ALS cortical hyperexcitability membrane stabilizer",
        "ALS ion channel modulator motor neuron survival",
    ],
    "regeneration_reinnervation": [
        "ALS motor neuron regeneration NMJ reinnervation neurotrophic",
        "ALS BDNF GDNF CNTF neurotrophic factor delivery",
        "ALS Schwann cell transplant remyelination motor nerve",
        "ALS Nogo receptor antagonist axonal growth sprouting",
    ],
    "adaptive_maintenance": [
        "ALS biomarker neurofilament monitoring disease progression",
        "ALS ALSFRS-R prediction model treatment response",
        "ALS digital biomarker wearable monitoring",
        "ALS blood biomarker p75 NTR TDP-43 CSF",
    ],
}


def get_layer_query(layer: str, step: int) -> str:
    """Return a rotating query string with year suffix for freshness."""
    queries = _BASE_LAYER_QUERIES.get(layer, [f"ALS {layer.replace('_', ' ')} treatment"])
    year = _dt.datetime.now().year
    base = queries[step % len(queries)]
    return f"{base} {year}"


# Legacy constant — kept for backward compat with any external readers
LAYER_SEARCH_QUERIES: dict[str, str] = {
    layer: queries[0] for layer, queries in _BASE_LAYER_QUERIES.items()
}

_ACQUISITION_ROTATION = [
    ActionType.SEARCH_PUBMED,
    ActionType.SEARCH_TRIALS,
    ActionType.QUERY_PATHWAYS,
    ActionType.QUERY_PPI_NETWORK,
    ActionType.CHECK_PHARMACOGENOMICS,
    ActionType.QUERY_GALEN_KG,
    ActionType.SEARCH_PREPRINTS,
    ActionType.QUERY_GALEN_SCM,
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_action(
    state: ResearchState,
    regen_threshold: int = 10,
    target_depth: int = 5,
    exploration_fraction: float = 0.15,
) -> tuple[ActionType, dict[str, Any]]:
    """Select next research action. Dispatches to Thompson or cycle."""
    try:
        from config.loader import ConfigLoader
        cfg = ConfigLoader()
    except Exception:
        cfg = None
    if cfg and cfg.get("thompson_policy_enabled", False):
        return select_action_thompson(state, regen_threshold, target_depth)
    return _select_action_cycle(state, regen_threshold, target_depth, exploration_fraction)


# ---------------------------------------------------------------------------
# Thompson sampling policy
# ---------------------------------------------------------------------------

def _update_posteriors(
    posteriors: dict[str, tuple[float, float]],
    key: str,
    success: bool,
) -> dict[str, tuple[float, float]]:
    """Update Beta posterior for an action-context key."""
    result = dict(posteriors)
    alpha, beta = result.get(key, (1.0, 1.0))
    if success:
        alpha += 1.0
    else:
        beta += 1.0
    result[key] = (alpha, beta)
    return result


def _apply_decay(
    posteriors: dict[str, tuple[float, float]],
    rate: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Apply multiplicative decay with floor at (1.0, 1.0)."""
    return {
        k: (max(1.0, a * rate), max(1.0, b * rate))
        for k, (a, b) in posteriors.items()
    }


def select_action_thompson(
    state: ResearchState,
    regen_threshold: int = 10,
    target_depth: int = 5,
) -> tuple[ActionType, dict[str, Any]]:
    """Select action via Thompson sampling over Beta posteriors."""
    # Preempt: protocol regeneration
    if state.new_evidence_since_regen >= regen_threshold and state.protocol_version >= 1:
        return ActionType.REGENERATE_PROTOCOL, build_action_params(ActionType.REGENERATE_PROTOCOL)

    # Build the full candidate action type list
    all_types: list[ActionType] = list(set(_ACQUISITION_ROTATION))
    all_types.extend([ActionType.GENERATE_HYPOTHESIS, ActionType.DEEPEN_CAUSAL_CHAIN])
    if hasattr(ActionType, "CHALLENGE_INTERVENTION"):
        all_types.append(ActionType.CHALLENGE_INTERVENTION)

    # Diversity floor: force any action type not used in 30 steps
    for at in all_types:
        last_used = state.last_action_per_type.get(at.value, 0)
        if state.step_count - last_used >= 30:
            return _build_thompson_params(at, state, target_depth)

    # Thompson sampling: draw from each Beta and pick the argmax
    posteriors = state.action_posteriors or {}
    best_action: ActionType | None = None
    best_sample = -1.0
    for at in all_types:
        alpha, beta = posteriors.get(at.value, (1.0, 1.0))
        sample = random.betavariate(max(alpha, 0.01), max(beta, 0.01))
        if sample > best_sample:
            best_sample = sample
            best_action = at

    return _build_thompson_params(
        best_action or ActionType.SEARCH_PUBMED, state, target_depth
    )


def _build_thompson_params(
    action: ActionType,
    state: ResearchState,
    target_depth: int,
) -> tuple[ActionType, dict[str, Any]]:
    """Build params for Thompson-selected action using existing helpers."""
    if action == ActionType.GENERATE_HYPOTHESIS:
        return ActionType.GENERATE_HYPOTHESIS, build_action_params(ActionType.GENERATE_HYPOTHESIS)
    elif action == ActionType.DEEPEN_CAUSAL_CHAIN:
        shallow = [(k, v) for k, v in state.causal_chains.items() if v < target_depth]
        if shallow:
            int_id = shallow[state.step_count % len(shallow)][0]
            return action, build_action_params(action, intervention_id=int_id)
        return ActionType.GENERATE_HYPOTHESIS, build_action_params(ActionType.GENERATE_HYPOTHESIS)
    elif action == ActionType.CHALLENGE_INTERVENTION:
        return action, build_action_params(action)
    elif action == ActionType.REGENERATE_PROTOCOL:
        return action, build_action_params(action)
    else:
        # Acquisition action — delegate to per-type helper
        return _build_acquisition_params(action, state, state.step_count)


def _select_acquisition_action_for_type(
    action: ActionType,
    state: ResearchState,
) -> tuple[ActionType, dict[str, Any]]:
    """Build params for a specific acquisition action type.

    DEPRECATED — use _build_acquisition_params() instead.
    Kept for backward compatibility with any external callers.
    """
    step = state.step_count

    if action == ActionType.SEARCH_PUBMED:
        layer_idx = (step // _CYCLE_LENGTH) % len(ALL_LAYERS)
        layer = ALL_LAYERS[layer_idx]
        query = get_layer_query(layer, step)
        return action, build_action_params(action, query=query, protocol_layer=layer)

    elif action == ActionType.SEARCH_TRIALS:
        return action, build_action_params(action, protocol_layer="circuit_stabilization")

    elif action == ActionType.QUERY_PATHWAYS:
        from targets.als_targets import ALS_TARGETS
        target_keys = list(ALS_TARGETS.keys())
        if target_keys:
            target_idx = step % len(target_keys)
            target_key = target_keys[target_idx]
            target = ALS_TARGETS[target_key]
            layers = target.get("protocol_layers", ["root_cause_suppression"])
            return action, build_action_params(
                action,
                target_name=target_key,
                uniprot_id=target.get("uniprot_id", ""),
                gene_symbol=target.get("gene", ""),
                protocol_layer=layers[0] if layers else "root_cause_suppression",
            )
        return _fallback_acquisition(state, step, skip=ActionType.QUERY_PATHWAYS)

    elif action == ActionType.QUERY_PPI_NETWORK:
        from targets.als_targets import ALS_TARGETS
        target_genes = [(k, t["gene"]) for k, t in ALS_TARGETS.items() if t.get("gene")]
        if target_genes:
            gene_idx = step % len(target_genes)
            target_key, gene = target_genes[gene_idx]
            target = ALS_TARGETS[target_key]
            layers = target.get("protocol_layers", ["root_cause_suppression"])
            return action, build_action_params(
                action,
                gene_symbol=gene,
                protocol_layer=layers[0] if layers else "root_cause_suppression",
            )
        return _fallback_acquisition(state, step, skip=ActionType.QUERY_PPI_NETWORK)

    elif action == ActionType.CHECK_PHARMACOGENOMICS:
        drugs = ["riluzole", "edaravone", "rapamycin", "pridopidine", "masitinib"]
        drug_idx = step % len(drugs)
        return action, build_action_params(
            action, drug_name=drugs[drug_idx], protocol_layer="adaptive_maintenance"
        )

    elif action == ActionType.QUERY_GALEN_KG:
        from connectors.galen_kg import ALS_CROSS_REFERENCE_GENES
        gene_idx = step % len(ALS_CROSS_REFERENCE_GENES)
        gene = ALS_CROSS_REFERENCE_GENES[gene_idx]
        return action, build_action_params(
            action, genes=[gene], protocol_layer="root_cause_suppression"
        )

    elif action == ActionType.SEARCH_PREPRINTS:
        try:
            from config.loader import ConfigLoader
            cfg = ConfigLoader()
        except Exception:
            cfg = None
        if cfg and not cfg.get("biorxiv_enabled", True):
            return _fallback_acquisition(state, step, skip=ActionType.SEARCH_PREPRINTS)
        layer_idx = (step // _CYCLE_LENGTH) % len(ALL_LAYERS)
        layer = ALL_LAYERS[layer_idx]
        query = get_layer_query(layer, step)
        return action, build_action_params(action, query=query, protocol_layer=layer)

    elif action == ActionType.QUERY_GALEN_SCM:
        try:
            from config.loader import ConfigLoader
            cfg = ConfigLoader()
        except Exception:
            cfg = None
        if cfg and not cfg.get("galen_scm_enabled", True):
            return _fallback_acquisition(state, step, skip=ActionType.QUERY_GALEN_SCM)
        from connectors.galen_kg import ALS_CROSS_REFERENCE_GENES
        gene_idx = step % len(ALS_CROSS_REFERENCE_GENES)
        gene = ALS_CROSS_REFERENCE_GENES[gene_idx]
        return action, build_action_params(action, target_gene=gene, protocol_layer="root_cause_suppression")

    # Unhandled type — fall back to PubMed
    return _fallback_acquisition(state, step, skip=action)


# ---------------------------------------------------------------------------
# Cycle-based policy (original logic, now private)
# ---------------------------------------------------------------------------

def _select_action_cycle(
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
    """Select an evidence acquisition action using rotation with yield-aware skip.

    Actions that have been tried ``yield_skip_min_count`` times with an EMA
    action value below ``yield_skip_threshold`` are skipped in favour of the
    next candidate in rotation.  This prevents the loop from wasting cycles
    on consistently-zero-yield actions.
    """
    # Read config thresholds (hot-reloadable)
    try:
        from config.loader import ConfigLoader
        _cfg = ConfigLoader()
    except Exception:
        _cfg = None
    min_count = _cfg.get("yield_skip_min_count", 5) if _cfg else 5
    threshold = _cfg.get("yield_skip_threshold", 0.1) if _cfg else 0.1

    step = state.step_count
    rotation_idx = step % len(_ACQUISITION_ROTATION)

    # Try each candidate in rotation order, skipping exhausted actions
    for offset in range(len(_ACQUISITION_ROTATION)):
        idx = (rotation_idx + offset) % len(_ACQUISITION_ROTATION)
        action = _ACQUISITION_ROTATION[idx]

        count = state.action_counts.get(action.value, 0)
        value = state.action_values.get(action.value, 1.0)  # Optimistic default
        if count >= min_count and value < threshold:
            continue  # Skip: this action consistently yields nothing

        return _build_acquisition_params(action, state, step)

    # All exhausted — ultimate fallback to SEARCH_PUBMED
    return _fallback_acquisition(state, step, skip=None)


def _build_acquisition_params(
    action: ActionType,
    state: ResearchState,
    step: int,
) -> tuple[ActionType, dict[str, Any]]:
    """Build params for a specific acquisition action."""
    if action == ActionType.SEARCH_PUBMED:
        layer_idx = (step // _CYCLE_LENGTH) % len(ALL_LAYERS)
        layer = ALL_LAYERS[layer_idx]
        query = get_layer_query(layer, step)
        return action, build_action_params(action, query=query, protocol_layer=layer)

    elif action == ActionType.SEARCH_TRIALS:
        return action, build_action_params(action, protocol_layer="circuit_stabilization")

    elif action == ActionType.QUERY_PATHWAYS:
        from targets.als_targets import ALS_TARGETS
        target_keys = list(ALS_TARGETS.keys())
        if target_keys:
            target_idx = step % len(target_keys)
            target_key = target_keys[target_idx]
            target = ALS_TARGETS[target_key]
            layers = target.get("protocol_layers", ["root_cause_suppression"])
            return action, build_action_params(
                action,
                target_name=target_key,
                uniprot_id=target.get("uniprot_id", ""),
                gene_symbol=target.get("gene", ""),
                protocol_layer=layers[0] if layers else "root_cause_suppression",
            )
        # Fallback: rotate to next acquisition action
        return _fallback_acquisition(state, step, skip=ActionType.QUERY_PATHWAYS)

    elif action == ActionType.QUERY_PPI_NETWORK:
        from targets.als_targets import ALS_TARGETS
        target_keys = list(ALS_TARGETS.keys())
        target_genes = [(k, t["gene"]) for k, t in ALS_TARGETS.items() if t.get("gene")]
        if target_genes:
            gene_idx = step % len(target_genes)
            target_key, gene = target_genes[gene_idx]
            target = ALS_TARGETS[target_key]
            layers = target.get("protocol_layers", ["root_cause_suppression"])
            return action, build_action_params(
                action,
                gene_symbol=gene,
                protocol_layer=layers[0] if layers else "root_cause_suppression",
            )
        # Fallback: rotate to next acquisition action
        return _fallback_acquisition(state, step, skip=ActionType.QUERY_PPI_NETWORK)

    elif action == ActionType.CHECK_PHARMACOGENOMICS:
        drugs = ["riluzole", "edaravone", "rapamycin", "pridopidine", "masitinib"]
        drug_idx = step % len(drugs)
        return action, build_action_params(action, drug_name=drugs[drug_idx], protocol_layer="adaptive_maintenance")

    elif action == ActionType.QUERY_GALEN_KG:
        from connectors.galen_kg import ALS_CROSS_REFERENCE_GENES
        gene_idx = step % len(ALS_CROSS_REFERENCE_GENES)
        gene = ALS_CROSS_REFERENCE_GENES[gene_idx]
        return action, build_action_params(action, genes=[gene], protocol_layer="root_cause_suppression")

    elif action == ActionType.SEARCH_PREPRINTS:
        try:
            from config.loader import ConfigLoader
            cfg = ConfigLoader()
        except Exception:
            cfg = None
        if cfg and not cfg.get("biorxiv_enabled", True):
            return _fallback_acquisition(state, step, skip=ActionType.SEARCH_PREPRINTS)
        layer_idx = (step // _CYCLE_LENGTH) % len(ALL_LAYERS)
        layer = ALL_LAYERS[layer_idx]
        query = get_layer_query(layer, step)
        return action, build_action_params(action, query=query, protocol_layer=layer)

    elif action == ActionType.QUERY_GALEN_SCM:
        try:
            from config.loader import ConfigLoader
            cfg = ConfigLoader()
        except Exception:
            cfg = None
        if cfg and not cfg.get("galen_scm_enabled", True):
            return _fallback_acquisition(state, step, skip=ActionType.QUERY_GALEN_SCM)
        from connectors.galen_kg import ALS_CROSS_REFERENCE_GENES
        gene_idx = step % len(ALS_CROSS_REFERENCE_GENES)
        gene = ALS_CROSS_REFERENCE_GENES[gene_idx]
        return action, build_action_params(action, target_gene=gene, protocol_layer="root_cause_suppression")

    return _fallback_acquisition(state, step, skip=action)


def _fallback_acquisition(
    state: ResearchState,
    step: int,
    skip: ActionType | None,
) -> tuple[ActionType, dict[str, Any]]:
    """When the preferred acquisition action can't execute, try the next one in rotation."""
    for offset in range(1, len(_ACQUISITION_ROTATION)):
        next_idx = (step + offset) % len(_ACQUISITION_ROTATION)
        candidate = _ACQUISITION_ROTATION[next_idx]
        if candidate != skip and candidate == ActionType.SEARCH_PUBMED:
            layer_idx = (step // _CYCLE_LENGTH) % len(ALL_LAYERS)
            layer = ALL_LAYERS[layer_idx]
            query = get_layer_query(layer, step)
            return candidate, build_action_params(candidate, query=query, protocol_layer=layer)
    # Ultimate fallback
    return ActionType.SEARCH_PUBMED, build_action_params(
        ActionType.SEARCH_PUBMED, query="ALS treatment 2025 2026",
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
