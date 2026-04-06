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
import json as _json
import pathlib as _pathlib

# ---------------------------------------------------------------------------
# Dynamic drug list from interventions.json
# ---------------------------------------------------------------------------

_DRUG_NAMES_CACHE: list[str] | None = None
_FALLBACK_DRUGS = ["riluzole", "edaravone", "rapamycin", "pridopidine", "masitinib"]


def _get_pharmacogenomics_drugs() -> list[str]:
    """Load drug names from interventions.json (drug/small_molecule/peptide classes).

    Cached after first call. Falls back to 5-drug list on any error.
    """
    global _DRUG_NAMES_CACHE
    if _DRUG_NAMES_CACHE is not None:
        return _DRUG_NAMES_CACHE
    try:
        path = _pathlib.Path(__file__).parent.parent.parent / "data" / "seed" / "interventions.json"
        with open(path) as f:
            items = _json.load(f)
        names: list[str] = []
        for item in items:
            cls = item.get("intervention_class", "")
            if cls in ("drug", "small_molecule", "peptide"):
                raw = item.get("name", "").split("(")[0].split("/")[0].strip()
                if raw:
                    names.append(raw.lower())
        _DRUG_NAMES_CACHE = names if names else _FALLBACK_DRUGS
    except Exception:
        _DRUG_NAMES_CACHE = _FALLBACK_DRUGS
    return _DRUG_NAMES_CACHE


# Base query bank per layer — rotated by step count for diversity
_BASE_LAYER_QUERIES: dict[str, list[str]] = {
    "root_cause_suppression": [
        "ALS TDP-43 loss-of-function therapy",
        "ALS gene therapy ASO intrabody clinical trial",
        "ALS C9orf72 repeat expansion treatment",
        "ALS SOD1 silencing tofersen long-term outcome",
        "ALS nuclear import receptor transportin-1 TDP-43",
        "ALS TARDBP splicing regulation cryptic exon therapy",
        "ALS RNA metabolism stress granule dissolution therapeutic",
        "ALS dipeptide repeat protein C9orf72 antisense oligonucleotide",
    ],
    "pathology_reversal": [
        "ALS sigma-1R proteostasis aggregation reversal therapy",
        "ALS TDP-43 aggregation clearance autophagy therapeutic",
        "ALS cryptic exon splicing UNC13A STMN2 rescue",
        "ALS protein misfolding chaperone therapy",
        "ALS ubiquitin proteasome system motor neuron degeneration",
        "ALS endoplasmic reticulum stress unfolded protein response",
        "ALS mitochondrial dysfunction complex I therapeutic",
        "ALS phase separation liquid-liquid TDP-43 therapeutic",
    ],
    "circuit_stabilization": [
        "ALS neuroprotection glutamate excitotoxicity riluzole combination",
        "ALS GABA interneuron inhibitory circuit therapy",
        "ALS cortical hyperexcitability membrane stabilizer",
        "ALS ion channel modulator motor neuron survival",
        "ALS neuromuscular junction preservation agrin LRP4",
        "ALS upper motor neuron corticospinal tract protection",
        "ALS astrocyte reactivity modulation neuroprotection",
        "ALS microglia polarization CSF1R TREM2 neuroinflammation",
    ],
    "regeneration_reinnervation": [
        "ALS motor neuron regeneration NMJ reinnervation neurotrophic",
        "ALS BDNF GDNF CNTF neurotrophic factor delivery",
        "ALS Schwann cell transplant remyelination motor nerve",
        "ALS Nogo receptor antagonist axonal growth sprouting",
        "ALS iPSC motor neuron transplant clinical trial",
        "ALS terminal sprouting compensatory reinnervation mechanism",
        "ALS exercise induced neuroplasticity motor unit recruitment",
        "ALS VEGF angiogenic factor motor neuron survival delivery",
    ],
    "adaptive_maintenance": [
        "ALS biomarker neurofilament monitoring disease progression",
        "ALS ALSFRS-R prediction model treatment response",
        "ALS digital biomarker wearable monitoring",
        "ALS blood biomarker p75 NTR TDP-43 CSF",
        "ALS respiratory function FVC sniff nasal pressure monitoring",
        "ALS nutritional status metabolic intervention body weight",
        "ALS multidisciplinary clinic survival benefit evidence",
        "ALS palliative concurrent early intervention quality life",
    ],
}


def get_layer_query(layer: str, step: int) -> str:
    """Return a rotating query string with year suffix for freshness."""
    queries = _BASE_LAYER_QUERIES.get(layer, [f"ALS {layer.replace('_', ' ')} treatment"])
    year = _dt.datetime.now().year
    base = queries[step % len(queries)]
    return f"{base} {year}"


def _get_targeted_query(state: ResearchState, step: int) -> str:
    """Generate a targeted query from ALS target gene definitions.

    Each of the 16 ALS targets has a gene symbol and name. Rotate through
    targets and query templates to systematically cover the ALS literature
    for specific molecular mechanisms.
    """
    from targets.als_targets import ALS_TARGETS
    targets = list(ALS_TARGETS.values())
    if not targets:
        return f"ALS treatment {_dt.datetime.now().year}"
    target = targets[step % len(targets)]
    gene = target.get("gene", "")
    name = target.get("name", "")
    year = _dt.datetime.now().year

    templates = [
        f"{gene} ALS mechanism motor neuron {year}",
        f"{gene} ALS therapeutic clinical trial {year}",
        f"{name} neuroprotection motor neuron survival {year}",
        f"{gene} ALS combination therapy riluzole {year}",
        f"{gene} ALS biomarker prognosis {year}",
        f"{name} ALS preclinical in vivo model {year}",
    ]
    return templates[(step // max(len(targets), 1)) % len(templates)]


def _get_dynamic_query(state: ResearchState, step: int, layer: str) -> str:
    """Build a search query from hypothesis-generated terms when available.

    Active hypotheses now store full statements (not IDs). Extract
    biomedical terms to build targeted queries that go beyond the
    static query bank.

    Falls back to static layer queries if no hypothesis terms exist.
    """
    # Walk hypotheses in reverse (most recent first) looking for useful terms
    for hyp_statement in reversed(state.active_hypotheses):
        # Skip old-format IDs that start with "hyp:"
        if hyp_statement.startswith("hyp:"):
            continue
        words = hyp_statement.replace(",", " ").replace(".", " ").replace("(", " ").replace(")", " ").split()
        bio_terms = [
            w for w in words
            if len(w) > 3 and (w[0].isupper() or any(c.isdigit() for c in w))
            and w.lower() not in {"this", "that", "with", "from", "when", "will", "would", "could", "should"}
        ]
        if len(bio_terms) >= 2:
            year = _dt.datetime.now().year
            return f"ALS {' '.join(bio_terms[:5])} {year}"

    # Fallback to static queries
    return get_layer_query(layer, step)


def _get_drug_centric_query(state: ResearchState, step: int) -> str:
    """Generate a drug-name-based PubMed query — a new dimension beyond gene-centric.

    Rotates through all drugs from interventions.json (drug/small_molecule/peptide)
    with ALS-specific query templates.
    """
    drugs = _get_pharmacogenomics_drugs()
    if not drugs:
        return f"ALS treatment {_dt.datetime.now().year}"
    drug = drugs[step % len(drugs)]
    year = _dt.datetime.now().year
    templates = [
        f"{drug} ALS motor neuron mechanism {year}",
        f"{drug} ALS clinical trial outcome {year}",
        f"{drug} neuroprotection combination therapy ALS {year}",
        f"{drug} pharmacokinetics blood brain barrier ALS {year}",
    ]
    return templates[(step // max(len(drugs), 1)) % len(templates)]


def _get_expanded_query(state: ResearchState, step: int, layer: str) -> str:
    """Generate a novel PubMed query by expanding exhausted targets via KG neighbors.

    Queries the KG for genes related to the current rotation target, then builds
    a query using neighbor gene names and ALS context. Falls back to targeted
    query if no KG neighbors are available or expansion is disabled.
    """
    try:
        from config.loader import ConfigLoader
        cfg = ConfigLoader()
        if not cfg.get("query_expansion_enabled", True):
            return _get_targeted_query(state, step)
    except Exception:
        return _get_targeted_query(state, step)

    from targets.als_targets import ALS_TARGETS
    from research.query_expansion import get_gene_neighbors_galen, get_gene_neighbors, get_expanded_queries

    targets = list(ALS_TARGETS.values())
    if not targets:
        return _get_targeted_query(state, step)
    target = targets[step % len(targets)]
    gene = target.get("gene", "")

    neighbors = get_gene_neighbors_galen(gene, max_neighbors=5, min_confidence=0.4)
    if not neighbors:
        neighbors = get_gene_neighbors(gene, max_neighbors=5, min_confidence=0.4)
    if not neighbors:
        return _get_targeted_query(state, step)

    queries = get_expanded_queries(gene, neighbors, state)
    if queries:
        return queries[0]

    return _get_targeted_query(state, step)


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
    ActionType.RUN_COMPUTATION,
    ActionType.QUERY_ALSOD,
    ActionType.QUERY_GTEX,
    ActionType.QUERY_CLINVAR,
    ActionType.QUERY_GWAS,
    ActionType.QUERY_BINDINGDB,
    ActionType.QUERY_HPA,
    ActionType.QUERY_DRUGBANK,
    ActionType.QUERY_ALPHAFOLD,
    ActionType.QUERY_REACTOME_LOCAL,
    # Phase 10: expanded evidence sources
    ActionType.QUERY_GNOMAD,
    ActionType.QUERY_UNIPROT,
    ActionType.QUERY_SPLICEAI,
    ActionType.QUERY_FAERS,
    ActionType.QUERY_DISGENET,
    ActionType.QUERY_GEO_ALS,
    ActionType.QUERY_CMAP,
]

# Depth-biased 5-step cycle: 2 acquisition, 1 reasoning, 1 validation, 1 deepening
_CYCLE_LENGTH = 5
_ACQUIRE_STEPS = {0, 4}      # 2 of 5 steps are acquisition (was 3)
_REASON_STEP = 1              # 1 of 5 is reasoning (hypothesis gen)
_VALIDATE_STEP = 2            # 1 of 5 is validation
_DEEPEN_STEP = 3              # 1 of 5 is dedicated causal chain deepening or computation

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


_MAX_CONSECUTIVE_SAME_ACTION = 3


def _action_is_feasible(action: ActionType, state: ResearchState, target_depth: int) -> bool:
    """Check if an action can actually execute without falling back.

    Actions that would silently convert to generate_hypothesis are
    not feasible — they bypass the consecutive cap and waste cycles.
    """
    if action == ActionType.DEEPEN_CAUSAL_CHAIN:
        shallow = [(k, v) for k, v in state.causal_chains.items() if v < target_depth]
        return len(shallow) > 0
    if action == ActionType.GENERATE_HYPOTHESIS:
        # Feasible unless active hypotheses are saturated (max_active reached)
        try:
            from config.loader import ConfigLoader
            max_active = ConfigLoader().get("research_hypothesis_max_active", 10)
        except Exception:
            max_active = 10
        return len(state.active_hypotheses) < max_active
    if action == ActionType.VALIDATE_HYPOTHESIS:
        return len(state.active_hypotheses) > 0
    return True  # Acquisition actions are always feasible


def select_action_thompson(
    state: ResearchState,
    regen_threshold: int = 10,
    target_depth: int = 5,
) -> tuple[ActionType, dict[str, Any]]:
    """Select action via Thompson sampling over Beta posteriors."""
    # Preempt: protocol regeneration (only if evidence is genuinely new)
    if state.new_evidence_since_regen >= regen_threshold and state.protocol_version >= 1:
        return ActionType.REGENERATE_PROTOCOL, build_action_params(ActionType.REGENERATE_PROTOCOL)

    # Expire stale hypotheses (same as cycle path — prevents deadlock)
    _maybe_expire_hypotheses(state)

    # Build the full candidate action type list, filtering infeasible actions
    all_types: list[ActionType] = list(set(_ACQUISITION_ROTATION))
    all_types.extend([ActionType.GENERATE_HYPOTHESIS, ActionType.DEEPEN_CAUSAL_CHAIN])
    if state.active_hypotheses:
        all_types.append(ActionType.VALIDATE_HYPOTHESIS)
    if hasattr(ActionType, "CHALLENGE_INTERVENTION"):
        all_types.append(ActionType.CHALLENGE_INTERVENTION)
    # Computational + drug design actions
    all_types.append(ActionType.RUN_COMPUTATION)
    if hasattr(ActionType, "DESIGN_MOLECULE"):
        all_types.append(ActionType.DESIGN_MOLECULE)

    # Remove infeasible actions (prevents silent fallback to generate_hypothesis)
    all_types = [at for at in all_types if _action_is_feasible(at, state, target_depth)]

    # --- Consecutive-action cap: exclude the ACTUAL last action if repeated too many times ---
    excluded_action: str | None = None
    if state.consecutive_same_action >= _MAX_CONSECUTIVE_SAME_ACTION and state.last_action:
        excluded_action = state.last_action

    candidates = [at for at in all_types if at.value != excluded_action]
    if not candidates:
        candidates = all_types if all_types else list(set(_ACQUISITION_ROTATION))

    # Diversity floor: force any action type not used in 30 steps
    for at in candidates:
        last_used = state.last_action_per_type.get(at.value, 0)
        if state.step_count - last_used >= 30:
            return _build_thompson_params(at, state, target_depth)

    # Thompson sampling: draw from each Beta and pick the argmax
    posteriors = state.action_posteriors or {}
    best_action: ActionType | None = None
    best_sample = -1.0
    for at in candidates:
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
    """Build params for Thompson-selected action using existing helpers.

    IMPORTANT: Never silently convert one action type to another.
    If an action can't execute, fall back to an acquisition action
    (not generate_hypothesis) to maintain action diversity.
    """
    if action == ActionType.GENERATE_HYPOTHESIS:
        return ActionType.GENERATE_HYPOTHESIS, build_action_params(ActionType.GENERATE_HYPOTHESIS)
    elif action == ActionType.VALIDATE_HYPOTHESIS:
        if state.active_hypotheses:
            hyp_id = state.active_hypotheses[0]
            return action, build_action_params(action, hypothesis_id=hyp_id)
        # No hypotheses to validate — fall back to acquisition
        return _build_acquisition_params(
            _ACQUISITION_ROTATION[state.step_count % len(_ACQUISITION_ROTATION)],
            state, state.step_count,
        )
    elif action == ActionType.DEEPEN_CAUSAL_CHAIN:
        shallow = [(k, v) for k, v in state.causal_chains.items() if v < target_depth]
        if shallow:
            int_id = shallow[state.step_count % len(shallow)][0]
            return action, build_action_params(action, intervention_id=int_id)
        # All chains full — fall back to acquisition (NOT generate_hypothesis)
        return _build_acquisition_params(
            _ACQUISITION_ROTATION[state.step_count % len(_ACQUISITION_ROTATION)],
            state, state.step_count,
        )
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
        drugs = _get_pharmacogenomics_drugs()
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

    # 1. Acquisition steps (positions 0, 4 in the cycle)
    if cycle_pos in _ACQUIRE_STEPS:
        return _select_acquisition_action(state)

    # 2. Reasoning step (position 1): hypothesis generation
    if cycle_pos == _REASON_STEP:
        if state.top_uncertainties or state.protocol_version > 0:
            return ActionType.GENERATE_HYPOTHESIS, build_action_params(
                ActionType.GENERATE_HYPOTHESIS,
            )
        return _select_acquisition_action(state)

    # 3. Validation step (position 2): validate if pending, otherwise acquire
    if cycle_pos == _VALIDATE_STEP:
        consecutive_validations = _count_consecutive_validations(state)
        if state.active_hypotheses and consecutive_validations < _MAX_CONSECUTIVE_VALIDATIONS:
            hyp_id = state.active_hypotheses[0]
            return ActionType.VALIDATE_HYPOTHESIS, build_action_params(
                ActionType.VALIDATE_HYPOTHESIS, hypothesis_id=hyp_id,
            )
        return _select_acquisition_action(state)

    # 4. Deepening step (position 3): causal chain deepening or computation
    if cycle_pos == _DEEPEN_STEP:
        return _select_deepening_action(state, target_depth)

    # Fallback
    return _select_acquisition_action(state)


def _select_deepening_action(
    state: ResearchState,
    target_depth: int,
) -> tuple[ActionType, dict[str, Any]]:
    """Dedicated step for causal chain deepening or computation.

    Prioritizes deepening shallow causal chains (70% probability), with
    fallback to computational experiments or hypothesis generation.
    """
    import random
    step = state.step_count

    # Prioritize chain deepening when chains are shallow
    shallow = [(k, v) for k, v in state.causal_chains.items() if v < target_depth]
    if shallow and random.random() < 0.7:
        # Sort by depth (shallowest first) to prioritize biggest gaps
        shallow.sort(key=lambda x: x[1])
        idx = step % len(shallow)
        int_id, _ = shallow[idx]
        return ActionType.DEEPEN_CAUSAL_CHAIN, build_action_params(
            ActionType.DEEPEN_CAUSAL_CHAIN, intervention_id=int_id,
        )

    # Alternate between computation and molecule design
    try:
        from config.loader import ConfigLoader
        cfg = ConfigLoader()
        if cfg.get("molecular_computation_enabled", False) and step % 2 == 0:
            return ActionType.DESIGN_MOLECULE, build_action_params(
                ActionType.DESIGN_MOLECULE,
            )
        elif cfg.get("computation_enabled", True):
            return ActionType.RUN_COMPUTATION, build_action_params(
                ActionType.RUN_COMPUTATION,
            )
    except Exception:
        pass

    # Fallback: generate hypothesis
    if state.top_uncertainties or state.protocol_version > 0:
        return ActionType.GENERATE_HYPOTHESIS, build_action_params(
            ActionType.GENERATE_HYPOTHESIS,
        )

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


def _maybe_expand_gene(
    gene: str,
    action: ActionType,
    state: ResearchState,
) -> str:
    """Check if a gene is exhausted for this action and expand if needed."""
    try:
        from config.loader import ConfigLoader
        _cfg = ConfigLoader()
        if not _cfg.get("query_expansion_enabled", True):
            return gene
        threshold = _cfg.get("query_expansion_exhaustion_threshold", 3)
        max_neighbors = _cfg.get("query_expansion_max_neighbors", 10)
        min_confidence = _cfg.get("query_expansion_min_confidence", 0.4)
    except Exception:
        return gene

    from research.query_expansion import should_expand, get_expanded_gene
    exhaustion_key = f"{gene}:{action.value}"
    if should_expand(exhaustion_key, state, threshold=threshold):
        expanded = get_expanded_gene(
            gene, action.value, state,
            max_neighbors=max_neighbors, min_confidence=min_confidence,
        )
        if expanded != gene:
            print(f"[RESEARCH] EXPANSION: {exhaustion_key} exhausted, expanding to {expanded}")
            return expanded
    return gene


def _build_acquisition_params(
    action: ActionType,
    state: ResearchState,
    step: int,
) -> tuple[ActionType, dict[str, Any]]:
    """Build params for a specific acquisition action."""
    if action == ActionType.SEARCH_PUBMED:
        layer_idx = (step // _CYCLE_LENGTH) % len(ALL_LAYERS)
        protocol_layer = ALL_LAYERS[layer_idx]

        # Layer-aware query selection: use research layer to pick appropriate queries
        from research.layer_orchestrator import ResearchLayer, get_layer_queries
        try:
            research_layer = ResearchLayer(state.research_layer)
        except (ValueError, AttributeError):
            research_layer = ResearchLayer.ALS_MECHANISMS

        layer_queries = get_layer_queries(
            research_layer,
            genetic_profile=getattr(state, "genetic_profile", None),
            validated_targets=[
                k for k, v in state.causal_chains.items() if v >= 3
            ] if research_layer == ResearchLayer.DRUG_DESIGN else None,
        )

        if layer_queries:
            # Rotate through layer-specific queries, with occasional dynamic/expanded
            strategy = step % 4
            if strategy <= 1 and layer_queries:
                # Primary: layer-specific query bank
                query = layer_queries[step % len(layer_queries)]
                year = __import__("datetime").datetime.now().year
                query = f"{query} {year}"
            elif strategy == 2:
                query = _get_dynamic_query(state, step, protocol_layer)
            else:
                query = _get_expanded_query(state, step, protocol_layer)
        else:
            # Fallback to existing 5-strategy cycling
            strategy = step % 5
            if strategy == 0:
                query = get_layer_query(protocol_layer, step)
            elif strategy == 1:
                query = _get_dynamic_query(state, step, protocol_layer)
            elif strategy == 2:
                query = _get_targeted_query(state, step)
            elif strategy == 3:
                query = _get_expanded_query(state, step, protocol_layer)
            else:
                query = _get_drug_centric_query(state, step)
        return action, build_action_params(action, query=query, protocol_layer=protocol_layer)

    elif action == ActionType.SEARCH_TRIALS:
        # Rotate through protocol interventions and layers
        interventions = list(state.causal_chains.keys())
        if interventions:
            int_name = interventions[step % len(interventions)].replace("int:", "")
            layer = ALL_LAYERS[step % len(ALL_LAYERS)]
            return action, build_action_params(action, query=f"ALS {int_name}", protocol_layer=layer)
        return action, build_action_params(action, protocol_layer=ALL_LAYERS[step % len(ALL_LAYERS)])

    elif action == ActionType.QUERY_PATHWAYS:
        from targets.als_targets import ALS_TARGETS
        target_keys = list(ALS_TARGETS.keys())
        if target_keys:
            target_idx = step % len(target_keys)
            target_key = target_keys[target_idx]
            target = ALS_TARGETS[target_key]
            gene = target.get("gene", "")
            gene = _maybe_expand_gene(gene, action, state)
            layers = target.get("protocol_layers", ["root_cause_suppression"])
            return action, build_action_params(
                action,
                target_name=target_key,
                uniprot_id=target.get("uniprot_id", ""),
                gene_symbol=gene,
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
            gene = _maybe_expand_gene(gene, action, state)
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
        drugs = _get_pharmacogenomics_drugs()
        drug_idx = step % len(drugs)
        return action, build_action_params(action, drug_name=drugs[drug_idx], protocol_layer="adaptive_maintenance")

    elif action == ActionType.QUERY_GALEN_KG:
        from connectors.galen_kg import ALS_CROSS_REFERENCE_GENES
        gene_idx = step % len(ALS_CROSS_REFERENCE_GENES)
        gene = ALS_CROSS_REFERENCE_GENES[gene_idx]
        gene = _maybe_expand_gene(gene, action, state)
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

        # Layer-aware query selection (same as SEARCH_PUBMED)
        from research.layer_orchestrator import ResearchLayer, get_layer_queries
        try:
            research_layer = ResearchLayer(state.research_layer)
        except (ValueError, AttributeError):
            research_layer = ResearchLayer.ALS_MECHANISMS

        layer_queries = get_layer_queries(
            research_layer,
            genetic_profile=getattr(state, "genetic_profile", None),
        )
        if layer_queries:
            query = layer_queries[step % len(layer_queries)]
            year = __import__("datetime").datetime.now().year
            query = f"{query} {year}"
        else:
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
        gene = _maybe_expand_gene(gene, action, state)
        return action, build_action_params(action, target_gene=gene, protocol_layer="root_cause_suppression")

    elif action == ActionType.RUN_COMPUTATION:
        return action, build_action_params(action, protocol_layer="root_cause_suppression")

    elif action == ActionType.QUERY_ALSOD:
        from connectors.alsod import ERIK_PRIORITY_GENES
        gene = ERIK_PRIORITY_GENES[step % len(ERIK_PRIORITY_GENES)]
        gene = _maybe_expand_gene(gene, action, state)
        return action, build_action_params(action, gene=gene, protocol_layer="root_cause_suppression")

    elif action in (ActionType.QUERY_GTEX, ActionType.QUERY_CLINVAR, ActionType.QUERY_GWAS,
                    ActionType.QUERY_HPA, ActionType.QUERY_DRUGBANK, ActionType.QUERY_ALPHAFOLD,
                    ActionType.QUERY_REACTOME_LOCAL,
                    ActionType.QUERY_GNOMAD, ActionType.QUERY_UNIPROT, ActionType.QUERY_SPLICEAI,
                    ActionType.QUERY_DISGENET, ActionType.QUERY_GEO_ALS):
        # All gene-rotating connectors: pick ALS target by step, expand if exhausted
        from targets.als_targets import ALS_TARGETS
        target_keys = list(ALS_TARGETS.keys())
        if target_keys:
            target = ALS_TARGETS[target_keys[step % len(target_keys)]]
            gene = target.get("gene", "TARDBP")
            gene = _maybe_expand_gene(gene, action, state)
            return action, build_action_params(action, gene=gene, protocol_layer="root_cause_suppression")
        return action, build_action_params(action, gene="TARDBP", protocol_layer="root_cause_suppression")

    elif action == ActionType.QUERY_BINDINGDB:
        # BindingDB needs drug + gene pair
        interventions = list(state.causal_chains.keys())
        from targets.als_targets import ALS_TARGETS
        target_keys = list(ALS_TARGETS.keys())
        drug = interventions[step % len(interventions)].replace("int:", "") if interventions else "riluzole"
        gene = ALS_TARGETS[target_keys[step % len(target_keys)]].get("gene", "TARDBP") if target_keys else "TARDBP"
        gene = _maybe_expand_gene(gene, action, state)
        return action, build_action_params(action, drug=drug, gene=gene, protocol_layer="root_cause_suppression")

    elif action == ActionType.QUERY_FAERS:
        drugs = _get_pharmacogenomics_drugs()
        drug = drugs[step % len(drugs)] if drugs else "riluzole"
        return action, build_action_params(action, drug_name=drug, protocol_layer="adaptive_maintenance")

    elif action == ActionType.QUERY_CMAP:
        interventions = list(state.causal_chains.keys())
        if interventions and step % 2 == 0:
            drug = interventions[step % len(interventions)].replace("int:", "")
            return action, build_action_params(action, compound=drug, protocol_layer="pathology_reversal")
        return action, build_action_params(action, protocol_layer="pathology_reversal")

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
    """Remove stale hypotheses. Two mechanisms:

    1. At max_active: force-expire oldest to prevent deadlock
    2. Validation-ratio: original logic for over-validated hypotheses

    Modifies state.active_hypotheses in place (mutable list).
    """
    if not state.active_hypotheses:
        return
    try:
        from config.loader import ConfigLoader
        max_active = ConfigLoader().get("research_hypothesis_max_active", 10)
    except Exception:
        max_active = 10
    # Force-expire when at capacity (breaks deadlock)
    if len(state.active_hypotheses) >= max_active:
        state.active_hypotheses.pop(0)
        state.resolved_hypotheses += 1
        return
    # Original validation-ratio expiry
    val_count = state.action_counts.get("validate_hypothesis", 0)
    hyp_count = len(state.active_hypotheses)
    if hyp_count > 0 and val_count > 0:
        avg_validations = val_count / max(hyp_count + state.resolved_hypotheses, 1)
        if avg_validations > _MAX_VALIDATION_ATTEMPTS_PER_HYP:
            if state.active_hypotheses:
                state.active_hypotheses.pop(0)
                state.resolved_hypotheses += 1
