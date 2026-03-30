"""Research action types, results, and parameter builders."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

class ActionType(str, Enum):
    SEARCH_PUBMED = "search_pubmed"
    SEARCH_TRIALS = "search_trials"
    QUERY_CHEMBL = "query_chembl"
    QUERY_OPENTARGETS = "query_opentargets"
    CHECK_INTERACTIONS = "check_interactions"
    GENERATE_HYPOTHESIS = "generate_hypothesis"
    DEEPEN_CAUSAL_CHAIN = "deepen_causal_chain"
    VALIDATE_HYPOTHESIS = "validate_hypothesis"
    SCORE_NEW_EVIDENCE = "score_new_evidence"
    REGENERATE_PROTOCOL = "regenerate_protocol"
    # Phase 3B: Evidence expansion
    QUERY_PATHWAYS = "query_pathways"
    QUERY_PPI_NETWORK = "query_ppi_network"
    MATCH_COHORT = "match_cohort"
    INTERPRET_VARIANT = "interpret_variant"
    CHECK_PHARMACOGENOMICS = "check_pharmacogenomics"
    QUERY_GALEN_KG = "query_galen_kg"
    UPDATE_TRAJECTORY = "update_trajectory"
    SEARCH_PREPRINTS = "search_preprints"
    CHALLENGE_INTERVENTION = "challenge_intervention"
    QUERY_GALEN_SCM = "query_galen_scm"
    RUN_COMPUTATION = "run_computation"

NETWORK_ACTIONS = {
    ActionType.SEARCH_PUBMED,
    ActionType.SEARCH_TRIALS,
    ActionType.QUERY_OPENTARGETS,
    ActionType.QUERY_PATHWAYS,
    ActionType.QUERY_PPI_NETWORK,
    ActionType.INTERPRET_VARIANT,
    ActionType.CHECK_PHARMACOGENOMICS,
    ActionType.QUERY_GALEN_KG,
    ActionType.SEARCH_PREPRINTS,
    ActionType.QUERY_GALEN_SCM,
}

LLM_ACTIONS = {
    ActionType.GENERATE_HYPOTHESIS,
    ActionType.DEEPEN_CAUSAL_CHAIN,
    ActionType.VALIDATE_HYPOTHESIS,
    ActionType.SCORE_NEW_EVIDENCE,
    ActionType.REGENERATE_PROTOCOL,
}

@dataclass
class ActionResult:
    action: ActionType
    success: bool = True
    error: Optional[str] = None
    evidence_items_added: int = 0
    interventions_added: int = 0
    hypothesis_generated: Optional[str] = None
    hypothesis_resolved: bool = False
    causal_depth_added: int = 0
    protocol_regenerated: bool = False
    protocol_score_delta: float = 0.0
    protocol_stable: bool = False
    interaction_safe: bool = False
    eligibility_confirmed: bool = False
    protocol_layer: Optional[str] = None
    evidence_strength: Optional[str] = None
    detail: dict[str, Any] = field(default_factory=dict)

def build_action_params(action: ActionType, **kwargs: Any) -> dict[str, Any]:
    params: dict[str, Any] = {"action": action}
    params.update(kwargs)
    if action == ActionType.SEARCH_PUBMED:
        params.setdefault("max_results", 20)
    elif action == ActionType.SEARCH_TRIALS:
        params.setdefault("max_results", 30)
    elif action == ActionType.GENERATE_HYPOTHESIS:
        params.setdefault("topic", "")
        params.setdefault("uncertainty", "")
    elif action == ActionType.DEEPEN_CAUSAL_CHAIN:
        params.setdefault("intervention_id", "")
    elif action == ActionType.VALIDATE_HYPOTHESIS:
        params.setdefault("hypothesis_id", "")
    return params
