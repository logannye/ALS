"""Stage 3 — Intervention scoring for the Erik ALS pipeline.

Scores each candidate intervention for relevance and predicted efficacy given
the patient state, subtype posterior, and supporting evidence.

Public API
----------
score_intervention(intervention, evidence_items, patient_state_json,
                   subtype_posterior_json, reasoning_engine, verify_critical)
    -> Optional[InterventionScore]

score_all_interventions(interventions, evidence_store, patient_state_json,
                        subtype_posterior_json, reasoning_engine)
    -> list[InterventionScore]  (sorted descending by relevance_score)

_parse_score_response(response)
    -> InterventionScore  (pure helper, exposed for testing)
"""
from __future__ import annotations

import json
from typing import Optional, Union

from pydantic import BaseModel, Field

from world_model.prompts.templates import INTERVENTION_SCORING_TEMPLATE
from world_model.reasoning_engine import ReasoningEngine


# ---------------------------------------------------------------------------
# InterventionScore — transient artifact (NOT a BaseEnvelope)
# ---------------------------------------------------------------------------

class InterventionScore(BaseModel):
    """Transient scoring result for a single intervention.

    This is NOT persisted as a canonical object.  It is produced in-memory
    during a scoring pass and consumed by the protocol assembler (Stage 4).
    """

    intervention_id: str
    intervention_name: str
    protocol_layer: str = ""
    relevance_score: float = Field(ge=0.0, le=1.0)
    mechanism_argument: str = ""
    evidence_strength: str = "unknown"
    erik_eligible: Union[bool, str] = True
    key_uncertainties: list[str]
    cited_evidence: list[str]
    contested_claims: list[str]


# ---------------------------------------------------------------------------
# _parse_score_response
# ---------------------------------------------------------------------------

def _parse_score_response(response: dict) -> InterventionScore:
    """Parse and validate a raw LLM scoring response into an InterventionScore.

    Parameters
    ----------
    response:
        Dict produced by ``ReasoningEngine.reason()``.

    Returns
    -------
    InterventionScore
        ``relevance_score`` is clamped to ``[0.0, 1.0]``.
    """
    raw_score = response.get("relevance_score", 0.0)
    try:
        clamped_score = max(0.0, min(1.0, float(raw_score)))
    except (TypeError, ValueError):
        clamped_score = 0.0

    return InterventionScore(
        intervention_id=response.get("intervention_id", ""),
        intervention_name=response.get("intervention_name", ""),
        protocol_layer=response.get("protocol_layer", ""),
        relevance_score=clamped_score,
        mechanism_argument=response.get("mechanism_argument", ""),
        evidence_strength=response.get("evidence_strength", "unknown"),
        erik_eligible=response.get("erik_eligible", True),
        key_uncertainties=list(response.get("key_uncertainties") or []),
        cited_evidence=list(response.get("cited_evidence") or []),
        contested_claims=list(response.get("contested_claims") or []),
    )


# ---------------------------------------------------------------------------
# score_intervention
# ---------------------------------------------------------------------------

def score_intervention(
    intervention: dict,
    evidence_items: list[dict],
    patient_state_json: str,
    subtype_posterior_json: str,
    reasoning_engine: ReasoningEngine,
    verify_critical: bool = False,
) -> Optional[InterventionScore]:
    """Score a single intervention against the patient state and evidence.

    Parameters
    ----------
    intervention:
        Dict representation of an Intervention object (must contain at least
        ``"id"`` and ``"name"``).
    evidence_items:
        Evidence items relevant to this intervention.
    patient_state_json:
        JSON-serialised patient state string.
    subtype_posterior_json:
        JSON-serialised subtype posterior string.
    reasoning_engine:
        Configured :class:`~world_model.reasoning_engine.ReasoningEngine`.
    verify_critical:
        When ``True``, run a second verification pass on ``mechanism_argument``.

    Returns
    -------
    Optional[InterventionScore]
        Parsed score, or ``None`` if the LLM fails to produce valid JSON.
    """
    intervention_json = json.dumps(intervention, indent=2)

    response = reasoning_engine.reason(
        INTERVENTION_SCORING_TEMPLATE,
        evidence_items,
        extra_context={
            "patient_state_json": patient_state_json,
            "subtype_posterior_json": subtype_posterior_json,
            "intervention_json": intervention_json,
        },
        verify_critical=verify_critical,
    )

    if response is None:
        return None

    return _parse_score_response(response)


# ---------------------------------------------------------------------------
# score_all_interventions
# ---------------------------------------------------------------------------

def score_all_interventions(
    interventions: list[dict],
    evidence_store,
    patient_state_json: str,
    subtype_posterior_json: str,
    reasoning_engine: Optional[ReasoningEngine] = None,
) -> list[InterventionScore]:
    """Score all candidate interventions and return them sorted by relevance.

    For each intervention:
    1. Gather evidence via ``query_by_intervention_ref`` + ``query_by_mechanism_target``.
    2. Deduplicate evidence by ``id``.
    3. Score via :func:`score_intervention`.

    After scoring, verify the ``mechanism_argument`` of the top 5 results via
    ``reasoning_engine._verify_claim``.

    Parameters
    ----------
    interventions:
        List of intervention dicts (each must contain ``"id"`` and ``"name"``).
    evidence_store:
        An :class:`~evidence.evidence_store.EvidenceStore` instance.
    patient_state_json:
        JSON-serialised patient state.
    subtype_posterior_json:
        JSON-serialised subtype posterior.
    reasoning_engine:
        Optional :class:`~world_model.reasoning_engine.ReasoningEngine`.
        If ``None``, a lazy engine is created automatically.

    Returns
    -------
    list[InterventionScore]
        Scored interventions, sorted descending by ``relevance_score``.
        Interventions where the LLM fails are silently excluded.
    """
    if reasoning_engine is None:
        reasoning_engine = ReasoningEngine(lazy=True)

    scores: list[InterventionScore] = []

    for intervention in interventions:
        int_id: str = intervention.get("id", "")
        targets: list[str] = intervention.get("targets", [])

        # Gather evidence: by intervention ref + by mechanism targets
        evidence_by_ref: list[dict] = evidence_store.query_by_intervention_ref(int_id)
        evidence_by_target: list[dict] = []
        for target in targets:
            evidence_by_target.extend(evidence_store.query_by_mechanism_target(target))

        # Deduplicate by ID
        seen_ids: set[str] = set()
        deduped_evidence: list[dict] = []
        for item in evidence_by_ref + evidence_by_target:
            item_id = item.get("id")
            if item_id and item_id not in seen_ids:
                seen_ids.add(item_id)
                deduped_evidence.append(item)

        result = score_intervention(
            intervention=intervention,
            evidence_items=deduped_evidence,
            patient_state_json=patient_state_json,
            subtype_posterior_json=subtype_posterior_json,
            reasoning_engine=reasoning_engine,
        )

        if result is not None:
            scores.append(result)

    # Sort descending by relevance_score
    scores.sort(key=lambda s: s.relevance_score, reverse=True)

    # Verify top 5 mechanism arguments
    for top_score in scores[:5]:
        if top_score.mechanism_argument:
            reasoning_engine._verify_claim(
                top_score.mechanism_argument,
                top_score.cited_evidence,
            )

    return scores
