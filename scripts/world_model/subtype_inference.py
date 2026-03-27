"""Stage 2 — Subtype inference for the Erik ALS pipeline.

Infers the posterior probability distribution over ALS subtypes for a patient
using evidence-grounded LLM reasoning via ReasoningEngine.

Public API
----------
infer_subtype(patient_state_json, evidence_items, subject_ref, reasoning_engine)
    -> EtiologicDriverProfile

_parse_subtype_response(response, subject_ref)
    -> EtiologicDriverProfile  (pure helper, exposed for testing)
"""
from __future__ import annotations

from typing import Optional

from ontology.enums import SubtypeClass
from ontology.interpretation import EtiologicDriverProfile
from world_model.prompts.templates import SUBTYPE_TEMPLATE
from world_model.reasoning_engine import ReasoningEngine


def _parse_subtype_response(
    response: dict,
    subject_ref: str,
) -> EtiologicDriverProfile:
    """Parse a raw LLM response dict into an EtiologicDriverProfile.

    Parameters
    ----------
    response:
        Dict produced by ``ReasoningEngine.reason()``.  Must contain a
        ``"posterior"`` key whose value is a dict of string→float, and a
        ``"cited_evidence"`` key.
    subject_ref:
        Identifier of the patient / subject this profile belongs to.

    Returns
    -------
    EtiologicDriverProfile
        Posterior over :class:`~ontology.enums.SubtypeClass` enum values,
        normalised to sum to 1.0.  Invalid string keys are silently skipped.
    """
    raw_posterior: dict = response.get("posterior") or {}

    # Map string keys to SubtypeClass enum values, skipping invalid keys
    valid_values = {member.value: member for member in SubtypeClass}
    mapped: dict[SubtypeClass, float] = {}
    for key, prob in raw_posterior.items():
        if key in valid_values:
            mapped[valid_values[key]] = float(prob)
        # else: skip invalid / unrecognised key

    # Normalise to sum exactly 1.0
    total = sum(mapped.values())
    if total > 0.0:
        mapped = {k: v / total for k, v in mapped.items()}

    # Build the body with reasoning/conditional_on_genetics
    body: dict = {}
    if "conditional_on_genetics" in response:
        body["conditional_on_genetics"] = response["conditional_on_genetics"]
    if "reasoning" in response:
        body["reasoning"] = response["reasoning"]

    return EtiologicDriverProfile(
        id=f"driver:{subject_ref}",
        subject_ref=subject_ref,
        posterior=mapped,
        supporting_evidence_refs=list(response.get("cited_evidence") or []),
        body=body,
    )


def infer_subtype(
    patient_state_json: str,
    evidence_items: list[dict],
    subject_ref: str,
    reasoning_engine: Optional[ReasoningEngine] = None,
) -> EtiologicDriverProfile:
    """Infer the ALS subtype posterior for a patient.

    Parameters
    ----------
    patient_state_json:
        JSON-serialised patient state (passed into the LLM prompt).
    evidence_items:
        List of evidence item dicts to ground the LLM reasoning.
    subject_ref:
        Patient / subject identifier used to construct the profile ``id``.
    reasoning_engine:
        Pre-configured :class:`~world_model.reasoning_engine.ReasoningEngine`.
        If ``None``, a lazy engine is created automatically.

    Returns
    -------
    EtiologicDriverProfile
        Parsed posterior if the LLM succeeds, or a uniform prior over all 8
        :class:`~ontology.enums.SubtypeClass` values (abstention) if the LLM
        returns ``None``.
    """
    if reasoning_engine is None:
        reasoning_engine = ReasoningEngine(lazy=True)

    response = reasoning_engine.reason(
        SUBTYPE_TEMPLATE,
        evidence_items,
        extra_context={"patient_state_json": patient_state_json},
    )

    if response is None:
        # Abstention: uniform prior over all subtypes
        all_subtypes = list(SubtypeClass)
        uniform_prob = 1.0 / len(all_subtypes)
        return EtiologicDriverProfile(
            id=f"driver:{subject_ref}",
            subject_ref=subject_ref,
            posterior={st: uniform_prob for st in all_subtypes},
            supporting_evidence_refs=[],
            body={"conditional_on_genetics": "", "reasoning": "abstention: no LLM response"},
        )

    return _parse_subtype_response(response, subject_ref)
