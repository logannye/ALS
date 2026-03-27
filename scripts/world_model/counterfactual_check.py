"""Stage 5 — Counterfactual verification of protocol layers.

For each layer in the assembled protocol, asks: "What if this layer were
removed?"  Identifies load-bearing layers, weakest evidence links, and the
single most valuable missing measurement.

Public API
----------
check_counterfactual(protocol_json, layer_name, layer_interventions,
                     evidence_items, reasoning_engine)
    -> Optional[CounterfactualResult]

run_counterfactual_analysis(protocol, evidence_store, reasoning_engine)
    -> list[CounterfactualResult]
"""
from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel, Field

from world_model.reasoning_engine import ReasoningEngine
from world_model.prompts.templates import COUNTERFACTUAL_TEMPLATE


# ---------------------------------------------------------------------------
# CounterfactualResult (transient artifact — NOT a BaseEnvelope)
# ---------------------------------------------------------------------------

class CounterfactualResult(BaseModel):
    """Result of removing one layer from the protocol."""

    layer: str
    removal_impact: str = "uncertain"  # critical, significant, moderate, minimal, uncertain
    reasoning: str = ""
    is_load_bearing: bool = False
    weakest_evidence: str = ""
    next_best_measurement: str = ""
    cited_evidence: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# check_counterfactual
# ---------------------------------------------------------------------------

def check_counterfactual(
    protocol_json: str,
    layer_name: str,
    layer_interventions: str,
    evidence_items: list[dict],
    reasoning_engine: ReasoningEngine,
) -> Optional[CounterfactualResult]:
    """Run counterfactual analysis for one protocol layer.

    Parameters
    ----------
    protocol_json:
        JSON-serialised full protocol (for context).
    layer_name:
        Name of the layer being assessed (e.g. ``"root_cause_suppression"``).
    layer_interventions:
        Comma-separated intervention IDs in this layer.
    evidence_items:
        Evidence items supporting this layer's interventions.
    reasoning_engine:
        Configured :class:`ReasoningEngine`.

    Returns
    -------
    Optional[CounterfactualResult]
        Parsed result, or a fallback ``uncertain`` result if LLM fails.
    """
    response = reasoning_engine.reason(
        template=COUNTERFACTUAL_TEMPLATE,
        evidence_items=evidence_items,
        extra_context={
            "protocol_json": protocol_json,
            "layer_name": layer_name,
            "layer_interventions": layer_interventions,
        },
        max_tokens=1000,
    )

    if response is None:
        return CounterfactualResult(
            layer=layer_name,
            removal_impact="uncertain",
            reasoning="LLM returned no response",
        )

    return CounterfactualResult(
        layer=response.get("layer", layer_name),
        removal_impact=response.get("removal_impact", "uncertain"),
        reasoning=response.get("reasoning", ""),
        is_load_bearing=response.get("is_load_bearing", False),
        weakest_evidence=response.get("weakest_evidence", ""),
        next_best_measurement=response.get("next_best_measurement", ""),
        cited_evidence=response.get("cited_evidence", []),
    )


# ---------------------------------------------------------------------------
# run_counterfactual_analysis
# ---------------------------------------------------------------------------

def run_counterfactual_analysis(
    protocol,
    evidence_store,
    reasoning_engine: ReasoningEngine,
) -> list[CounterfactualResult]:
    """Run counterfactual analysis on all layers of a protocol.

    Abstained layers (those with ``ABSTENTION`` in their notes) are
    automatically assigned ``removal_impact="low"`` without an LLM call.

    Parameters
    ----------
    protocol:
        A :class:`~ontology.protocol.CureProtocolCandidate`.
    evidence_store:
        An :class:`~evidence.evidence_store.EvidenceStore` instance.
    reasoning_engine:
        Configured :class:`ReasoningEngine`.

    Returns
    -------
    list[CounterfactualResult]
        One result per protocol layer.
    """
    results: list[CounterfactualResult] = []
    protocol_json = json.dumps(
        protocol.model_dump(mode="json"), indent=2, default=str,
    )

    for layer_entry in protocol.layers:
        if "ABSTENTION" in layer_entry.notes:
            results.append(CounterfactualResult(
                layer=layer_entry.layer.value,
                removal_impact="low",
                reasoning="Layer already abstained — no intervention to remove.",
                is_load_bearing=False,
            ))
            continue

        # Gather evidence for this layer's interventions
        evidence: list[dict] = []
        for int_ref in layer_entry.intervention_refs:
            evidence.extend(evidence_store.query_by_intervention_ref(int_ref))

        layer_interventions = ", ".join(layer_entry.intervention_refs)

        result = check_counterfactual(
            protocol_json=protocol_json,
            layer_name=layer_entry.layer.value,
            layer_interventions=layer_interventions,
            evidence_items=evidence,
            reasoning_engine=reasoning_engine,
        )
        if result:
            results.append(result)

    return results
