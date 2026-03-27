"""Stage 6 — Full pipeline orchestrator for Erik's cure protocol.

Runs all 6 stages in sequence:

1. State materialization (observations → latent disease state)
2. Subtype inference (state + evidence → posterior over 8 ALS subtypes)
3. Intervention scoring (score ~25 interventions against Erik's profile)
4. Protocol assembly (select best per layer, sequence, check interactions)
5. Counterfactual verification (stress-test each layer)
6. Output (finalize and persist)

Public API
----------
generate_cure_protocol(use_llm, model_path)
    -> dict with keys: patient, trajectory, snapshot, subtype_profile,
       intervention_scores, protocol, counterfactuals
"""
from __future__ import annotations

import json
from typing import Any, Optional

from ingestion.patient_builder import build_erik_draper
from evidence.evidence_store import EvidenceStore
from world_model.reasoning_engine import ReasoningEngine
from world_model.state_materializer import materialize_state
from world_model.subtype_inference import infer_subtype
from world_model.intervention_scorer import score_all_interventions
from world_model.protocol_assembler import assemble_protocol
from world_model.counterfactual_check import run_counterfactual_analysis


def generate_cure_protocol(
    use_llm: bool = True,
    model_path: Optional[str] = None,
) -> dict[str, Any]:
    """Generate Erik Draper's cure protocol candidate.

    Parameters
    ----------
    use_llm:
        When ``True``, run all 6 stages including LLM-backed inference.
        When ``False``, only Stage 1 (deterministic) runs.
    model_path:
        Override LLM model path (defaults to config).

    Returns
    -------
    dict
        Keys: ``patient``, ``trajectory``, ``snapshot``, ``subtype_profile``,
        ``intervention_scores``, ``protocol``, ``counterfactuals``.
    """
    print("[PIPELINE] Loading patient data...")
    patient, trajectory, observations = build_erik_draper()

    # ------------------------------------------------------------------
    # Stage 1: State Materialization (deterministic — no LLM needed)
    # ------------------------------------------------------------------
    print("[PIPELINE] Stage 1: Materializing disease state...")
    snapshot = materialize_state(trajectory, observations, use_llm=False)

    result: dict[str, Any] = {
        "patient": patient,
        "trajectory": trajectory,
        "snapshot": snapshot,
        "subtype_profile": None,
        "intervention_scores": [],
        "protocol": None,
        "counterfactuals": [],
    }

    if not use_llm:
        print("[PIPELINE] LLM disabled — returning state snapshot only.")
        return result

    # ------------------------------------------------------------------
    # LLM engine + evidence store
    # ------------------------------------------------------------------
    reasoning_engine = ReasoningEngine(model_path=model_path)
    store = EvidenceStore()

    patient_state_json = json.dumps(snapshot.body, indent=2, default=str)

    # ------------------------------------------------------------------
    # Stage 2: Subtype Inference
    # ------------------------------------------------------------------
    print("[PIPELINE] Stage 2: Inferring subtype...")
    subtype_evidence: list[dict] = []
    for layer in ["root_cause_suppression", "pathology_reversal"]:
        subtype_evidence.extend(store.query_by_protocol_layer(layer))
    # Cap context size to avoid exceeding LLM context window
    subtype_evidence = subtype_evidence[:30]

    subtype_profile = infer_subtype(
        patient_state_json=patient_state_json,
        evidence_items=subtype_evidence,
        subject_ref=trajectory.id,
        reasoning_engine=reasoning_engine,
    )
    result["subtype_profile"] = subtype_profile

    subtype_posterior_json = json.dumps(
        {k.value: v for k, v in subtype_profile.posterior.items()},
        indent=2,
    )

    # ------------------------------------------------------------------
    # Stage 3: Intervention Scoring
    # ------------------------------------------------------------------
    print("[PIPELINE] Stage 3: Scoring interventions...")
    interventions_raw = store.query_all_interventions()

    scores = score_all_interventions(
        interventions=interventions_raw,
        evidence_store=store,
        patient_state_json=patient_state_json,
        subtype_posterior_json=subtype_posterior_json,
        reasoning_engine=reasoning_engine,
    )
    result["intervention_scores"] = scores

    # ------------------------------------------------------------------
    # Stage 4: Protocol Assembly
    # ------------------------------------------------------------------
    print("[PIPELINE] Stage 4: Assembling protocol...")
    protocol = assemble_protocol(scores, trajectory.id)
    result["protocol"] = protocol

    # ------------------------------------------------------------------
    # Stage 5: Counterfactual Verification
    # ------------------------------------------------------------------
    print("[PIPELINE] Stage 5: Running counterfactual analysis...")
    counterfactuals = run_counterfactual_analysis(
        protocol=protocol,
        evidence_store=store,
        reasoning_engine=reasoning_engine,
    )
    result["counterfactuals"] = counterfactuals

    # ------------------------------------------------------------------
    # Stage 6: Finalize
    # ------------------------------------------------------------------
    print("[PIPELINE] Stage 6: Finalizing protocol...")
    protocol.body["counterfactual_analysis"] = [
        cf.model_dump() for cf in counterfactuals
    ]
    protocol.body["subtype_posterior"] = {
        k.value: v for k, v in subtype_profile.posterior.items()
    }

    # Persist to PostgreSQL
    try:
        store.upsert_object(protocol)
    except Exception as e:
        print(f"[PIPELINE] Warning: could not persist protocol: {e}")

    print(f"[PIPELINE] Complete. Protocol: {protocol.id}")
    print(f"  Layers: {len(protocol.layers)}")
    print(f"  Evidence cited: {len(protocol.evidence_bundle_refs)}")
    print(f"  Approval state: {protocol.approval_state.value}")

    return result
