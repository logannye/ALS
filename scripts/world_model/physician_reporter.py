"""Physician report generator — structured clinical output from the research engine.

Produces an evidence-grounded report for Erik Draper's neurologist containing:
- Clinical summary and disease trajectory
- Protocol explanation (5-layer strategy, intervention rationale)
- Actionable recommendations (sorted by timeline urgency)
- Missing measurements that would improve the analysis
- Key uncertainties affecting treatment decisions

The report combines deterministic data (protocol, gaps, trial urgency scores)
with LLM-generated narrative (clinical summary, recommendation rationale).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from db.pool import get_connection

logger = logging.getLogger(__name__)


def generate_physician_report(use_llm: bool = True) -> dict[str, Any]:
    """Generate a comprehensive physician report.

    Parameters
    ----------
    use_llm:
        When True, uses the LLM for narrative sections (summary, rationale).
        When False, returns only deterministic/structured data.

    Returns
    -------
    Report dict ready for API response and frontend rendering.
    """
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "patient": "Erik Draper",
        "physician": "Dr. Thakore, Cleveland Clinic",
    }

    # 1. Load current protocol
    protocol_data = _load_protocol()
    report["protocol"] = protocol_data

    # 2. Load research state
    state_data = _load_research_state()
    report["research_state"] = {
        "step_count": state_data.get("step_count", 0),
        "evidence_count": _count_evidence(),
        "protocol_version": state_data.get("protocol_version", 0),
        "converged": state_data.get("converged", False),
        "causal_chain_depths": state_data.get("causal_chains", {}),
    }

    # 3. Load causal gaps
    gaps = _load_causal_gaps()
    report["causal_gaps"] = gaps

    # 4. Load trial urgency scores
    trials = _load_trial_urgency()
    report["urgent_trials"] = trials

    # 5. Compute deterministic recommendations
    recommendations = _compute_recommendations(protocol_data, gaps, trials)
    report["recommendations"] = recommendations

    # 6. LLM-generated narrative sections
    if use_llm:
        narrative = _generate_narrative(protocol_data, gaps, recommendations)
        if narrative:
            report["clinical_summary"] = narrative.get("clinical_summary", "")
            report["protocol_explanation"] = narrative.get("protocol_explanation", "")
            # Merge LLM recommendations with deterministic ones
            llm_recs = narrative.get("recommendations", [])
            if llm_recs:
                report["recommendations"] = _merge_recommendations(recommendations, llm_recs)
            report["key_uncertainties"] = narrative.get("key_uncertainties", [])
            report["cited_evidence"] = narrative.get("cited_evidence", [])
    else:
        report["clinical_summary"] = (
            "Erik Draper, 67M, diagnosed with definite ALS (Gold Coast Criteria) in March 2026. "
            "Limb-onset with ALSFRS-R 43/48, NfL 5.82 pg/mL (elevated), FVC 100%. "
            f"The research engine has analyzed {report['research_state']['evidence_count']} evidence items "
            f"across {report['research_state']['step_count']} research steps."
        )
        report["key_uncertainties"] = [g["missing_link"] for g in gaps[:5]]

    return report


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_protocol() -> dict:
    try:
        with get_connection() as conn:
            row = conn.execute(
                """SELECT id, body, confidence, updated_at FROM erik_core.objects
                   WHERE type = 'CureProtocolCandidate' AND status = 'active'
                   ORDER BY updated_at DESC LIMIT 1"""
            ).fetchone()
        if row:
            body = json.loads(row[1]) if isinstance(row[1], str) else row[1]
            return {
                "id": row[0],
                "body": body,
                "confidence": row[2],
                "updated_at": str(row[3]),
            }
    except Exception as e:
        logger.warning("Could not load protocol: %s", e)
    return {}


def _load_research_state() -> dict:
    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT state_json FROM erik_ops.research_state ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
        if row:
            return json.loads(row[0]) if isinstance(row[0], str) else row[0]
    except Exception as e:
        logger.warning("Could not load research state: %s", e)
    return {}


def _count_evidence() -> int:
    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM erik_core.objects WHERE type = 'EvidenceItem' AND status = 'active'"
            ).fetchone()
            return row[0] if row else 0
    except Exception:
        return 0


def _load_causal_gaps() -> list[dict]:
    try:
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT data FROM erik_ops.causal_gaps
                   WHERE status = 'open'
                   ORDER BY therapeutic_leverage DESC LIMIT 10"""
            ).fetchall()
        return [json.loads(r[0]) if isinstance(r[0], str) else r[0] for r in rows]
    except Exception:
        return []


def _load_trial_urgency() -> list[dict]:
    try:
        from research.trial_urgency import score_all_trials
        scores = score_all_trials()
        return [s.to_dict() for s in scores[:5]]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Deterministic recommendations
# ---------------------------------------------------------------------------

def _compute_recommendations(protocol: dict, gaps: list[dict], trials: list[dict]) -> list[dict]:
    """Generate recommendations from structured data (no LLM needed)."""
    recs: list[dict] = []

    # Genetic testing is always top priority if pending
    genetic_gap = next((g for g in gaps if "genetic" in g.get("id", "").lower()), None)
    if genetic_gap and genetic_gap.get("status") == "open":
        recs.append({
            "recommendation": "Obtain Invitae genetic panel results",
            "rationale": "Genetic subtype determination is the single highest-leverage data point. "
                         "The treatment strategy changes fundamentally based on whether Erik has SOD1, "
                         "C9orf72, FUS, or sporadic TDP-43 ALS.",
            "timeline": "immediate",
            "type": "genetic_testing",
            "cited_evidence": [],
        })

    # CSF biomarkers
    csf_gap = next((g for g in gaps if "csf" in g.get("id", "").lower()), None)
    if csf_gap and csf_gap.get("status") == "open":
        recs.append({
            "recommendation": "Order CSF NfL and phospho-TDP-43 panel",
            "rationale": "CSF biomarkers provide quantitative disease activity measurement "
                         "and establish a baseline for monitoring target engagement of future interventions.",
            "timeline": "within_month",
            "type": "biomarker",
            "cited_evidence": [],
        })

    # Urgent trials
    for trial in trials[:3]:
        if trial.get("urgency_score", 0) > 0.3:
            recs.append({
                "recommendation": f"Evaluate enrollment in {trial.get('nct_id', 'trial')} ({trial.get('title', '')[:60]})",
                "rationale": f"Urgency score {trial['urgency_score']:.2f}: "
                             f"{trial.get('rationale', 'Protocol-aligned and recruiting')}",
                "timeline": "within_month",
                "type": "trial_enrollment",
                "cited_evidence": [],
            })

    # Protocol-derived recommendations
    body = protocol.get("body", {})
    layers = body.get("layers", [])
    for layer in layers:
        refs = layer.get("intervention_refs", [])
        if refs:
            layer_name = layer.get("layer", "unknown").replace("_", " ")
            recs.append({
                "recommendation": f"Discuss {layer_name} interventions: {', '.join(r.replace('int:', '') for r in refs[:3])}",
                "rationale": f"Protocol layer '{layer_name}' recommends these interventions based on evidence analysis.",
                "timeline": "within_month",
                "type": "intervention_start",
                "cited_evidence": [],
            })

    return recs


# ---------------------------------------------------------------------------
# LLM narrative generation
# ---------------------------------------------------------------------------

def _generate_narrative(protocol: dict, gaps: list[dict], recommendations: list[dict]) -> Optional[dict]:
    """Use the LLM to generate narrative sections of the report."""
    try:
        from llm.inference import create_llm
        from world_model.prompts.templates import PHYSICIAN_REPORT_TEMPLATE

        llm = create_llm()

        # Gather top evidence for context
        evidence_items = _load_top_evidence(limit=20)

        prompt = PHYSICIAN_REPORT_TEMPLATE.replace(
            "{patient_state_json}", json.dumps({
                "alsfrs_r": 43, "decline_rate": -0.39,
                "nfl_plasma": 5.82, "fvc": 100,
                "genetic_status": "pending",
            }),
        ).replace(
            "{protocol_json}", json.dumps(protocol.get("body", {}), default=str)[:3000],
        ).replace(
            "{evidence_items_json}", json.dumps(evidence_items[:15], default=str)[:3000],
        ).replace(
            "{causal_gaps_json}", json.dumps(gaps[:5], default=str)[:1500],
        )

        from world_model.prompts.templates import SYSTEM_PROMPT
        full_prompt = SYSTEM_PROMPT + "\n\n" + prompt

        result = llm.generate_json(full_prompt, max_tokens=1500)
        return result
    except Exception as e:
        logger.warning("LLM narrative generation failed: %s", e)
        return None


def _load_top_evidence(limit: int = 20) -> list[dict]:
    """Load the most relevant evidence items for the report."""
    try:
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT id, body FROM erik_core.objects
                   WHERE type = 'EvidenceItem' AND status = 'active'
                     AND confidence IS NOT NULL
                   ORDER BY confidence DESC
                   LIMIT %s""",
                (limit,),
            ).fetchall()
        items = []
        for row in rows:
            body = json.loads(row[1]) if isinstance(row[1], str) else row[1]
            items.append({
                "id": row[0],
                "claim": body.get("claim", ""),
                "strength": body.get("evidence_strength", body.get("strength")),
                "direction": body.get("direction"),
            })
        return items
    except Exception:
        return []


def _merge_recommendations(deterministic: list[dict], llm_recs: list[dict]) -> list[dict]:
    """Merge deterministic and LLM-generated recommendations, deduplicating."""
    seen = {r["recommendation"][:30] for r in deterministic}
    merged = list(deterministic)
    for rec in llm_recs:
        if rec.get("recommendation", "")[:30] not in seen:
            merged.append(rec)
            seen.add(rec.get("recommendation", "")[:30])
    return merged
