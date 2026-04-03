"""Clinical trial tracker endpoint with urgency scoring."""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Query

from db.pool import get_connection

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.get("/trials")
def list_trials(limit: int = Query(50, ge=1, le=200)):
    """Return clinical trials with urgency scores, most urgent first."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, body, confidence, provenance_source_system, created_at
               FROM erik_core.objects
               WHERE type IN ('ClinicalTrial', 'EvidenceItem')
                 AND status = 'active'
                 AND (provenance_source_system = 'clinicaltrials.gov'
                      OR body->>'source' = 'clinicaltrials.gov'
                      OR body->>'experiment_type' = 'trial_search')
               ORDER BY created_at DESC
               LIMIT %s""",
            (limit,),
        ).fetchall()

    # Compute urgency scores
    try:
        from research.trial_urgency import compute_trial_urgency
        use_urgency = True
    except Exception:
        use_urgency = False

    trials = []
    for row in rows:
        obj_id, body, confidence, prov, created = row
        if isinstance(body, str):
            body = json.loads(body)

        trial_data = {
            "id": obj_id,
            "title": body.get("claim", body.get("title", "")),
            "nct_id": body.get("nct_id", ""),
            "phase": body.get("trial_phase", ""),
            "status": body.get("trial_status", ""),
            "eligibility": body.get("erik_eligible"),
            "body": body,
            "confidence": confidence,
            "created_at": str(created),
        }

        if use_urgency:
            try:
                urgency = compute_trial_urgency(body, trial_id=obj_id)
                trial_data["urgency"] = urgency.to_dict()
            except Exception:
                pass

        trials.append(trial_data)

    # Sort by urgency score if available
    if use_urgency:
        trials.sort(
            key=lambda t: t.get("urgency", {}).get("urgency_score", 0),
            reverse=True,
        )

    return {"trials": trials}
