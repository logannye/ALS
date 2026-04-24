"""Clinical trial tracker endpoint with urgency scoring.

Rewritten 2026-04-24: the previous version only queried erik_core.objects
for trial-typed evidence items. The research loop's actual eligibility
output lives in erik_ops.trial_watchlist (populated by
research/eligibility.upsert_watchlist). The endpoint now unions both
sources and returns a meaningful empty-state describing which trials
Galen is actively monitoring.
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Query

from db.pool import get_connection

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


def _fetch_from_watchlist(conn, limit: int) -> list[dict]:
    """Return rows from erik_ops.trial_watchlist — Galen's canonical
    source of truth for trials it has evaluated for Erik."""
    try:
        rows = conn.execute(
            """SELECT nct_id, title, eligible_status, phase,
                      intervention_name, protocol_alignment,
                      sites, enrollment_status, last_checked
                 FROM erik_ops.trial_watchlist
                WHERE eligible_status IN ('yes', 'likely', 'pending_data')
                ORDER BY
                    CASE eligible_status
                        WHEN 'yes'           THEN 0
                        WHEN 'likely'        THEN 1
                        WHEN 'pending_data'  THEN 2
                        ELSE 3
                    END,
                    protocol_alignment DESC,
                    last_checked DESC
                LIMIT %s""",
            (limit,),
        ).fetchall()
    except Exception as e:
        logger.warning("trial_watchlist query failed: %s", e)
        return []

    out = []
    for r in rows:
        nct_id, title, status, phase, intervention, alignment, sites, enrollment, last_checked = r
        out.append({
            "id": f"nct:{nct_id}",
            "nct_id": nct_id,
            "title": title or "",
            "phase": phase or "",
            "status": enrollment or "",
            "eligibility": status,
            "intervention_name": intervention or "",
            "protocol_alignment": float(alignment or 0.0),
            "sites": sites if isinstance(sites, list) else [],
            "last_checked": str(last_checked) if last_checked else None,
            "source": "trial_watchlist",
        })
    return out


def _fetch_from_objects(conn, limit: int) -> list[dict]:
    """Legacy source — EvidenceItem rows from the research loop's
    search_trials action. Kept alongside the watchlist so neither source
    silently disappears."""
    try:
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
    except Exception as e:
        logger.warning("objects-based trial query failed: %s", e)
        return []

    out = []
    for row in rows:
        obj_id, body, confidence, prov, created = row
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except Exception:
                body = {}
        body = body or {}
        out.append({
            "id": obj_id,
            "nct_id": body.get("nct_id", ""),
            "title": body.get("claim", body.get("title", "")),
            "phase": body.get("trial_phase", ""),
            "status": body.get("trial_status", ""),
            "eligibility": body.get("erik_eligible"),
            "confidence": confidence,
            "created_at": str(created),
            "source": "evidence",
        })
    return out


def _watchlist_summary(conn) -> dict:
    """Counts across the full watchlist — used to render the empty-state
    message so the family sees "Galen is monitoring N trials" instead of
    a blank page on days with no matches."""
    try:
        rows = conn.execute(
            """SELECT eligible_status, COUNT(*)
                 FROM erik_ops.trial_watchlist
                GROUP BY eligible_status"""
        ).fetchall()
    except Exception:
        return {"total_monitored": 0, "by_status": {}}

    by_status = {r[0]: int(r[1]) for r in rows}
    return {
        "total_monitored": sum(by_status.values()),
        "by_status": by_status,
    }


@router.get("/trials")
def list_trials(limit: int = Query(50, ge=1, le=200)):
    """Return clinical trials with urgency scores, most urgent first.

    Response shape:
        {"trials": [...], "summary": {...}}
    """
    with get_connection() as conn:
        watchlist = _fetch_from_watchlist(conn, limit=limit)
        objects = _fetch_from_objects(conn, limit=limit)
        summary = _watchlist_summary(conn)

    trials: list[dict] = list(watchlist)
    nct_seen = {t.get("nct_id") for t in trials if t.get("nct_id")}
    for t in objects:
        if t.get("nct_id") and t["nct_id"] in nct_seen:
            continue
        trials.append(t)

    # Urgency scoring (best-effort — scorer may not be importable).
    try:
        from research.trial_urgency import compute_trial_urgency
        for t in trials:
            try:
                urgency = compute_trial_urgency(t.get("body") or t, trial_id=t.get("id"))
                t["urgency"] = urgency.to_dict() if hasattr(urgency, "to_dict") else urgency
            except Exception:
                pass
        trials.sort(
            key=lambda t: t.get("urgency", {}).get("urgency_score", 0)
                          if isinstance(t.get("urgency"), dict) else 0,
            reverse=True,
        )
    except Exception:
        _tier = {"yes": 0, "likely": 1, "pending_data": 2}
        trials.sort(key=lambda t: (
            _tier.get(t.get("eligibility"), 9),
            -float(t.get("protocol_alignment") or 0.0),
        ))

    # Empty-state message: the family should never see a blank page.
    total = summary["total_monitored"]
    by_status = summary["by_status"]
    if not trials:
        if total == 0:
            summary["empty_state_message"] = (
                "Galen has not yet evaluated any ALS trials for Erik. "
                "This page will populate once the research loop completes "
                "its first trial-eligibility sweep."
            )
        else:
            ineligible = by_status.get("no", 0)
            pending = by_status.get("pending_data", 0)
            summary["empty_state_message"] = (
                f"Galen is monitoring {total} ALS "
                f"{'trial' if total == 1 else 'trials'} — "
                f"{ineligible} aren't a fit for Erik's profile and "
                f"{pending} are pending more data. None currently match. "
                "Galen re-checks eligibility when Erik's data changes."
            )
    else:
        summary["empty_state_message"] = None

    return {"trials": trials, "summary": summary}
