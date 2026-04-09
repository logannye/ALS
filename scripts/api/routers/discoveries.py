"""Daily research discoveries endpoint — family-facing timeline."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Query

from db.pool import get_connection

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# ---------------------------------------------------------------------------
# Highlight categories
# ---------------------------------------------------------------------------

_CAT_RESEARCH = "research"
_CAT_TREATMENT = "treatment"
_CAT_TRIAL = "trial"
_CAT_DRUG = "drug_design"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _query_day_metrics(conn, day_start: datetime, day_end: datetime) -> dict:
    """Query all metrics for a single calendar day from the DB."""

    # Evidence items added
    row = conn.execute(
        """SELECT COUNT(*) FROM erik_core.objects
           WHERE type = 'EvidenceItem'
             AND created_at >= %s AND created_at < %s""",
        (day_start, day_end),
    ).fetchone()
    evidence_added = row[0] if row else 0

    # Entities added
    row = conn.execute(
        """SELECT COUNT(*) FROM erik_core.entities
           WHERE created_at >= %s AND created_at < %s""",
        (day_start, day_end),
    ).fetchone()
    entities_added = row[0] if row else 0

    # Trials found (clinicaltrials.gov provenance)
    row = conn.execute(
        """SELECT COUNT(*) FROM erik_core.objects
           WHERE type = 'EvidenceItem'
             AND provenance_source_system = 'clinicaltrials.gov'
             AND created_at >= %s AND created_at < %s""",
        (day_start, day_end),
    ).fetchone()
    trials_found = row[0] if row else 0

    # Drug molecules designed
    row = conn.execute(
        """SELECT COUNT(*) FROM erik_core.objects
           WHERE type = 'EvidenceItem'
             AND body->>'provenance' LIKE '%%design_molecule%%'
             AND created_at >= %s AND created_at < %s""",
        (day_start, day_end),
    ).fetchone()
    drug_molecules = row[0] if row else 0

    # Step count from research_state
    row = conn.execute(
        """SELECT (state_json->>'step_count')::int
           FROM erik_ops.research_state
           WHERE subject_ref = 'traj:draper_001'
           ORDER BY updated_at DESC LIMIT 1"""
    ).fetchone()
    step_count = row[0] if row else 0

    return {
        "evidence_added": evidence_added,
        "entities_added": entities_added,
        "trials_found": trials_found,
        "drug_molecules": drug_molecules,
        "step_count": step_count,
    }


# ---------------------------------------------------------------------------
# Highlight builder
# ---------------------------------------------------------------------------

def _build_highlights(metrics: dict) -> list[dict[str, str]]:
    """Convert raw metrics into plain-English highlight sentences."""
    highlights: list[dict[str, str]] = []

    evi = metrics["evidence_added"]
    if evi > 0:
        highlights.append({
            "text": f"Analyzed {evi} new piece{'s' if evi != 1 else ''} of ALS research evidence.",
            "category": _CAT_RESEARCH,
        })

    ent = metrics["entities_added"]
    if ent > 0:
        highlights.append({
            "text": f"Discovered {ent} new biological entit{'ies' if ent != 1 else 'y'} (genes, proteins, pathways).",
            "category": _CAT_RESEARCH,
        })

    trials = metrics["trials_found"]
    if trials > 0:
        highlights.append({
            "text": f"Found {trials} clinical trial{'s' if trials != 1 else ''} that may be relevant.",
            "category": _CAT_TRIAL,
        })

    drugs = metrics["drug_molecules"]
    if drugs > 0:
        highlights.append({
            "text": f"Designed {drugs} candidate drug molecule{'s' if drugs != 1 else ''} for evaluation.",
            "category": _CAT_DRUG,
        })

    # If nothing happened, add a quiet-day note
    if not highlights:
        highlights.append({
            "text": "Research continued in the background with no major new findings.",
            "category": _CAT_RESEARCH,
        })

    return highlights


# ---------------------------------------------------------------------------
# Public API: build summaries
# ---------------------------------------------------------------------------

def build_daily_summary(target_date: date, dry_run: bool = False) -> dict[str, Any]:
    """Build a single day's summary.

    In dry_run mode, returns a minimal valid response without touching the DB.
    """
    date_str = target_date.isoformat()

    if dry_run:
        return {
            "date": date_str,
            "highlights": [
                {"text": "Research continued in the background with no major new findings.", "category": _CAT_RESEARCH},
            ],
            "milestone": None,
            "evidence_added": 0,
            "step_count": 0,
        }

    day_start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)

    try:
        with get_connection() as conn:
            metrics = _query_day_metrics(conn, day_start, day_end)
    except Exception:
        logger.exception("Failed to query metrics for %s", date_str)
        metrics = {
            "evidence_added": 0,
            "entities_added": 0,
            "trials_found": 0,
            "drug_molecules": 0,
            "step_count": 0,
        }

    highlights = _build_highlights(metrics)

    return {
        "date": date_str,
        "highlights": highlights,
        "milestone": None,
        "evidence_added": metrics["evidence_added"],
        "step_count": metrics["step_count"],
    }


def build_discoveries_response(days: int = 14, dry_run: bool = False) -> dict[str, Any]:
    """Build the full discoveries response for *days* calendar days, newest first."""
    today = date.today()
    day_list = []
    for offset in range(days):
        target = today - timedelta(days=offset)
        day_list.append(build_daily_summary(target, dry_run=dry_run))

    return {"days": day_list}


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.get("/discoveries")
def get_discoveries(days: int = Query(14, ge=1, le=90)):
    """Return daily research summaries for the family timeline."""
    return build_discoveries_response(days=days)
