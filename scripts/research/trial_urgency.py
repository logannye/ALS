"""Trial urgency scoring — time-sensitive eligibility assessment.

Erik Draper's ALSFRS-R is declining at -0.39 points/month.  Many ALS trials
have minimum ALSFRS-R thresholds (typically 24-30) and maximum disease
duration requirements (typically 18-24 months from onset).  This module
scores trials by how urgently Erik should consider enrollment before his
eligibility window closes.

Urgency = eligibility_margin / |decline_rate| × protocol_alignment × enrollment_factor
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

from db.pool import get_connection

logger = logging.getLogger(__name__)

# Erik's current disease trajectory
CURRENT_ALSFRS_R = 43
DECLINE_RATE_PER_MONTH = -0.39  # points/month (negative)
ONSET_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)
DIAGNOSIS_DATE = datetime(2026, 3, 6, tzinfo=timezone.utc)

# Common trial thresholds
DEFAULT_MIN_ALSFRS_R = 24  # Most trials require at least this
DEFAULT_MAX_DISEASE_DURATION_MONTHS = 24

# Protocol layer → alignment keywords
PROTOCOL_ALIGNMENT_KEYWORDS = {
    "root_cause_suppression": ["sod1", "gene therapy", "antisense", "aso", "gene silencing", "tdp-43", "fus", "c9orf72"],
    "pathology_reversal": ["autophagy", "mtor", "rapamycin", "tdp-43", "aggregation", "proteostasis", "clearance"],
    "circuit_stabilization": ["riluzole", "edaravone", "glutamate", "neuroprotect", "excitotoxicity", "masitinib", "microglia"],
    "regeneration_reinnervation": ["neurotrophic", "bdnf", "gdnf", "stem cell", "reinnervation", "regenerat"],
    "adaptive_maintenance": ["respiratory", "nutrition", "exercise", "symptomatic", "palliative"],
}


@dataclass
class TrialUrgencyScore:
    """Urgency assessment for a clinical trial."""
    trial_id: str
    nct_id: str
    title: str
    months_until_ineligible: float   # Based on ALSFRS-R decline to min threshold
    protocol_alignment: float        # 0-1: how well does this trial align with our protocol?
    enrollment_status: str           # "recruiting" | "not_yet_recruiting" | "completed" | etc.
    enrollment_factor: float         # 1.0 if recruiting, 0.5 if not yet, 0.0 if completed
    urgency_score: float             # Composite score (higher = more urgent)
    phase: str
    eligibility_status: str          # "eligible" | "ineligible" | "uncertain"
    rationale: str                   # Human-readable explanation

    def to_dict(self) -> dict:
        return asdict(self)


def compute_trial_urgency(trial_body: dict, trial_id: str = "") -> TrialUrgencyScore:
    """Compute urgency score for a single trial.

    Parameters
    ----------
    trial_body:
        The ``body`` dict from an evidence item of type ClinicalTrial or
        EvidenceItem with ``provenance_source_system='clinicaltrials.gov'``.
    trial_id:
        The evidence item ID.
    """
    title = trial_body.get("claim", trial_body.get("title", "Unknown trial"))
    nct_id = trial_body.get("nct_id", "")
    phase = trial_body.get("trial_phase", "")
    status = trial_body.get("trial_status", trial_body.get("overall_status", "unknown"))
    eligibility = trial_body.get("erik_eligible", "uncertain")

    # 1. Months until ALSFRS-R drops below typical minimum threshold
    min_threshold = _extract_alsfrs_threshold(trial_body)
    margin = CURRENT_ALSFRS_R - min_threshold
    if abs(DECLINE_RATE_PER_MONTH) > 0.01:
        months_until_ineligible = margin / abs(DECLINE_RATE_PER_MONTH)
    else:
        months_until_ineligible = 120.0  # Very slow decline, effectively unlimited

    # Also check disease duration limit
    now = datetime.now(timezone.utc)
    months_since_onset = (now - ONSET_DATE).days / 30.44
    max_duration = _extract_duration_limit(trial_body)
    if max_duration:
        months_until_duration_limit = max_duration - months_since_onset
        months_until_ineligible = min(months_until_ineligible, months_until_duration_limit)

    months_until_ineligible = max(0.0, months_until_ineligible)

    # 2. Protocol alignment — does this trial target the same pathways?
    protocol_alignment = _compute_protocol_alignment(trial_body)

    # 3. Enrollment factor
    status_lower = status.lower() if status else ""
    if "recruiting" in status_lower and "not" not in status_lower:
        enrollment_factor = 1.0
    elif "not yet" in status_lower or "enrolling" in status_lower:
        enrollment_factor = 0.7
    elif "active" in status_lower:
        enrollment_factor = 0.3
    else:
        enrollment_factor = 0.0

    # 4. Composite urgency score
    # Higher urgency for: less time remaining, better alignment, actively recruiting
    if months_until_ineligible <= 0:
        urgency_score = 0.0  # Already ineligible
        rationale = "Erik likely no longer meets eligibility criteria"
    elif eligibility == "ineligible":
        urgency_score = 0.0
        rationale = f"Erik assessed as ineligible for this trial"
    else:
        # Time pressure: inverse of months remaining, capped at 1.0
        time_pressure = min(1.0, 12.0 / max(1.0, months_until_ineligible))
        urgency_score = time_pressure * (0.4 + 0.3 * protocol_alignment + 0.3 * enrollment_factor)

        parts = []
        if months_until_ineligible < 12:
            parts.append(f"~{months_until_ineligible:.0f} months until eligibility window closes")
        if protocol_alignment > 0.5:
            parts.append(f"aligns with protocol ({protocol_alignment:.0%})")
        if enrollment_factor >= 0.7:
            parts.append("actively recruiting")
        rationale = "; ".join(parts) if parts else "Moderate urgency"

    return TrialUrgencyScore(
        trial_id=trial_id,
        nct_id=nct_id,
        title=title[:200],
        months_until_ineligible=round(months_until_ineligible, 1),
        protocol_alignment=round(protocol_alignment, 2),
        enrollment_status=status,
        enrollment_factor=enrollment_factor,
        urgency_score=round(urgency_score, 3),
        phase=phase,
        eligibility_status=str(eligibility),
        rationale=rationale,
    )


def score_all_trials() -> list[TrialUrgencyScore]:
    """Score all clinical trial evidence items by urgency.

    Returns sorted list (most urgent first).
    """
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, body FROM erik_core.objects
            WHERE type = 'EvidenceItem' AND status = 'active'
              AND (provenance_source_system = 'clinicaltrials.gov'
                   OR body->>'source' = 'clinicaltrials.gov'
                   OR body->>'experiment_type' = 'trial_search')
        """).fetchall()

    scores = []
    for row in rows:
        obj_id, body = row
        if isinstance(body, str):
            body = json.loads(body)
        score = compute_trial_urgency(body, trial_id=obj_id)
        if score.urgency_score > 0:
            scores.append(score)

    scores.sort(key=lambda s: s.urgency_score, reverse=True)
    return scores


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_alsfrs_threshold(body: dict) -> int:
    """Extract minimum ALSFRS-R requirement from trial body, or default."""
    # Look for explicit threshold in eligibility criteria text
    criteria = str(body.get("eligibility_criteria", "")).lower()
    inclusion = str(body.get("inclusion_criteria", "")).lower()
    full_text = criteria + " " + inclusion

    # Common patterns: "ALSFRS-R >= 24", "ALSFRS-R score of at least 24"
    import re
    match = re.search(r"alsfrs[- ]?r?\s*(?:>=|≥|score\s+(?:of\s+)?(?:at\s+least\s+)?)\s*(\d+)", full_text)
    if match:
        return int(match.group(1))

    return DEFAULT_MIN_ALSFRS_R


def _extract_duration_limit(body: dict) -> Optional[float]:
    """Extract maximum disease duration requirement from trial body."""
    criteria = str(body.get("eligibility_criteria", "")).lower()
    inclusion = str(body.get("inclusion_criteria", "")).lower()
    full_text = criteria + " " + inclusion

    import re
    match = re.search(r"(?:disease\s+duration|onset)\s*(?:<=|≤|less\s+than|within|no\s+more\s+than)\s*(\d+)\s*months", full_text)
    if match:
        return float(match.group(1))

    # Default: no explicit limit found
    return None


def _compute_protocol_alignment(body: dict) -> float:
    """Score how well a trial aligns with Erik's current protocol layers."""
    # Combine all text fields for keyword matching
    text = " ".join(str(v) for v in [
        body.get("claim", ""),
        body.get("title", ""),
        body.get("intervention_name", ""),
        body.get("mechanism_target", ""),
        body.get("description", ""),
    ]).lower()

    if not text.strip():
        return 0.0

    matches = 0
    total_keywords = 0
    for layer, keywords in PROTOCOL_ALIGNMENT_KEYWORDS.items():
        for kw in keywords:
            total_keywords += 1
            if kw in text:
                matches += 1

    return min(1.0, matches / max(1, total_keywords) * 5)  # Scale up since most trials match few keywords
