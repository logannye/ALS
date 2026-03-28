"""Clinical trial eligibility matching for Erik (ALS patient).

Provides:
- ERIK_ELIGIBILITY_PROFILE — canonical patient profile dict
- EligibilityVerdict — Pydantic v2 result model
- check_structured_eligibility() — structured field checks (age, sex)
- extract_criteria_from_text() — regex extraction from free-text criteria
- compute_eligibility() — combined structured + free-text verdict
- upsert_watchlist() — persist verdict to erik_ops.trial_watchlist
"""
from __future__ import annotations

import logging
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Erik canonical patient profile
# ---------------------------------------------------------------------------

ERIK_ELIGIBILITY_PROFILE: dict = {
    "age": 67,
    "sex": "male",
    "diagnosis": "ALS",
    "onset_region": "lower_limb",
    "alsfrs_r": 43,
    "fvc_percent": 100,
    "disease_duration_months": 14,
    "on_riluzole": True,
    "genetic_status": "pending",
    "comorbidities": ["hypertension", "prediabetes", "cervical_stenosis"],
}

# ---------------------------------------------------------------------------
# EligibilityVerdict model
# ---------------------------------------------------------------------------

EligibleStatus = Literal["yes", "no", "likely", "pending_data"]


class EligibilityVerdict(BaseModel):
    """Result of matching Erik's profile against a single clinical trial."""

    trial_nct_id: str
    trial_title: str
    trial_phase: str
    intervention_name: str
    eligible: EligibleStatus
    blocking_criteria: list[str] = Field(default_factory=list)
    pending_criteria: list[str] = Field(default_factory=list)
    matching_criteria: list[str] = Field(default_factory=list)
    protocol_alignment: float = Field(ge=0.0, le=1.0)
    urgency: str
    sites_near_erik: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Structured eligibility check (age, sex, healthy_volunteers)
# ---------------------------------------------------------------------------

def check_structured_eligibility(
    min_age: Optional[int],
    max_age: Optional[int],
    sex: str,
    healthy_volunteers: bool,
) -> dict[str, list[str]]:
    """Check Erik's structured profile against structured eligibility fields.

    Parameters
    ----------
    min_age, max_age:
        Integer age bounds. None means no limit.
    sex:
        "MALE", "FEMALE", or "ALL".
    healthy_volunteers:
        True if trial requires healthy (non-patient) volunteers.

    Returns
    -------
    dict with "matching" and "blocking" lists of human-readable reason strings.
    """
    erik_age = ERIK_ELIGIBILITY_PROFILE["age"]
    erik_sex = ERIK_ELIGIBILITY_PROFILE["sex"].upper()

    matching: list[str] = []
    blocking: list[str] = []

    # Age
    if min_age is not None or max_age is not None:
        lo = min_age if min_age is not None else 0
        hi = max_age if max_age is not None else 999
        if lo <= erik_age <= hi:
            matching.append(f"Age {erik_age} in range [{lo}, {hi}]")
        else:
            blocking.append(f"Age {erik_age} outside range [{lo}, {hi}]")

    # Sex
    sex_upper = sex.upper() if sex else "ALL"
    if sex_upper == "ALL":
        matching.append("Sex: All genders accepted")
    elif sex_upper == erik_sex:
        matching.append(f"Sex matches ({sex_upper})")
    else:
        blocking.append(f"Sex mismatch: trial requires {sex_upper}, Erik is {erik_sex}")

    # Healthy volunteers — Erik has ALS, so healthy-only trials are excluded
    if healthy_volunteers:
        blocking.append("Trial requires healthy volunteers (Erik has ALS diagnosis)")

    return {"matching": matching, "blocking": blocking}


# ---------------------------------------------------------------------------
# Free-text criteria extraction
# ---------------------------------------------------------------------------

_ALSFRS_PATTERN = re.compile(
    r"ALSFRS-?R?\s*(?:score\s*)?(?:>=?|≥|greater\s+than|at\s+least)\s*(\d+)",
    re.IGNORECASE,
)
_FVC_PATTERN = re.compile(
    r"FVC\s*(?:>=?|≥)\s*(\d+)",
    re.IGNORECASE,
)
_DURATION_PATTERN = re.compile(
    r"(?:within|less\s+than|<)\s*(\d+)\s*months?",
    re.IGNORECASE,
)
_RILUZOLE_PATTERN = re.compile(r"\briluzole\b", re.IGNORECASE)
_GENETIC_KEYWORDS = re.compile(
    r"\b(SOD1|C9orf72|FUS|TARDBP|confirmed\s+mutation|genetic\s+confirmation)\b",
    re.IGNORECASE,
)


def extract_criteria_from_text(text: str) -> dict:
    """Extract structured thresholds and requirements from free-text eligibility criteria.

    Parameters
    ----------
    text:
        Raw eligibility criteria text from ClinicalTrials.gov.

    Returns
    -------
    dict with keys:
        alsfrs_r_min (int | None)
        fvc_min_percent (int | None)
        max_duration_months (int | None)
        riluzole_required (bool)
        genetic_required (bool)
    """
    alsfrs_r_min: Optional[int] = None
    fvc_min_percent: Optional[int] = None
    max_duration_months: Optional[int] = None
    riluzole_required: bool = False
    genetic_required: bool = False

    if not text:
        return {
            "alsfrs_r_min": alsfrs_r_min,
            "fvc_min_percent": fvc_min_percent,
            "max_duration_months": max_duration_months,
            "riluzole_required": riluzole_required,
            "genetic_required": genetic_required,
        }

    # ALSFRS-R minimum
    m = _ALSFRS_PATTERN.search(text)
    if m:
        alsfrs_r_min = int(m.group(1))

    # FVC minimum
    m = _FVC_PATTERN.search(text)
    if m:
        fvc_min_percent = int(m.group(1))

    # Maximum disease duration in months
    m = _DURATION_PATTERN.search(text)
    if m:
        max_duration_months = int(m.group(1))

    # Riluzole required
    if _RILUZOLE_PATTERN.search(text):
        riluzole_required = True

    # Genetic confirmation required
    if _GENETIC_KEYWORDS.search(text):
        genetic_required = True

    return {
        "alsfrs_r_min": alsfrs_r_min,
        "fvc_min_percent": fvc_min_percent,
        "max_duration_months": max_duration_months,
        "riluzole_required": riluzole_required,
        "genetic_required": genetic_required,
    }


# ---------------------------------------------------------------------------
# Enrollment status → urgency mapping
# ---------------------------------------------------------------------------

_URGENCY_MAP: dict[str, str] = {
    "RECRUITING": "enrolling_now",
    "ENROLLING_BY_INVITATION": "enrolling_now",
    "NOT_YET_RECRUITING": "not_yet_recruiting",
}


def _map_urgency(enrollment_status: str) -> str:
    return _URGENCY_MAP.get(enrollment_status.upper(), "completed")


# ---------------------------------------------------------------------------
# Protocol alignment scorer
# ---------------------------------------------------------------------------

def _compute_protocol_alignment(
    intervention_name: str,
    current_protocol_top_interventions: list[str],
) -> float:
    """Return a 0-1 protocol alignment score.

    0.9 — exact match (case-insensitive) against top intervention list
    0.6 — partial substring match
    0.1 — no match
    """
    name_lower = intervention_name.lower()
    for top in current_protocol_top_interventions:
        top_lower = top.lower()
        if name_lower == top_lower:
            return 0.9
        if name_lower in top_lower or top_lower in name_lower:
            return 0.6
    return 0.1


# ---------------------------------------------------------------------------
# Main compute_eligibility
# ---------------------------------------------------------------------------

def compute_eligibility(
    nct_id: str,
    title: str,
    phase: str,
    intervention_name: str,
    min_age: Optional[int],
    max_age: Optional[int],
    sex: str,
    healthy_volunteers: bool,
    eligibility_text: str,
    enrollment_status: str,
    sites: list[str],
    current_protocol_top_interventions: list[str],
    geographic_region: Optional[str] = None,
) -> EligibilityVerdict:
    """Compute full eligibility verdict for a single clinical trial.

    Parameters
    ----------
    nct_id:
        NCT identifier string (e.g. "NCT06012345").
    title:
        Trial brief title.
    phase:
        Trial phase string (e.g. "Phase 2").
    intervention_name:
        Primary intervention name.
    min_age, max_age:
        Integer age bounds. None means no limit.
    sex:
        "MALE", "FEMALE", or "ALL".
    healthy_volunteers:
        True if trial is healthy-volunteer only.
    eligibility_text:
        Free-text eligibility criteria from ClinicalTrials.gov.
    enrollment_status:
        e.g. "RECRUITING", "NOT_YET_RECRUITING", "COMPLETED".
    sites:
        List of site strings (facility + city + state).
    current_protocol_top_interventions:
        Ordered list of top interventions in current Erik research protocol.
    geographic_region:
        Region keyword used to filter sites (default "Ohio").

    Returns
    -------
    EligibilityVerdict
    """
    from config.loader import ConfigLoader
    cfg = ConfigLoader()

    profile = dict(ERIK_ELIGIBILITY_PROFILE)
    profile["alsfrs_r"] = cfg.get("trial_alsfrs_r_current", profile["alsfrs_r"])
    profile["fvc_percent"] = cfg.get("trial_fvc_current", profile["fvc_percent"])

    geographic_region = geographic_region or cfg.get("trial_geographic_region", "Ohio")

    erik = profile

    blocking: list[str] = []
    pending: list[str] = []
    matching: list[str] = []

    # --- Structured checks ---
    struct = check_structured_eligibility(min_age, max_age, sex, healthy_volunteers)
    blocking.extend(struct["blocking"])
    matching.extend(struct["matching"])

    # --- Free-text extraction ---
    extracted = extract_criteria_from_text(eligibility_text)

    # ALSFRS-R
    if extracted["alsfrs_r_min"] is not None:
        alsfrs_min = extracted["alsfrs_r_min"]
        if erik["alsfrs_r"] >= alsfrs_min:
            matching.append(f"ALSFRS-R {erik['alsfrs_r']} >= required {alsfrs_min}")
        else:
            blocking.append(
                f"ALSFRS-R {erik['alsfrs_r']} < required minimum {alsfrs_min}"
            )

    # FVC
    if extracted["fvc_min_percent"] is not None:
        fvc_min = extracted["fvc_min_percent"]
        if erik["fvc_percent"] >= fvc_min:
            matching.append(f"FVC {erik['fvc_percent']}% >= required {fvc_min}%")
        else:
            blocking.append(
                f"FVC {erik['fvc_percent']}% < required minimum {fvc_min}%"
            )

    # Disease duration
    if extracted["max_duration_months"] is not None:
        dur_max = extracted["max_duration_months"]
        if erik["disease_duration_months"] <= dur_max:
            matching.append(
                f"Disease duration {erik['disease_duration_months']}mo <= {dur_max}mo limit"
            )
        else:
            blocking.append(
                f"Disease duration {erik['disease_duration_months']}mo > {dur_max}mo limit"
            )

    # Riluzole
    if extracted["riluzole_required"]:
        if erik["on_riluzole"]:
            matching.append("On riluzole (required)")
        else:
            blocking.append("Riluzole required but Erik is not on riluzole")

    # Genetic status
    if extracted["genetic_required"]:
        if erik["genetic_status"] == "pending":
            pending.append("Genetic confirmation required; Erik's status is pending")
        elif erik["genetic_status"] == "confirmed":
            matching.append("Genetic status confirmed")
        else:
            blocking.append(
                f"Genetic confirmation required; Erik status: {erik['genetic_status']}"
            )

    # --- Protocol alignment ---
    protocol_alignment = _compute_protocol_alignment(
        intervention_name, current_protocol_top_interventions
    )

    # --- Urgency ---
    urgency = _map_urgency(enrollment_status)

    # --- Sites near Erik ---
    region_lower = geographic_region.lower()
    sites_near = [s for s in sites if region_lower in s.lower()]

    # --- Final verdict ---
    if blocking:
        eligible: EligibleStatus = "no"
    elif pending:
        eligible = "pending_data"
    elif len(matching) >= 2:
        eligible = "yes"
    else:
        eligible = "likely"

    return EligibilityVerdict(
        trial_nct_id=nct_id,
        trial_title=title,
        trial_phase=phase,
        intervention_name=intervention_name,
        eligible=eligible,
        blocking_criteria=blocking,
        pending_criteria=pending,
        matching_criteria=matching,
        protocol_alignment=protocol_alignment,
        urgency=urgency,
        sites_near_erik=sites_near,
    )


# ---------------------------------------------------------------------------
# Watchlist persistence
# ---------------------------------------------------------------------------

def upsert_watchlist(verdict: EligibilityVerdict, reviewed: bool = False) -> None:
    """Persist an EligibilityVerdict to the erik_ops.trial_watchlist table.

    Uses the shared db pool. Safe to call multiple times (upsert on nct_id).

    Parameters
    ----------
    verdict:
        The computed eligibility verdict for a trial.
    reviewed:
        Whether the trial has been manually reviewed. Defaults to False.
    """
    import json
    from db.pool import get_connection

    sql = """
        INSERT INTO erik_ops.trial_watchlist
            (nct_id, title, eligible_status, enrollment_status, phase,
             intervention_name, protocol_alignment, sites, reviewed, last_checked)
        VALUES
            (%(nct_id)s, %(title)s, %(eligible_status)s, %(enrollment_status)s,
             %(phase)s, %(intervention_name)s, %(protocol_alignment)s,
             %(sites)s::jsonb, %(reviewed)s, now())
        ON CONFLICT (nct_id) DO UPDATE SET
            title               = EXCLUDED.title,
            eligible_status     = EXCLUDED.eligible_status,
            enrollment_status   = EXCLUDED.enrollment_status,
            phase               = EXCLUDED.phase,
            intervention_name   = EXCLUDED.intervention_name,
            protocol_alignment  = EXCLUDED.protocol_alignment,
            sites               = EXCLUDED.sites,
            reviewed            = EXCLUDED.reviewed,
            last_checked        = now()
    """
    params = {
        "nct_id": verdict.trial_nct_id,
        "title": verdict.trial_title,
        "eligible_status": verdict.eligible,
        "enrollment_status": verdict.urgency,
        "phase": verdict.trial_phase,
        "intervention_name": verdict.intervention_name,
        "protocol_alignment": verdict.protocol_alignment,
        "sites": json.dumps(verdict.sites_near_erik),
        "reviewed": reviewed,
    }
    with get_connection() as conn:
        conn.execute(sql, params)
        conn.commit()
