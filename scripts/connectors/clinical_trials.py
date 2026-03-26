"""ClinicalTrialsConnector — fetches trial evidence from ClinicalTrials.gov v2 API.

Parses study JSON into EvidenceItem + Intervention pairs and upserts to DB.
Includes Erik-specific eligibility matching logic.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import requests

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import Provenance
from ontology.enums import (
    EvidenceDirection,
    EvidenceStrength,
    InterventionClass,
    SourceSystem,
)
from ontology.evidence import EvidenceItem
from ontology.intervention import Intervention

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Erik patient profile
# ---------------------------------------------------------------------------

ERIK_PROFILE: dict = {
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
    "medications": ["riluzole", "amlodipine", "atorvastatin", "ramipril"],
}


# ---------------------------------------------------------------------------
# Intervention type → InterventionClass mapping
# ---------------------------------------------------------------------------

_TYPE_TO_CLASS: dict[str, InterventionClass] = {
    "DRUG": InterventionClass.drug,
    "BIOLOGICAL": InterventionClass.drug,
    "GENETIC": InterventionClass.gene_therapy,
    "DEVICE": InterventionClass.supportive_care,
    "BEHAVIORAL": InterventionClass.rehabilitation,
    "PROCEDURE": InterventionClass.supportive_care,
    "DIETARY_SUPPLEMENT": InterventionClass.supportive_care,
    "OTHER": InterventionClass.supportive_care,
}


# ---------------------------------------------------------------------------
# Eligibility checker
# ---------------------------------------------------------------------------

def _parse_age_years(age_str: str) -> Optional[int]:
    """Parse '67 Years' format into integer years. Returns None on failure."""
    if not age_str:
        return None
    m = re.match(r"(\d+)", age_str.strip())
    return int(m.group(1)) if m else None


def check_eligibility(criteria: dict, profile: dict) -> str:
    """Check whether the profile is eligible for a trial based on criteria.

    Parameters
    ----------
    criteria:
        The ``eligibilityModule`` dict from ClinicalTrials.gov v2 API.
    profile:
        Patient profile dict (e.g. ERIK_PROFILE).

    Returns
    -------
    "eligible", "ineligible", or "uncertain"
    """
    # 1. Age range check
    min_age = _parse_age_years(criteria.get("minimumAge", ""))
    max_age = _parse_age_years(criteria.get("maximumAge", ""))
    patient_age = profile.get("age")
    if patient_age is not None:
        if min_age is not None and patient_age < min_age:
            return "ineligible"
        if max_age is not None and patient_age > max_age:
            return "ineligible"

    # 2. Sex check
    sex = criteria.get("sex", "ALL").upper()
    if sex != "ALL":
        patient_sex = profile.get("sex", "").upper()
        if patient_sex and sex != patient_sex:
            return "ineligible"

    # Get free-text eligibility criteria for further checks
    text = criteria.get("eligibilityCriteria", "")
    text_lower = text.lower()

    # 3. ALSFRS-R minimum from text
    alsfrs_match = re.search(r"alsfrs[- ]?r?\s*>=?\s*(\d+)", text_lower)
    if alsfrs_match:
        required_min = int(alsfrs_match.group(1))
        patient_alsfrs = profile.get("alsfrs_r")
        if patient_alsfrs is not None and patient_alsfrs < required_min:
            return "ineligible"

    # 4. FVC minimum from text
    fvc_match = re.search(r"fvc\s*>=?\s*(\d+)", text_lower)
    if fvc_match:
        required_fvc = int(fvc_match.group(1))
        patient_fvc = profile.get("fvc_percent")
        if patient_fvc is not None and patient_fvc < required_fvc:
            return "ineligible"

    # 5. Genetic-specific keywords when genetics pending
    genetic_status = profile.get("genetic_status", "")
    if genetic_status == "pending":
        genetic_keywords = ["sod1", "c9orf72", "fus", "tardbp", "genetic", "mutation"]
        for kw in genetic_keywords:
            if kw in text_lower:
                return "uncertain"

    # 6. Comorbidity exclusions
    exclusion_idx = text_lower.find("exclusion")
    if exclusion_idx >= 0:
        exclusion_text = text_lower[exclusion_idx:]
        comorbidities = profile.get("comorbidities", [])
        for comorbidity in comorbidities:
            if comorbidity.lower() in exclusion_text:
                return "ineligible"

        # 7. Medication exclusions
        medications = profile.get("medications", [])
        for med in medications:
            if med.lower() in exclusion_text:
                return "ineligible"

    # 8. Default: eligible
    return "eligible"


# ---------------------------------------------------------------------------
# Free function: parse a single trial study dict
# ---------------------------------------------------------------------------

def _parse_trial(study: dict) -> tuple[EvidenceItem, Intervention]:
    """Parse a ClinicalTrials.gov v2 study dict into (EvidenceItem, Intervention).

    Parameters
    ----------
    study:
        A study dict from the ClinicalTrials.gov v2 API response.

    Returns
    -------
    Tuple of (EvidenceItem, Intervention).
    """
    protocol = study["protocolSection"]
    ident = protocol["identificationModule"]
    nct_id = ident["nctId"]
    title = ident.get("briefTitle", "")

    status_mod = protocol.get("statusModule", {})
    overall_status = status_mod.get("overallStatus", "")

    design = protocol.get("designModule", {})
    phases = design.get("phases", [])
    enrollment_info = design.get("enrollmentInfo", {})
    enrollment = enrollment_info.get("count", 0)

    # Interventions
    arms_mod = protocol.get("armsInterventionsModule", {})
    interventions = arms_mod.get("interventions", [])
    first_intv = interventions[0] if interventions else {}
    intv_name = first_intv.get("name", "Unknown")
    intv_type = first_intv.get("type", "OTHER")
    intv_class = _TYPE_TO_CLASS.get(intv_type, InterventionClass.supportive_care)

    # Primary endpoint
    outcomes_mod = protocol.get("outcomesModule", {})
    primary_outcomes = outcomes_mod.get("primaryOutcomes", [])
    primary_endpoint = primary_outcomes[0].get("measure", "") if primary_outcomes else ""

    # Ohio sites
    contacts_mod = protocol.get("contactsLocationsModule", {})
    locations = contacts_mod.get("locations", [])
    ohio_sites = [
        loc.get("facility", "")
        for loc in locations
        if loc.get("state", "").lower() == "ohio"
    ]

    # Eligibility check against Erik profile
    eligibility_mod = protocol.get("eligibilityModule", {})
    erik_eligible = check_eligibility(eligibility_mod, ERIK_PROFILE)

    evidence_item = EvidenceItem(
        id=f"evi:trial:{nct_id}",
        claim=title,
        direction=EvidenceDirection.insufficient,
        strength=EvidenceStrength.unknown,
        source_refs=[f"nct:{nct_id}"],
        provenance=Provenance(
            source_system=SourceSystem.trial,
            asserted_by="trial_connector",
            source_artifact_id=nct_id,
        ),
        body={
            "protocol_layer": "",
            "mechanism_target": "",
            "applicable_subtypes": ["sporadic_tdp43", "unresolved"],
            "erik_eligible": erik_eligible,
            "pch_layer": 1,
            "phase": phases,
            "enrollment": enrollment,
            "overall_status": overall_status,
            "primary_endpoint": primary_endpoint,
            "ohio_sites": ohio_sites,
            "intervention_name": intv_name,
        },
    )

    intervention = Intervention(
        id=f"int:trial:{nct_id}",
        name=intv_name,
        intervention_class=intv_class,
        provenance=Provenance(
            source_system=SourceSystem.trial,
            asserted_by="trial_connector",
            source_artifact_id=nct_id,
        ),
    )

    return evidence_item, intervention


# ---------------------------------------------------------------------------
# ClinicalTrialsConnector
# ---------------------------------------------------------------------------

class ClinicalTrialsConnector(BaseConnector):
    """Connector for ClinicalTrials.gov v2 REST API.

    Fetches active ALS trials, parses into EvidenceItem + Intervention,
    and upserts to the Erik evidence store.
    """

    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    def __init__(self, *, store=None):
        self._store = store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, **kwargs) -> ConnectorResult:
        """Default fetch: delegates to fetch_active_als_trials."""
        return self.fetch_active_als_trials(**kwargs)

    def fetch_active_als_trials(
        self,
        *,
        phases: Optional[list[str]] = None,
        max_results: int = 50,
    ) -> ConnectorResult:
        """Query for active ALS trials matching specified phases."""
        if phases is None:
            phases = ["PHASE2", "PHASE3"]

        result = ConnectorResult()
        params = {
            "query.cond": "amyotrophic lateral sclerosis",
            "filter.overallStatus": "RECRUITING,ENROLLING_BY_INVITATION,ACTIVE_NOT_RECRUITING",
            "filter.phase": ",".join(phases),
            "pageSize": min(max_results, 100),
            "format": "json",
        }
        try:
            resp = self._retry_with_backoff(
                requests.get, self.BASE_URL, params=params, timeout=self.REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            result.errors.append(f"ClinicalTrials API failed: {e}")
            return result

        studies = data.get("studies", [])
        for study in studies[:max_results]:
            try:
                evi, intv = _parse_trial(study)
                if self._store:
                    self._store.upsert_evidence_item(evi)
                    self._store.upsert_intervention(intv)
                result.evidence_items_added += 1
                result.interventions_added += 1
            except Exception as e:
                nct = study.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "unknown")
                result.errors.append(f"Parse/upsert failed for {nct}: {e}")

        return result

    def fetch_trial_details(self, nct_id: str) -> Optional[tuple[EvidenceItem, Intervention]]:
        """Fetch a single trial by NCT ID and return parsed objects."""
        url = f"{self.BASE_URL}/{nct_id}"
        params = {"format": "json"}
        try:
            resp = self._retry_with_backoff(
                requests.get, url, params=params, timeout=self.REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            study = resp.json()
        except Exception as e:
            logger.error("Failed to fetch trial %s: %s", nct_id, e)
            return None

        try:
            evi, intv = _parse_trial(study)
            if self._store:
                self._store.upsert_evidence_item(evi)
                self._store.upsert_intervention(intv)
            return evi, intv
        except Exception as e:
            logger.error("Failed to parse trial %s: %s", nct_id, e)
            return None
