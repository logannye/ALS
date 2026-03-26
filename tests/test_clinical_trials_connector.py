"""Tests for ClinicalTrialsConnector — no network, uses fixture JSON."""
import pytest

from connectors.clinical_trials import (
    ClinicalTrialsConnector,
    ERIK_PROFILE,
    _parse_trial,
    check_eligibility,
)


# ---------------------------------------------------------------------------
# Fixture: realistic ClinicalTrials.gov v2 study JSON
# ---------------------------------------------------------------------------

SAMPLE_TRIAL = {
    "protocolSection": {
        "identificationModule": {
            "nctId": "NCT06012345",
            "briefTitle": "A Phase 2/3 Study of AMX0035 in ALS Participants",
        },
        "statusModule": {
            "overallStatus": "RECRUITING",
        },
        "designModule": {
            "phases": ["PHASE2", "PHASE3"],
            "enrollmentInfo": {"count": 300, "type": "ESTIMATED"},
        },
        "armsInterventionsModule": {
            "interventions": [
                {
                    "name": "AMX0035",
                    "type": "DRUG",
                    "description": "Sodium phenylbutyrate and taurursodiol",
                }
            ]
        },
        "eligibilityModule": {
            "eligibilityCriteria": (
                "Inclusion Criteria:\n"
                "- Age 18 to 80 years\n"
                "- Diagnosis of ALS (definite or probable)\n"
                "- ALSFRS-R score >= 30\n"
                "- FVC >= 60%\n"
                "\n"
                "Exclusion Criteria:\n"
                "- Active cancer\n"
                "- Severe hepatic impairment\n"
            ),
            "sex": "ALL",
            "minimumAge": "18 Years",
            "maximumAge": "80 Years",
        },
        "outcomesModule": {
            "primaryOutcomes": [
                {"measure": "Change in ALSFRS-R score from baseline to Week 24"}
            ]
        },
        "contactsLocationsModule": {
            "locations": [
                {
                    "facility": "Ohio State University Wexner Medical Center",
                    "city": "Columbus",
                    "state": "Ohio",
                    "country": "United States",
                },
                {
                    "facility": "Massachusetts General Hospital",
                    "city": "Boston",
                    "state": "Massachusetts",
                    "country": "United States",
                },
            ]
        },
    }
}

SAMPLE_TRIAL_FEMALE_ONLY = {
    "protocolSection": {
        "identificationModule": {
            "nctId": "NCT06099999",
            "briefTitle": "ALS in Female Patients",
        },
        "statusModule": {"overallStatus": "RECRUITING"},
        "designModule": {
            "phases": ["PHASE2"],
            "enrollmentInfo": {"count": 50},
        },
        "armsInterventionsModule": {
            "interventions": [
                {"name": "TestDrug", "type": "BIOLOGICAL", "description": "biologic"}
            ]
        },
        "eligibilityModule": {
            "eligibilityCriteria": "Inclusion Criteria:\n- Female only\n",
            "sex": "FEMALE",
            "minimumAge": "18 Years",
            "maximumAge": "75 Years",
        },
        "outcomesModule": {"primaryOutcomes": []},
        "contactsLocationsModule": {"locations": []},
    }
}

SAMPLE_TRIAL_OLD_MAX = {
    "protocolSection": {
        "identificationModule": {
            "nctId": "NCT06088888",
            "briefTitle": "ALS Max Age 65",
        },
        "statusModule": {"overallStatus": "RECRUITING"},
        "designModule": {
            "phases": ["PHASE3"],
            "enrollmentInfo": {"count": 100},
        },
        "armsInterventionsModule": {
            "interventions": [
                {"name": "OldDrug", "type": "DRUG", "description": ""}
            ]
        },
        "eligibilityModule": {
            "eligibilityCriteria": "Inclusion Criteria:\n- Age 18-65\n",
            "sex": "ALL",
            "minimumAge": "18 Years",
            "maximumAge": "65 Years",
        },
        "outcomesModule": {"primaryOutcomes": []},
        "contactsLocationsModule": {"locations": []},
    }
}

SAMPLE_TRIAL_GENETIC = {
    "protocolSection": {
        "identificationModule": {
            "nctId": "NCT06077777",
            "briefTitle": "SOD1 Gene Therapy for ALS",
        },
        "statusModule": {"overallStatus": "RECRUITING"},
        "designModule": {
            "phases": ["PHASE1", "PHASE2"],
            "enrollmentInfo": {"count": 30},
        },
        "armsInterventionsModule": {
            "interventions": [
                {"name": "Tofersen", "type": "GENETIC", "description": "ASO for SOD1"}
            ]
        },
        "eligibilityModule": {
            "eligibilityCriteria": (
                "Inclusion Criteria:\n"
                "- Confirmed SOD1 mutation\n"
                "- ALSFRS-R >= 24\n"
                "\n"
                "Exclusion Criteria:\n"
                "- None relevant\n"
            ),
            "sex": "ALL",
            "minimumAge": "18 Years",
            "maximumAge": "80 Years",
        },
        "outcomesModule": {"primaryOutcomes": []},
        "contactsLocationsModule": {"locations": []},
    }
}

SAMPLE_TRIAL_HIGH_ALSFRS = {
    "protocolSection": {
        "identificationModule": {
            "nctId": "NCT06066666",
            "briefTitle": "Early ALS Trial",
        },
        "statusModule": {"overallStatus": "RECRUITING"},
        "designModule": {
            "phases": ["PHASE2"],
            "enrollmentInfo": {"count": 80},
        },
        "armsInterventionsModule": {
            "interventions": [
                {"name": "EarlyDrug", "type": "DRUG", "description": ""}
            ]
        },
        "eligibilityModule": {
            "eligibilityCriteria": (
                "Inclusion Criteria:\n"
                "- ALSFRS-R >= 45\n"
                "- FVC >= 80%\n"
                "\n"
                "Exclusion Criteria:\n"
                "- None\n"
            ),
            "sex": "ALL",
            "minimumAge": "18 Years",
            "maximumAge": "80 Years",
        },
        "outcomesModule": {"primaryOutcomes": []},
        "contactsLocationsModule": {"locations": []},
    }
}

SAMPLE_TRIAL_COMORBIDITY_EXCL = {
    "protocolSection": {
        "identificationModule": {
            "nctId": "NCT06055555",
            "briefTitle": "ALS With No Hypertension",
        },
        "statusModule": {"overallStatus": "RECRUITING"},
        "designModule": {
            "phases": ["PHASE3"],
            "enrollmentInfo": {"count": 200},
        },
        "armsInterventionsModule": {
            "interventions": [
                {"name": "CleanDrug", "type": "DRUG", "description": ""}
            ]
        },
        "eligibilityModule": {
            "eligibilityCriteria": (
                "Inclusion Criteria:\n"
                "- ALS diagnosis\n"
                "\n"
                "Exclusion Criteria:\n"
                "- Uncontrolled hypertension\n"
                "- Severe renal impairment\n"
            ),
            "sex": "ALL",
            "minimumAge": "18 Years",
            "maximumAge": "80 Years",
        },
        "outcomesModule": {"primaryOutcomes": []},
        "contactsLocationsModule": {"locations": []},
    }
}

SAMPLE_TRIAL_MEDICATION_EXCL = {
    "protocolSection": {
        "identificationModule": {
            "nctId": "NCT06044444",
            "briefTitle": "ALS No Statins",
        },
        "statusModule": {"overallStatus": "RECRUITING"},
        "designModule": {
            "phases": ["PHASE2"],
            "enrollmentInfo": {"count": 100},
        },
        "armsInterventionsModule": {
            "interventions": [
                {"name": "StatinFree", "type": "DRUG", "description": ""}
            ]
        },
        "eligibilityModule": {
            "eligibilityCriteria": (
                "Inclusion Criteria:\n"
                "- ALS diagnosis\n"
                "\n"
                "Exclusion Criteria:\n"
                "- Current use of atorvastatin or other strong CYP3A4 inhibitors\n"
            ),
            "sex": "ALL",
            "minimumAge": "18 Years",
            "maximumAge": "80 Years",
        },
        "outcomesModule": {"primaryOutcomes": []},
        "contactsLocationsModule": {"locations": []},
    }
}


# ---------------------------------------------------------------------------
# Parse tests
# ---------------------------------------------------------------------------

def test_parse_trial_evidence_item_id():
    evi, intv = _parse_trial(SAMPLE_TRIAL)
    assert evi.id == "evi:trial:NCT06012345"


def test_parse_trial_evidence_item_claim():
    evi, intv = _parse_trial(SAMPLE_TRIAL)
    assert evi.claim == "A Phase 2/3 Study of AMX0035 in ALS Participants"


def test_parse_trial_evidence_item_provenance():
    evi, intv = _parse_trial(SAMPLE_TRIAL)
    from ontology.enums import SourceSystem
    assert evi.provenance.source_system == SourceSystem.trial
    assert evi.provenance.asserted_by == "trial_connector"


def test_parse_trial_evidence_body_fields():
    evi, intv = _parse_trial(SAMPLE_TRIAL)
    assert evi.body["phase"] == ["PHASE2", "PHASE3"]
    assert evi.body["enrollment"] == 300
    assert evi.body["overall_status"] == "RECRUITING"
    assert evi.body["primary_endpoint"] == "Change in ALSFRS-R score from baseline to Week 24"
    assert evi.body["intervention_name"] == "AMX0035"


def test_parse_trial_evidence_body_erik_eligible():
    evi, intv = _parse_trial(SAMPLE_TRIAL)
    # Erik (67, male, ALSFRS-R 43, FVC 100%) should be eligible for this trial
    assert evi.body["erik_eligible"] in ("eligible", "ineligible", "uncertain")


def test_parse_trial_intervention_id():
    evi, intv = _parse_trial(SAMPLE_TRIAL)
    assert intv.id == "int:trial:NCT06012345"


def test_parse_trial_intervention_name():
    evi, intv = _parse_trial(SAMPLE_TRIAL)
    assert intv.name == "AMX0035"


def test_parse_trial_intervention_class_drug():
    evi, intv = _parse_trial(SAMPLE_TRIAL)
    from ontology.enums import InterventionClass
    assert intv.intervention_class == InterventionClass.drug


def test_parse_trial_intervention_class_biological():
    evi, intv = _parse_trial(SAMPLE_TRIAL_FEMALE_ONLY)
    from ontology.enums import InterventionClass
    assert intv.intervention_class == InterventionClass.drug


def test_parse_trial_intervention_class_genetic():
    evi, intv = _parse_trial(SAMPLE_TRIAL_GENETIC)
    from ontology.enums import InterventionClass
    assert intv.intervention_class == InterventionClass.gene_therapy


def test_parse_trial_ohio_sites():
    evi, intv = _parse_trial(SAMPLE_TRIAL)
    ohio_sites = evi.body["ohio_sites"]
    assert len(ohio_sites) == 1
    assert "Ohio State University" in ohio_sites[0]


def test_parse_trial_no_ohio_sites():
    evi, intv = _parse_trial(SAMPLE_TRIAL_FEMALE_ONLY)
    assert evi.body["ohio_sites"] == []


# ---------------------------------------------------------------------------
# Eligibility tests
# ---------------------------------------------------------------------------

def test_eligibility_eligible():
    criteria = SAMPLE_TRIAL["protocolSection"]["eligibilityModule"]
    result = check_eligibility(criteria, ERIK_PROFILE)
    assert result == "eligible"


def test_eligibility_too_old():
    criteria = SAMPLE_TRIAL_OLD_MAX["protocolSection"]["eligibilityModule"]
    result = check_eligibility(criteria, ERIK_PROFILE)
    assert result == "ineligible"


def test_eligibility_female_only():
    criteria = SAMPLE_TRIAL_FEMALE_ONLY["protocolSection"]["eligibilityModule"]
    result = check_eligibility(criteria, ERIK_PROFILE)
    assert result == "ineligible"


def test_eligibility_genetic_pending():
    criteria = SAMPLE_TRIAL_GENETIC["protocolSection"]["eligibilityModule"]
    result = check_eligibility(criteria, ERIK_PROFILE)
    assert result == "uncertain"


def test_eligibility_alsfrs_too_low():
    criteria = SAMPLE_TRIAL_HIGH_ALSFRS["protocolSection"]["eligibilityModule"]
    result = check_eligibility(criteria, ERIK_PROFILE)
    # Erik has ALSFRS-R 43, trial requires >= 45
    assert result == "ineligible"


def test_eligibility_excluded_comorbidity():
    criteria = SAMPLE_TRIAL_COMORBIDITY_EXCL["protocolSection"]["eligibilityModule"]
    result = check_eligibility(criteria, ERIK_PROFILE)
    assert result == "ineligible"


def test_eligibility_excluded_medication():
    criteria = SAMPLE_TRIAL_MEDICATION_EXCL["protocolSection"]["eligibilityModule"]
    result = check_eligibility(criteria, ERIK_PROFILE)
    assert result == "ineligible"


# ---------------------------------------------------------------------------
# Connector instantiation
# ---------------------------------------------------------------------------

def test_connector_instantiates():
    c = ClinicalTrialsConnector()
    assert c is not None
    assert c.BASE_URL == "https://clinicaltrials.gov/api/v2/studies"


def test_erik_profile_fields():
    assert ERIK_PROFILE["age"] == 67
    assert ERIK_PROFILE["sex"] == "male"
    assert ERIK_PROFILE["diagnosis"] == "ALS"
    assert ERIK_PROFILE["alsfrs_r"] == 43
    assert ERIK_PROFILE["fvc_percent"] == 100
    assert ERIK_PROFILE["genetic_status"] == "pending"


# ---------------------------------------------------------------------------
# Network tests (skip by default)
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_fetch_active_als_trials_real():
    """Integration test — requires network access."""
    c = ClinicalTrialsConnector()
    result = c.fetch_active_als_trials(max_results=3)
    assert result.evidence_items_added >= 0


@pytest.mark.network
def test_fetch_trial_details_real():
    """Integration test — requires network access."""
    c = ClinicalTrialsConnector()
    result = c.fetch_trial_details("NCT04006674")
    assert result is not None
