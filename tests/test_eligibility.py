"""Tests for the clinical trial eligibility matching module.

TDD: tests are written before implementation.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# TestEligibilityVerdict
# ---------------------------------------------------------------------------

class TestEligibilityVerdict:
    def test_creation_defaults(self):
        from research.eligibility import EligibilityVerdict
        v = EligibilityVerdict(
            trial_nct_id="NCT99000001",
            trial_title="Test Trial",
            trial_phase="Phase 2",
            intervention_name="TestDrug",
            eligible="likely",
            protocol_alignment=0.5,
            urgency="standard",
        )
        assert v.trial_nct_id == "NCT99000001"
        assert v.trial_title == "Test Trial"
        assert v.trial_phase == "Phase 2"
        assert v.intervention_name == "TestDrug"
        assert v.eligible == "likely"
        assert v.blocking_criteria == []
        assert v.pending_criteria == []
        assert v.matching_criteria == []
        assert v.protocol_alignment == 0.5
        assert v.urgency == "standard"
        assert v.sites_near_erik == []

    def test_eligible_literal_yes(self):
        from research.eligibility import EligibilityVerdict
        v = EligibilityVerdict(
            trial_nct_id="NCT1",
            trial_title="T",
            trial_phase="Phase 3",
            intervention_name="Drug",
            eligible="yes",
            protocol_alignment=0.9,
            urgency="high",
        )
        assert v.eligible == "yes"

    def test_eligible_literal_no(self):
        from research.eligibility import EligibilityVerdict
        v = EligibilityVerdict(
            trial_nct_id="NCT2",
            trial_title="T",
            trial_phase="Phase 1",
            intervention_name="Drug",
            eligible="no",
            protocol_alignment=0.1,
            urgency="low",
        )
        assert v.eligible == "no"

    def test_eligible_literal_pending_data(self):
        from research.eligibility import EligibilityVerdict
        v = EligibilityVerdict(
            trial_nct_id="NCT3",
            trial_title="T",
            trial_phase="Phase 2",
            intervention_name="Drug",
            eligible="pending_data",
            protocol_alignment=0.6,
            urgency="standard",
        )
        assert v.eligible == "pending_data"

    def test_eligible_invalid_literal_raises(self):
        from research.eligibility import EligibilityVerdict
        with pytest.raises(ValidationError):
            EligibilityVerdict(
                trial_nct_id="NCT4",
                trial_title="T",
                trial_phase="Phase 2",
                intervention_name="Drug",
                eligible="maybe",  # invalid
                protocol_alignment=0.5,
                urgency="standard",
            )

    def test_full_fields(self):
        from research.eligibility import EligibilityVerdict
        v = EligibilityVerdict(
            trial_nct_id="NCT5",
            trial_title="Full Trial",
            trial_phase="Phase 2/3",
            intervention_name="AMX0035",
            eligible="yes",
            blocking_criteria=[],
            pending_criteria=[],
            matching_criteria=["Age in range", "Sex matches", "ALSFRS-R in range"],
            protocol_alignment=0.9,
            urgency="enrolling_now",
            sites_near_erik=["Cleveland Clinic, Cleveland, Ohio"],
        )
        assert len(v.matching_criteria) == 3
        assert len(v.sites_near_erik) == 1


# ---------------------------------------------------------------------------
# TestStructuredEligibility
# ---------------------------------------------------------------------------

class TestStructuredEligibility:
    def test_age_in_range(self):
        from research.eligibility import check_structured_eligibility
        result = check_structured_eligibility(
            min_age=18,
            max_age=80,
            sex="ALL",
            healthy_volunteers=False,
        )
        assert "Age 67 in range [18, 80]" in result["matching"] or any(
            "age" in m.lower() for m in result["matching"]
        )
        assert result["blocking"] == []

    def test_age_too_old_blocks(self):
        """max_age=65 should block Erik (age 67)."""
        from research.eligibility import check_structured_eligibility
        result = check_structured_eligibility(
            min_age=18,
            max_age=65,
            sex="ALL",
            healthy_volunteers=False,
        )
        assert len(result["blocking"]) >= 1
        assert any("age" in b.lower() for b in result["blocking"])

    def test_age_too_young_blocks(self):
        """min_age=70 should block Erik (age 67)."""
        from research.eligibility import check_structured_eligibility
        result = check_structured_eligibility(
            min_age=70,
            max_age=80,
            sex="ALL",
            healthy_volunteers=False,
        )
        assert len(result["blocking"]) >= 1
        assert any("age" in b.lower() for b in result["blocking"])

    def test_sex_match_male(self):
        from research.eligibility import check_structured_eligibility
        result = check_structured_eligibility(
            min_age=18,
            max_age=80,
            sex="MALE",
            healthy_volunteers=False,
        )
        assert result["blocking"] == []
        assert any("sex" in m.lower() or "male" in m.lower() for m in result["matching"])

    def test_sex_mismatch_female_only(self):
        from research.eligibility import check_structured_eligibility
        result = check_structured_eligibility(
            min_age=18,
            max_age=80,
            sex="FEMALE",
            healthy_volunteers=False,
        )
        assert len(result["blocking"]) >= 1
        assert any("sex" in b.lower() or "female" in b.lower() for b in result["blocking"])

    def test_sex_all_matches(self):
        """sex='ALL' should match regardless of Erik's sex."""
        from research.eligibility import check_structured_eligibility
        result = check_structured_eligibility(
            min_age=18,
            max_age=80,
            sex="ALL",
            healthy_volunteers=False,
        )
        assert result["blocking"] == []

    def test_no_age_limits(self):
        """None age limits should not block."""
        from research.eligibility import check_structured_eligibility
        result = check_structured_eligibility(
            min_age=None,
            max_age=None,
            sex="ALL",
            healthy_volunteers=False,
        )
        assert result["blocking"] == []

    def test_healthy_volunteers_only_blocks(self):
        """healthy_volunteers=True should block Erik (has ALS)."""
        from research.eligibility import check_structured_eligibility
        result = check_structured_eligibility(
            min_age=18,
            max_age=80,
            sex="ALL",
            healthy_volunteers=True,
        )
        assert len(result["blocking"]) >= 1
        assert any("healthy" in b.lower() for b in result["blocking"])


# ---------------------------------------------------------------------------
# TestCriteriaExtraction
# ---------------------------------------------------------------------------

class TestCriteriaExtraction:
    def test_alsfrs_r_threshold(self):
        from research.eligibility import extract_criteria_from_text
        text = "Inclusion Criteria:\n- ALSFRS-R score >= 24\n- ALS diagnosis"
        result = extract_criteria_from_text(text)
        assert result["alsfrs_r_min"] == 24

    def test_alsfrs_r_threshold_variant_spelling(self):
        from research.eligibility import extract_criteria_from_text
        text = "- ALSFRS score greater than 30"
        result = extract_criteria_from_text(text)
        assert result["alsfrs_r_min"] == 30

    def test_alsfrs_r_threshold_geq_symbol(self):
        from research.eligibility import extract_criteria_from_text
        text = "- ALSFRS-R ≥ 36 at screening"
        result = extract_criteria_from_text(text)
        assert result["alsfrs_r_min"] == 36

    def test_alsfrs_r_at_least_phrasing(self):
        from research.eligibility import extract_criteria_from_text
        text = "- ALSFRS-R at least 28"
        result = extract_criteria_from_text(text)
        assert result["alsfrs_r_min"] == 28

    def test_fvc_threshold(self):
        from research.eligibility import extract_criteria_from_text
        text = "- FVC >= 50% of predicted"
        result = extract_criteria_from_text(text)
        assert result["fvc_min_percent"] == 50

    def test_fvc_threshold_geq_symbol(self):
        from research.eligibility import extract_criteria_from_text
        text = "- FVC ≥ 60%"
        result = extract_criteria_from_text(text)
        assert result["fvc_min_percent"] == 60

    def test_disease_duration(self):
        from research.eligibility import extract_criteria_from_text
        text = "- Symptom onset within 24 months prior to enrollment"
        result = extract_criteria_from_text(text)
        assert result["max_duration_months"] == 24

    def test_disease_duration_less_than(self):
        from research.eligibility import extract_criteria_from_text
        text = "- Duration of ALS less than 36 months"
        result = extract_criteria_from_text(text)
        assert result["max_duration_months"] == 36

    def test_disease_duration_lt_symbol(self):
        from research.eligibility import extract_criteria_from_text
        text = "- Disease duration < 18 months"
        result = extract_criteria_from_text(text)
        assert result["max_duration_months"] == 18

    def test_riluzole_required(self):
        from research.eligibility import extract_criteria_from_text
        text = "Inclusion Criteria:\n- Must be on riluzole for at least 30 days"
        result = extract_criteria_from_text(text)
        assert result["riluzole_required"] is True

    def test_genetic_required_sod1(self):
        from research.eligibility import extract_criteria_from_text
        text = "Inclusion Criteria:\n- Confirmed SOD1 mutation\n- ALSFRS-R >= 24"
        result = extract_criteria_from_text(text)
        assert result["genetic_required"] is True

    def test_genetic_required_c9orf72(self):
        from research.eligibility import extract_criteria_from_text
        text = "- C9orf72 repeat expansion confirmed"
        result = extract_criteria_from_text(text)
        assert result["genetic_required"] is True

    def test_no_criteria_found(self):
        from research.eligibility import extract_criteria_from_text
        text = "Inclusion Criteria:\n- Diagnosis of ALS\n- Age 18-80\n"
        result = extract_criteria_from_text(text)
        assert result["alsfrs_r_min"] is None
        assert result["fvc_min_percent"] is None
        assert result["max_duration_months"] is None
        assert result["riluzole_required"] is False
        assert result["genetic_required"] is False

    def test_empty_text(self):
        from research.eligibility import extract_criteria_from_text
        result = extract_criteria_from_text("")
        assert result["alsfrs_r_min"] is None
        assert result["fvc_min_percent"] is None
        assert result["max_duration_months"] is None
        assert result["riluzole_required"] is False
        assert result["genetic_required"] is False


# ---------------------------------------------------------------------------
# TestComputeEligibility
# ---------------------------------------------------------------------------

# Shared protocol top interventions for all compute tests
_PROTOCOL_TOP = ["AMX0035", "tofersen", "riluzole"]


class TestComputeEligibility:
    def test_eligible_trial(self):
        """Trial with ALSFRS>=24, FVC>=50%, within 36mo, RECRUITING → yes or likely."""
        from research.eligibility import compute_eligibility
        text = (
            "Inclusion Criteria:\n"
            "- ALSFRS-R >= 24\n"
            "- FVC >= 50%\n"
            "- Symptom onset within 36 months\n"
        )
        verdict = compute_eligibility(
            nct_id="NCT99001001",
            title="Good ALS Trial",
            phase="Phase 2",
            intervention_name="AMX0035",
            min_age=18,
            max_age=80,
            sex="ALL",
            healthy_volunteers=False,
            eligibility_text=text,
            enrollment_status="RECRUITING",
            sites=["Ohio State University Wexner Medical Center, Columbus, Ohio"],
            current_protocol_top_interventions=_PROTOCOL_TOP,
        )
        assert verdict.eligible in ("yes", "likely")
        assert verdict.blocking_criteria == []

    def test_blocked_by_age(self):
        """max_age=55 blocks Erik (67)."""
        from research.eligibility import compute_eligibility
        verdict = compute_eligibility(
            nct_id="NCT99001002",
            title="Young ALS Trial",
            phase="Phase 3",
            intervention_name="SomeDrug",
            min_age=18,
            max_age=55,
            sex="ALL",
            healthy_volunteers=False,
            eligibility_text="Inclusion Criteria:\n- Age 18-55\n",
            enrollment_status="RECRUITING",
            sites=[],
            current_protocol_top_interventions=_PROTOCOL_TOP,
        )
        assert verdict.eligible == "no"
        assert len(verdict.blocking_criteria) >= 1

    def test_pending_genetics(self):
        """SOD1 required but Erik's genetic status is pending."""
        from research.eligibility import compute_eligibility
        text = (
            "Inclusion Criteria:\n"
            "- Confirmed SOD1 mutation\n"
            "- ALSFRS-R >= 24\n"
        )
        verdict = compute_eligibility(
            nct_id="NCT99001003",
            title="SOD1 Gene Therapy",
            phase="Phase 1/2",
            intervention_name="Tofersen",
            min_age=18,
            max_age=80,
            sex="ALL",
            healthy_volunteers=False,
            eligibility_text=text,
            enrollment_status="RECRUITING",
            sites=[],
            current_protocol_top_interventions=_PROTOCOL_TOP,
        )
        assert verdict.eligible == "pending_data"
        assert len(verdict.pending_criteria) >= 1

    def test_protocol_alignment_high(self):
        """Intervention in top protocol list → alignment ~0.9."""
        from research.eligibility import compute_eligibility
        verdict = compute_eligibility(
            nct_id="NCT99001004",
            title="AMX Trial",
            phase="Phase 2/3",
            intervention_name="AMX0035",
            min_age=18,
            max_age=80,
            sex="ALL",
            healthy_volunteers=False,
            eligibility_text="Inclusion Criteria:\n- ALS diagnosis\n",
            enrollment_status="RECRUITING",
            sites=[],
            current_protocol_top_interventions=_PROTOCOL_TOP,
        )
        assert verdict.protocol_alignment >= 0.8

    def test_protocol_alignment_low(self):
        """Intervention not in protocol list → alignment ~0.1."""
        from research.eligibility import compute_eligibility
        verdict = compute_eligibility(
            nct_id="NCT99001005",
            title="Obscure Trial",
            phase="Phase 1",
            intervention_name="XYZDrug999",
            min_age=18,
            max_age=80,
            sex="ALL",
            healthy_volunteers=False,
            eligibility_text="Inclusion Criteria:\n- ALS diagnosis\n",
            enrollment_status="RECRUITING",
            sites=[],
            current_protocol_top_interventions=_PROTOCOL_TOP,
        )
        assert verdict.protocol_alignment <= 0.2

    def test_enrollment_status_recruiting(self):
        from research.eligibility import compute_eligibility
        verdict = compute_eligibility(
            nct_id="NCT99001006",
            title="T",
            phase="Phase 2",
            intervention_name="Drug",
            min_age=18,
            max_age=80,
            sex="ALL",
            healthy_volunteers=False,
            eligibility_text="",
            enrollment_status="RECRUITING",
            sites=[],
            current_protocol_top_interventions=[],
        )
        assert verdict.urgency == "enrolling_now"

    def test_enrollment_status_not_yet_recruiting(self):
        from research.eligibility import compute_eligibility
        verdict = compute_eligibility(
            nct_id="NCT99001007",
            title="T",
            phase="Phase 2",
            intervention_name="Drug",
            min_age=18,
            max_age=80,
            sex="ALL",
            healthy_volunteers=False,
            eligibility_text="",
            enrollment_status="NOT_YET_RECRUITING",
            sites=[],
            current_protocol_top_interventions=[],
        )
        assert verdict.urgency == "not_yet_recruiting"

    def test_enrollment_status_completed(self):
        from research.eligibility import compute_eligibility
        verdict = compute_eligibility(
            nct_id="NCT99001008",
            title="T",
            phase="Phase 3",
            intervention_name="Drug",
            min_age=18,
            max_age=80,
            sex="ALL",
            healthy_volunteers=False,
            eligibility_text="",
            enrollment_status="COMPLETED",
            sites=[],
            current_protocol_top_interventions=[],
        )
        assert verdict.urgency == "completed"

    def test_enrollment_status_active_not_recruiting(self):
        from research.eligibility import compute_eligibility
        verdict = compute_eligibility(
            nct_id="NCT99001009",
            title="T",
            phase="Phase 3",
            intervention_name="Drug",
            min_age=18,
            max_age=80,
            sex="ALL",
            healthy_volunteers=False,
            eligibility_text="",
            enrollment_status="ACTIVE_NOT_RECRUITING",
            sites=[],
            current_protocol_top_interventions=[],
        )
        # ACTIVE_NOT_RECRUITING maps to "completed" category (not recruiting)
        assert verdict.urgency == "completed"

    def test_site_proximity_ohio(self):
        """Sites containing 'Ohio' should appear in sites_near_erik."""
        from research.eligibility import compute_eligibility
        verdict = compute_eligibility(
            nct_id="NCT99001010",
            title="T",
            phase="Phase 2",
            intervention_name="Drug",
            min_age=18,
            max_age=80,
            sex="ALL",
            healthy_volunteers=False,
            eligibility_text="",
            enrollment_status="RECRUITING",
            sites=[
                "Cleveland Clinic, Cleveland, Ohio",
                "Ohio State University Wexner Medical Center, Columbus, Ohio",
                "Massachusetts General Hospital, Boston, Massachusetts",
            ],
            current_protocol_top_interventions=[],
            geographic_region="Ohio",
        )
        assert len(verdict.sites_near_erik) == 2
        assert all("Ohio" in s or "Ohio" in s for s in verdict.sites_near_erik)

    def test_site_proximity_non_ohio_excluded(self):
        """Non-Ohio sites should not appear in sites_near_erik."""
        from research.eligibility import compute_eligibility
        verdict = compute_eligibility(
            nct_id="NCT99001011",
            title="T",
            phase="Phase 2",
            intervention_name="Drug",
            min_age=18,
            max_age=80,
            sex="ALL",
            healthy_volunteers=False,
            eligibility_text="",
            enrollment_status="RECRUITING",
            sites=[
                "Massachusetts General Hospital, Boston, Massachusetts",
                "UCSF Medical Center, San Francisco, California",
            ],
            current_protocol_top_interventions=[],
            geographic_region="Ohio",
        )
        assert verdict.sites_near_erik == []

    def test_verdict_yes_requires_two_matching(self):
        """Final verdict 'yes' requires at least 2 matching criteria with no blocking."""
        from research.eligibility import compute_eligibility
        text = (
            "Inclusion Criteria:\n"
            "- ALSFRS-R >= 24\n"
            "- FVC >= 50%\n"
        )
        verdict = compute_eligibility(
            nct_id="NCT99001012",
            title="Good Trial",
            phase="Phase 2",
            intervention_name="AMX0035",
            min_age=18,
            max_age=80,
            sex="ALL",
            healthy_volunteers=False,
            eligibility_text=text,
            enrollment_status="RECRUITING",
            sites=[],
            current_protocol_top_interventions=_PROTOCOL_TOP,
        )
        # Should be yes since age + sex + ALSFRS + FVC all match
        assert verdict.eligible in ("yes", "likely")

    def test_verdict_likely_with_one_match(self):
        """A trial with structured match but no text criteria → 'likely'."""
        from research.eligibility import compute_eligibility
        verdict = compute_eligibility(
            nct_id="NCT99001013",
            title="Minimal Trial",
            phase="Phase 1",
            intervention_name="NewDrug",
            min_age=18,
            max_age=80,
            sex="ALL",
            healthy_volunteers=False,
            eligibility_text="Inclusion Criteria:\n- ALS diagnosis\n",
            enrollment_status="RECRUITING",
            sites=[],
            current_protocol_top_interventions=[],
        )
        # No blocking, some matching (age, sex) — either yes or likely
        assert verdict.eligible in ("yes", "likely")
        assert verdict.blocking_criteria == []


# ---------------------------------------------------------------------------
# TestEligibilityProfile
# ---------------------------------------------------------------------------

class TestEligibilityProfile:
    def test_erik_profile_present(self):
        from research.eligibility import ERIK_ELIGIBILITY_PROFILE
        assert ERIK_ELIGIBILITY_PROFILE["age"] == 67
        assert ERIK_ELIGIBILITY_PROFILE["sex"] == "male"
        assert ERIK_ELIGIBILITY_PROFILE["diagnosis"] == "ALS"
        assert ERIK_ELIGIBILITY_PROFILE["alsfrs_r"] == 43
        assert ERIK_ELIGIBILITY_PROFILE["fvc_percent"] == 100
        assert ERIK_ELIGIBILITY_PROFILE["disease_duration_months"] == 14
        assert ERIK_ELIGIBILITY_PROFILE["on_riluzole"] is True
        assert ERIK_ELIGIBILITY_PROFILE["genetic_status"] == "pending"
        assert "hypertension" in ERIK_ELIGIBILITY_PROFILE["comorbidities"]
        assert "prediabetes" in ERIK_ELIGIBILITY_PROFILE["comorbidities"]
        assert "cervical_stenosis" in ERIK_ELIGIBILITY_PROFILE["comorbidities"]
        assert ERIK_ELIGIBILITY_PROFILE["onset_region"] == "lower_limb"
