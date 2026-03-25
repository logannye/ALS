"""Tests for ontology.patient — Patient, ALSTrajectory, ALSFRSRScore."""
import json
import pytest
from datetime import date

from ontology.enums import ALSOnsetRegion, PrivacyClass
from ontology.patient import ALSFRSRScore, Patient, ALSTrajectory


# ---------------------------------------------------------------------------
# ALSFRSRScore
# ---------------------------------------------------------------------------

class TestALSFRSRScore:
    """Erik's actual ALS-FRS-R values from his most recent assessment."""

    def _eriks_score(self) -> ALSFRSRScore:
        return ALSFRSRScore(
            speech=4,
            salivation=4,
            swallowing=4,
            handwriting=4,
            cutting_food=4,
            dressing_hygiene=3,
            turning_in_bed=3,
            walking=3,
            climbing_stairs=2,
            dyspnea=4,
            orthopnea=4,
            respiratory_insufficiency=4,
            assessment_date=date(2025, 6, 1),
        )

    def test_total_is_43(self):
        s = self._eriks_score()
        assert s.total == 43

    def test_bulbar_subscore_is_12(self):
        s = self._eriks_score()
        assert s.bulbar_subscore == 12

    def test_fine_motor_subscore_is_11(self):
        s = self._eriks_score()
        assert s.fine_motor_subscore == 11

    def test_gross_motor_subscore_is_8(self):
        s = self._eriks_score()
        assert s.gross_motor_subscore == 8

    def test_respiratory_subscore_is_12(self):
        s = self._eriks_score()
        assert s.respiratory_subscore == 12

    def test_decline_rate_from_onset_is_negative(self):
        s = self._eriks_score()
        onset = date(2025, 1, 15)
        rate = s.decline_rate_from_onset(onset)
        assert rate < 0

    def test_decline_rate_range(self):
        """5 points lost over 12 months gives ~-0.42/month (within -0.50 to -0.30).

        Assessment date set to 2026-01-15 so that exactly 12 months have
        elapsed since onset 2025-01-15, producing a rate of ~-0.42/month.
        """
        s = ALSFRSRScore(
            speech=4, salivation=4, swallowing=4,
            handwriting=4, cutting_food=4, dressing_hygiene=3,
            turning_in_bed=3, walking=3, climbing_stairs=2,
            dyspnea=4, orthopnea=4, respiratory_insufficiency=4,
            assessment_date=date(2026, 1, 15),
        )
        onset = date(2025, 1, 15)
        rate = s.decline_rate_from_onset(onset)
        assert -0.50 <= rate <= -0.30, f"Expected -0.50 <= rate <= -0.30, got {rate}"

    def test_all_items_in_valid_range(self):
        """Each item must be 0–4."""
        s = self._eriks_score()
        for field_name in [
            "speech", "salivation", "swallowing",
            "handwriting", "cutting_food", "dressing_hygiene",
            "turning_in_bed", "walking", "climbing_stairs",
            "dyspnea", "orthopnea", "respiratory_insufficiency",
        ]:
            val = getattr(s, field_name)
            assert 0 <= val <= 4, f"{field_name}={val} out of range"

    def test_minimum_score_total(self):
        s = ALSFRSRScore(
            speech=0, salivation=0, swallowing=0,
            handwriting=0, cutting_food=0, dressing_hygiene=0,
            turning_in_bed=0, walking=0, climbing_stairs=0,
            dyspnea=0, orthopnea=0, respiratory_insufficiency=0,
            assessment_date=date(2025, 1, 1),
        )
        assert s.total == 0

    def test_maximum_score_total(self):
        s = ALSFRSRScore(
            speech=4, salivation=4, swallowing=4,
            handwriting=4, cutting_food=4, dressing_hygiene=4,
            turning_in_bed=4, walking=4, climbing_stairs=4,
            dyspnea=4, orthopnea=4, respiratory_insufficiency=4,
            assessment_date=date(2025, 1, 1),
        )
        assert s.total == 48


# ---------------------------------------------------------------------------
# Patient
# ---------------------------------------------------------------------------

class TestPatient:
    def _make_patient(self) -> Patient:
        return Patient(
            id="patient:erik_draper",
            patient_key="erik_draper",
            birth_year=1985,
            sex_at_birth="male",
            family_history_of_als=False,
            family_history_notes="No known family history",
            consent_profiles=["consent:research_v1"],
            preference_profile_ref="preference:erik_draper_v1",
            allergies=["penicillin"],
            medications=["riluzole", "edaravone"],
            comorbidities=[],
        )

    def test_type_is_patient(self):
        p = self._make_patient()
        assert p.type == "Patient"

    def test_privacy_is_phi(self):
        p = self._make_patient()
        assert p.privacy.classification == PrivacyClass.phi

    def test_patient_key_stored(self):
        p = self._make_patient()
        assert p.patient_key == "erik_draper"

    def test_birth_year(self):
        p = self._make_patient()
        assert p.birth_year == 1985

    def test_sex_at_birth(self):
        p = self._make_patient()
        assert p.sex_at_birth == "male"

    def test_family_history_false(self):
        p = self._make_patient()
        assert p.family_history_of_als is False

    def test_consent_profiles_list(self):
        p = self._make_patient()
        assert "consent:research_v1" in p.consent_profiles

    def test_medications_list(self):
        p = self._make_patient()
        assert "riluzole" in p.medications

    def test_allergies_list(self):
        p = self._make_patient()
        assert "penicillin" in p.allergies

    def test_comorbidities_empty_list(self):
        p = self._make_patient()
        assert p.comorbidities == []

    def test_optional_fields_have_defaults(self):
        p = Patient(id="patient:minimal", patient_key="minimal")
        assert p.type == "Patient"
        assert p.privacy.classification == PrivacyClass.phi
        assert p.consent_profiles == []
        assert p.allergies == []
        assert p.medications == []
        assert p.comorbidities == []

    def test_cannot_override_type(self):
        """type must always be 'Patient' regardless of what is passed."""
        p = Patient(id="patient:x", patient_key="x", type="SomethingElse")
        assert p.type == "Patient"

    def test_cannot_override_privacy(self):
        """privacy must always be PHI."""
        from ontology.base import Privacy
        p = Patient(
            id="patient:x",
            patient_key="x",
            privacy=Privacy(classification=PrivacyClass.public),
        )
        assert p.privacy.classification == PrivacyClass.phi


# ---------------------------------------------------------------------------
# ALSTrajectory
# ---------------------------------------------------------------------------

class TestALSTrajectory:
    def _eriks_score(self) -> ALSFRSRScore:
        return ALSFRSRScore(
            speech=4, salivation=4, swallowing=4,
            handwriting=4, cutting_food=4, dressing_hygiene=3,
            turning_in_bed=3, walking=3, climbing_stairs=2,
            dyspnea=4, orthopnea=4, respiratory_insufficiency=4,
            assessment_date=date(2025, 6, 1),
        )

    def _make_trajectory(self) -> ALSTrajectory:
        return ALSTrajectory(
            id="trajectory:erik_draper_v1",
            patient_ref="patient:erik_draper",
            onset_date=date(2025, 1, 15),
            diagnosis_date=date(2025, 3, 10),
            onset_region=ALSOnsetRegion.upper_limb,
            episode_status="active",
            site_of_care_refs=["site:mayo_rochester"],
            etiologic_driver_profile_ref="driver:erik_draper_v1",
            current_state_snapshot_ref="snapshot:erik_draper_latest",
            alsfrs_r_scores=[self._eriks_score()],
            linked_observation_refs=[],
            linked_intervention_refs=["intervention:riluzole_start"],
            linked_outcome_refs=[],
        )

    def test_type_is_als_trajectory(self):
        t = self._make_trajectory()
        assert t.type == "ALSTrajectory"

    def test_privacy_is_phi(self):
        t = self._make_trajectory()
        assert t.privacy.classification == PrivacyClass.phi

    def test_patient_ref(self):
        t = self._make_trajectory()
        assert t.patient_ref == "patient:erik_draper"

    def test_onset_region_upper_limb(self):
        t = self._make_trajectory()
        assert t.onset_region == ALSOnsetRegion.upper_limb

    def test_onset_date(self):
        t = self._make_trajectory()
        assert t.onset_date == date(2025, 1, 15)

    def test_diagnosis_date(self):
        t = self._make_trajectory()
        assert t.diagnosis_date == date(2025, 3, 10)

    def test_alsfrs_r_scores_list(self):
        t = self._make_trajectory()
        assert len(t.alsfrs_r_scores) == 1
        assert t.alsfrs_r_scores[0].total == 43

    def test_linked_observation_refs_empty(self):
        t = self._make_trajectory()
        assert t.linked_observation_refs == []

    def test_linked_intervention_refs(self):
        t = self._make_trajectory()
        assert "intervention:riluzole_start" in t.linked_intervention_refs

    def test_optional_dates_can_be_none(self):
        t = ALSTrajectory(
            id="trajectory:minimal",
            patient_ref="patient:minimal",
            onset_region=ALSOnsetRegion.unknown,
            episode_status="active",
        )
        assert t.onset_date is None
        assert t.diagnosis_date is None

    def test_cannot_override_type(self):
        t = ALSTrajectory(
            id="trajectory:x",
            patient_ref="patient:x",
            onset_region=ALSOnsetRegion.unknown,
            episode_status="active",
            type="Wrong",
        )
        assert t.type == "ALSTrajectory"

    def test_cannot_override_privacy(self):
        from ontology.base import Privacy
        t = ALSTrajectory(
            id="trajectory:x",
            patient_ref="patient:x",
            onset_region=ALSOnsetRegion.unknown,
            episode_status="active",
            privacy=Privacy(classification=PrivacyClass.public),
        )
        assert t.privacy.classification == PrivacyClass.phi

    def test_json_roundtrip(self):
        t = self._make_trajectory()
        serialised = t.model_dump_json()
        restored = ALSTrajectory.model_validate_json(serialised)

        assert restored.id == t.id
        assert restored.type == t.type
        assert restored.patient_ref == t.patient_ref
        assert restored.onset_region == t.onset_region
        assert restored.onset_date == t.onset_date
        assert len(restored.alsfrs_r_scores) == 1
        assert restored.alsfrs_r_scores[0].total == 43
        assert restored.privacy.classification == PrivacyClass.phi
