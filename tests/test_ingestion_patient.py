"""Tests for scripts/ingestion/patient_builder.py — Erik Draper patient build."""

from datetime import date

import pytest

from ingestion.patient_builder import build_erik_draper
from ontology.patient import Patient, ALSTrajectory, ALSFRSRScore
from ontology.observation import Observation
from ontology.enums import ALSOnsetRegion, ObservationKind


class TestBuildErikDraper:
    """build_erik_draper returns a fully populated (Patient, ALSTrajectory, [Obs])."""

    @pytest.fixture(scope="class")
    def erik(self):
        return build_erik_draper()

    @pytest.fixture(scope="class")
    def patient(self, erik):
        return erik[0]

    @pytest.fixture(scope="class")
    def trajectory(self, erik):
        return erik[1]

    @pytest.fixture(scope="class")
    def observations(self, erik):
        return erik[2]

    # ------------------------------------------------------------------ Patient
    def test_patient_type(self, patient):
        assert isinstance(patient, Patient)

    def test_patient_birth_year(self, patient):
        assert patient.birth_year == 1958

    def test_patient_sex(self, patient):
        assert patient.sex_at_birth == "male"

    def test_no_als_family_history(self, patient):
        assert patient.family_history_of_als is False

    def test_family_history_notes(self, patient):
        assert "Alzheimer" in patient.family_history_notes

    def test_no_allergies(self, patient):
        # "No known allergies"
        assert patient.allergies == []

    def test_medications_present(self, patient):
        assert len(patient.medications) >= 9
        med_lower = [m.lower() for m in patient.medications]
        assert any("riluzole" in m for m in med_lower)
        assert any("ramipril" in m for m in med_lower)

    def test_comorbidities(self, patient):
        assert len(patient.comorbidities) >= 3
        combo_lower = [c.lower() for c in patient.comorbidities]
        assert any("hypertension" in c for c in combo_lower)

    # -------------------------------------------------------------- Trajectory
    def test_trajectory_type(self, trajectory):
        assert isinstance(trajectory, ALSTrajectory)

    def test_onset_region(self, trajectory):
        assert trajectory.onset_region == ALSOnsetRegion.lower_limb

    def test_onset_date(self, trajectory):
        assert trajectory.onset_date == date(2025, 1, 15)

    def test_diagnosis_date(self, trajectory):
        assert trajectory.diagnosis_date == date(2026, 3, 6)

    def test_patient_ref_link(self, trajectory, patient):
        assert trajectory.patient_ref == patient.id

    # -------------------------------------------------------------- ALSFRS-R
    def test_alsfrs_total(self, trajectory):
        assert len(trajectory.alsfrs_r_scores) >= 1
        score = trajectory.alsfrs_r_scores[0]
        assert score.total == 43

    def test_alsfrs_subscores(self, trajectory):
        score = trajectory.alsfrs_r_scores[0]
        assert score.bulbar_subscore == 12
        assert score.fine_motor_subscore == 11
        assert score.gross_motor_subscore == 8
        assert score.respiratory_subscore == 12

    # ----------------------------------------------------------- Observations
    def test_observations_are_observations(self, observations):
        assert all(isinstance(o, Observation) for o in observations)

    def test_has_lab_results(self, observations):
        labs = [o for o in observations
                if o.observation_kind == ObservationKind.lab_result]
        assert len(labs) >= 20  # ~20 Feb + ~7 Jun labs

    def test_has_emg(self, observations):
        emgs = [o for o in observations
                if o.observation_kind == ObservationKind.emg_feature]
        assert len(emgs) >= 2

    def test_has_imaging(self, observations):
        imgs = [o for o in observations
                if o.observation_kind == ObservationKind.imaging_finding]
        assert len(imgs) >= 2

    def test_has_respiratory(self, observations):
        resps = [o for o in observations
                 if o.observation_kind == ObservationKind.respiratory_metric]
        assert len(resps) >= 1

    def test_nfl_flagged_abnormal(self, observations):
        nfl = [o for o in observations
               if o.lab_result is not None and "NfL" in o.lab_result.name]
        assert len(nfl) == 1
        assert nfl[0].lab_result.is_abnormal is True

    def test_has_physical_exam(self, observations):
        exams = [o for o in observations
                 if o.observation_kind == ObservationKind.physical_exam_finding]
        assert len(exams) >= 10

    def test_has_weight(self, observations):
        weights = [o for o in observations
                   if o.observation_kind == ObservationKind.weight_measurement]
        assert len(weights) >= 2

    def test_has_vital_signs(self, observations):
        vitals = [o for o in observations
                  if o.observation_kind == ObservationKind.vital_sign]
        assert len(vitals) >= 1

    def test_has_medication_event(self, observations):
        meds = [o for o in observations
                if o.observation_kind == ObservationKind.medication_event]
        assert len(meds) >= 1
