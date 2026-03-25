"""Tests for ontology.observation — Observation envelope and sub-models."""
import json
import pytest
from datetime import date

from ontology.enums import ObservationKind
from ontology.observation import (
    EMGFinding,
    ImagingFinding,
    LabResult,
    Observation,
    PhysicalExamFinding,
    RespiratoryMetric,
)


# ---------------------------------------------------------------------------
# LabResult
# ---------------------------------------------------------------------------

class TestLabResult:
    def test_nfl_is_abnormal_high(self):
        """NfL 5.82 with reference 0–3.65 → is_high and is_abnormal."""
        lr = LabResult(
            name="Neurofilament Light Chain (NfL)",
            value=5.82,
            unit="pg/mL",
            reference_low=0.0,
            reference_high=3.65,
            collection_date=date(2025, 5, 10),
            method="Simoa immunoassay",
            notes="Elevated, consistent with active neurodegeneration",
        )
        assert lr.is_high is True
        assert lr.is_abnormal is True
        assert lr.is_low is False

    def test_ck_is_normal(self):
        """CK 200 with reference 51–298 → is_abnormal=False."""
        lr = LabResult(
            name="Creatine Kinase (CK)",
            value=200.0,
            unit="U/L",
            reference_low=51.0,
            reference_high=298.0,
            collection_date=date(2025, 5, 10),
            method="Enzymatic colorimetric",
            notes="Within normal limits",
        )
        assert lr.is_high is False
        assert lr.is_low is False
        assert lr.is_abnormal is False

    def test_low_value_is_low_and_abnormal(self):
        lr = LabResult(
            name="Sodium",
            value=130.0,
            unit="mEq/L",
            reference_low=135.0,
            reference_high=145.0,
            collection_date=date(2025, 5, 1),
            method="Ion-selective electrode",
            notes="Hyponatremia",
        )
        assert lr.is_low is True
        assert lr.is_high is False
        assert lr.is_abnormal is True

    def test_no_reference_range_not_abnormal(self):
        """Without reference bounds, is_high/is_low/is_abnormal should be False."""
        lr = LabResult(
            name="Custom marker",
            value=42.0,
            unit="arbitrary",
            collection_date=date(2025, 1, 1),
            method="internal",
            notes="",
        )
        assert lr.is_high is False
        assert lr.is_low is False
        assert lr.is_abnormal is False

    def test_exact_reference_high_boundary_not_high(self):
        """Value == reference_high is NOT high (strictly greater than)."""
        lr = LabResult(
            name="X",
            value=3.65,
            unit="pg/mL",
            reference_low=0.0,
            reference_high=3.65,
            collection_date=date(2025, 1, 1),
            method="test",
            notes="",
        )
        assert lr.is_high is False

    def test_exact_reference_low_boundary_not_low(self):
        """Value == reference_low is NOT low (strictly less than)."""
        lr = LabResult(
            name="X",
            value=51.0,
            unit="U/L",
            reference_low=51.0,
            reference_high=298.0,
            collection_date=date(2025, 1, 1),
            method="test",
            notes="",
        )
        assert lr.is_low is False


# ---------------------------------------------------------------------------
# Observation wrapping a LabResult
# ---------------------------------------------------------------------------

class TestObservationWithLabResult:
    def _nfl_lab(self) -> LabResult:
        return LabResult(
            name="Neurofilament Light Chain (NfL)",
            value=5.82,
            unit="pg/mL",
            reference_low=0.0,
            reference_high=3.65,
            collection_date=date(2025, 5, 10),
            method="Simoa immunoassay",
            notes="",
        )

    def test_observation_wraps_lab_result(self):
        obs = Observation(
            id="observation:nfl_2025_05_10",
            subject_ref="patient:erik_draper",
            observation_kind=ObservationKind.lab_result,
            name="NfL plasma",
            measurement_method="Simoa immunoassay",
            specimen_or_context="plasma",
            source_ref="lab:mayo_lims_001",
            lab_result=self._nfl_lab(),
        )
        assert obs.type == "Observation"
        assert obs.observation_kind == ObservationKind.lab_result
        assert obs.lab_result is not None
        assert obs.lab_result.is_high is True
        assert obs.lab_result.value == pytest.approx(5.82)

    def test_observation_has_no_emg_when_lab(self):
        obs = Observation(
            id="observation:nfl_test",
            subject_ref="patient:erik_draper",
            observation_kind=ObservationKind.lab_result,
            name="NfL",
            lab_result=self._nfl_lab(),
        )
        assert obs.emg_finding is None
        assert obs.respiratory_metric is None

    def test_observation_json_roundtrip(self):
        obs = Observation(
            id="observation:nfl_roundtrip",
            subject_ref="patient:erik_draper",
            observation_kind=ObservationKind.lab_result,
            name="NfL plasma",
            measurement_method="Simoa",
            specimen_or_context="plasma",
            source_ref="lab:001",
            lab_result=self._nfl_lab(),
            value=5.82,
            unit="pg/mL",
        )
        serialised = obs.model_dump_json()
        restored = Observation.model_validate_json(serialised)

        assert restored.id == obs.id
        assert restored.type == "Observation"
        assert restored.observation_kind == ObservationKind.lab_result
        assert restored.lab_result is not None
        assert restored.lab_result.value == pytest.approx(5.82)
        assert restored.lab_result.is_high is True


# ---------------------------------------------------------------------------
# EMGFinding
# ---------------------------------------------------------------------------

class TestEMGFinding:
    def _make_emg(self) -> EMGFinding:
        return EMGFinding(
            study_date=date(2025, 4, 20),
            summary="Active and chronic denervation in multiple limb regions, supporting ALS diagnosis",
            performing_physician="Dr. Sarah Chen",
            regions_with_active_denervation=["left_first_dorsal_interosseous", "right_tibialis_anterior"],
            regions_with_chronic_denervation=["left_biceps", "right_deltoid"],
            regions_with_reinnervation=["left_first_dorsal_interosseous"],
            fasciculation_potentials=["left_first_dorsal_interosseous"],
            nerve_conduction_abnormalities=[],
            supports_als=True,
            raw_report_ref="doc:emg_2025_04_20",
        )

    def test_supports_als_true(self):
        emg = self._make_emg()
        assert emg.supports_als is True

    def test_regions_with_active_denervation(self):
        emg = self._make_emg()
        assert "left_first_dorsal_interosseous" in emg.regions_with_active_denervation

    def test_regions_with_chronic_denervation(self):
        emg = self._make_emg()
        assert len(emg.regions_with_chronic_denervation) == 2

    def test_observation_wraps_emg(self):
        obs = Observation(
            id="observation:emg_2025_04_20",
            subject_ref="patient:erik_draper",
            observation_kind=ObservationKind.emg_feature,
            name="EMG upper and lower limbs",
            emg_finding=self._make_emg(),
        )
        assert obs.emg_finding is not None
        assert obs.emg_finding.supports_als is True
        assert obs.lab_result is None


# ---------------------------------------------------------------------------
# RespiratoryMetric
# ---------------------------------------------------------------------------

class TestRespiratoryMetric:
    def _make_resp(self) -> RespiratoryMetric:
        return RespiratoryMetric(
            measurement_date=date(2025, 5, 15),
            fvc_percent_predicted=95.0,
            fvc_liters_sitting=5.0,
            fvc_liters_supine=4.8,
            fev1_liters=4.1,
            snip=95.0,
            mip=120.0,
            notes="Mildly reduced supine FVC, monitoring",
        )

    def test_supine_drop_between_0_and_10_percent(self):
        rm = self._make_resp()
        drop = rm.supine_drop_percent
        assert drop is not None
        assert 0.0 <= drop <= 10.0, f"Expected 0-10%, got {drop}"

    def test_supine_drop_calculation(self):
        """FVC sitting=5.0, supine=4.8: drop = (5.0-4.8)/5.0 * 100 = 4.0%"""
        rm = self._make_resp()
        assert rm.supine_drop_percent == pytest.approx(4.0, abs=0.01)

    def test_supine_drop_none_when_missing_values(self):
        rm = RespiratoryMetric(
            measurement_date=date(2025, 1, 1),
            fvc_percent_predicted=95.0,
            notes="",
        )
        assert rm.supine_drop_percent is None

    def test_supine_drop_none_when_only_sitting(self):
        rm = RespiratoryMetric(
            measurement_date=date(2025, 1, 1),
            fvc_liters_sitting=5.0,
            notes="",
        )
        assert rm.supine_drop_percent is None

    def test_observation_wraps_respiratory_metric(self):
        obs = Observation(
            id="observation:resp_2025_05_15",
            subject_ref="patient:erik_draper",
            observation_kind=ObservationKind.respiratory_metric,
            name="Pulmonary function test",
            respiratory_metric=self._make_resp(),
        )
        assert obs.respiratory_metric is not None
        assert obs.respiratory_metric.fvc_percent_predicted == pytest.approx(95.0)


# ---------------------------------------------------------------------------
# ImagingFinding
# ---------------------------------------------------------------------------

class TestImagingFinding:
    def test_basic_creation(self):
        img = ImagingFinding(
            study_date=date(2025, 2, 1),
            modality="MRI Brain",
            summary="Bilateral corticospinal tract T2 signal changes",
            findings=["bilateral CST hyperintensity on FLAIR"],
            incidental_findings=[],
            als_relevant=True,
            raw_report_ref="doc:mri_2025_02_01",
        )
        assert img.als_relevant is True
        assert len(img.findings) == 1

    def test_observation_wraps_imaging(self):
        img = ImagingFinding(
            study_date=date(2025, 2, 1),
            modality="MRI Brain",
            summary="CST changes",
            findings=["CST hyperintensity"],
            incidental_findings=[],
            als_relevant=True,
            raw_report_ref="doc:mri_001",
        )
        obs = Observation(
            id="observation:mri_001",
            subject_ref="patient:erik_draper",
            observation_kind=ObservationKind.imaging_finding,
            name="Brain MRI",
            imaging_finding=img,
        )
        assert obs.imaging_finding is not None
        assert obs.imaging_finding.als_relevant is True


# ---------------------------------------------------------------------------
# PhysicalExamFinding
# ---------------------------------------------------------------------------

class TestPhysicalExamFinding:
    def test_basic_creation(self):
        pef = PhysicalExamFinding(
            exam_date=date(2025, 3, 10),
            category="motor",
            region="upper_limb",
            finding="fasciculations",
            laterality="bilateral",
            value="3+",
            notes="Prominent fasciculations in both hands",
        )
        assert pef.finding == "fasciculations"
        assert pef.laterality == "bilateral"

    def test_optional_value_is_none(self):
        pef = PhysicalExamFinding(
            exam_date=date(2025, 1, 1),
            category="reflexes",
            region="lower_limb",
            finding="hyperreflexia",
            laterality="bilateral",
            notes="",
        )
        assert pef.value is None

    def test_observation_wraps_exam_finding(self):
        pef = PhysicalExamFinding(
            exam_date=date(2025, 3, 10),
            category="motor",
            region="upper_limb",
            finding="fasciculations",
            laterality="bilateral",
            notes="",
        )
        obs = Observation(
            id="observation:exam_001",
            subject_ref="patient:erik_draper",
            observation_kind=ObservationKind.physical_exam_finding,
            name="Physical exam — motor",
            physical_exam_finding=pef,
        )
        assert obs.physical_exam_finding is not None
        assert obs.physical_exam_finding.finding == "fasciculations"


# ---------------------------------------------------------------------------
# Observation generic fields
# ---------------------------------------------------------------------------

class TestObservationGenericFields:
    def test_observation_with_generic_value(self):
        obs = Observation(
            id="observation:generic_001",
            subject_ref="patient:erik_draper",
            observation_kind=ObservationKind.vital_sign,
            name="Weight",
            value=78.5,
            value_str="78.5 kg",
            unit="kg",
        )
        assert obs.value == pytest.approx(78.5)
        assert obs.value_str == "78.5 kg"
        assert obs.unit == "kg"

    def test_observation_minimal_creation(self):
        obs = Observation(
            id="observation:min_001",
            subject_ref="patient:erik_draper",
            observation_kind=ObservationKind.workflow_signal,
            name="Trigger",
        )
        assert obs.type == "Observation"
        assert obs.lab_result is None
        assert obs.emg_finding is None
        assert obs.respiratory_metric is None
        assert obs.imaging_finding is None
        assert obs.physical_exam_finding is None
        assert obs.value is None
        assert obs.value_str is None

    def test_cannot_override_type(self):
        obs = Observation(
            id="observation:x",
            subject_ref="patient:x",
            observation_kind=ObservationKind.lab_result,
            name="X",
            type="Wrong",
        )
        assert obs.type == "Observation"
