"""Build the canonical Patient, ALSTrajectory, and Observations for Erik Draper.

Usage::

    from ingestion.patient_builder import build_erik_draper

    patient, trajectory, observations = build_erik_draper()
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from ontology.base import Provenance
from ontology.enums import ALSOnsetRegion, ObservationKind, SourceSystem
from ontology.observation import (
    EMGFinding,
    ImagingFinding,
    LabResult,
    Observation,
    PhysicalExamFinding,
    RespiratoryMetric,
)
from ontology.patient import ALSFRSRScore, ALSTrajectory, Patient

from ingestion.lab_results import parse_lab_panel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBJECT_REF = "patient:erik_draper"
_EHR = Provenance(source_system=SourceSystem.ehr)


# ---------------------------------------------------------------------------
# Lab data
# ---------------------------------------------------------------------------

_FEB_2026_LABS: list[dict] = [
    {"name": "NfL Plasma", "value": 5.82, "unit": "pg/mL",
     "ref_low": 0.0, "ref_high": 3.65, "date": "2026-02-20"},
    {"name": "CK", "value": 200.0, "unit": "U/L",
     "ref_low": 51.0, "ref_high": 298.0, "date": "2026-02-20"},
    {"name": "B12", "value": 603.0, "unit": "pg/mL",
     "ref_low": 232.0, "ref_high": 1245.0, "date": "2026-02-20"},
    {"name": "Folate", "value": 11.2, "unit": "ng/mL",
     "ref_low": 4.7, "ref_high": None, "date": "2026-02-20"},
    {"name": "Copper", "value": 95.0, "unit": "ug/dL",
     "ref_low": 70.0, "ref_high": 140.0, "date": "2026-02-20"},
    {"name": "Sed Rate", "value": 2.0, "unit": "mm/hr",
     "ref_low": 0.0, "ref_high": 15.0, "date": "2026-02-20"},
    {"name": "WBC", "value": 8.92, "unit": "k/uL",
     "ref_low": 3.7, "ref_high": 11.0, "date": "2026-02-20"},
    {"name": "Hemoglobin", "value": 15.2, "unit": "g/dL",
     "ref_low": 13.0, "ref_high": 17.0, "date": "2026-02-20"},
    {"name": "Hematocrit", "value": 46.5, "unit": "%",
     "ref_low": 39.0, "ref_high": 51.0, "date": "2026-02-20"},
    {"name": "Platelets", "value": 366.0, "unit": "k/uL",
     "ref_low": 150.0, "ref_high": 400.0, "date": "2026-02-20"},
    {"name": "Glucose", "value": 128.0, "unit": "mg/dL",
     "ref_low": 74.0, "ref_high": 99.0, "date": "2026-02-20"},
    {"name": "BUN", "value": 21.0, "unit": "mg/dL",
     "ref_low": 9.0, "ref_high": 24.0, "date": "2026-02-20"},
    {"name": "Creatinine", "value": 1.06, "unit": "mg/dL",
     "ref_low": 0.73, "ref_high": 1.22, "date": "2026-02-20"},
    {"name": "eGFR", "value": 77.0, "unit": "mL/min/1.73m2",
     "ref_low": 60.0, "ref_high": None, "date": "2026-02-20"},
    {"name": "Sodium", "value": 139.0, "unit": "mEq/L",
     "ref_low": 136.0, "ref_high": 145.0, "date": "2026-02-20"},
    {"name": "Potassium", "value": 4.3, "unit": "mEq/L",
     "ref_low": 3.5, "ref_high": 5.0, "date": "2026-02-20"},
    {"name": "AST", "value": 19.0, "unit": "U/L",
     "ref_low": 10.0, "ref_high": 40.0, "date": "2026-02-20"},
    {"name": "ALT", "value": 28.0, "unit": "U/L",
     "ref_low": 7.0, "ref_high": 56.0, "date": "2026-02-20"},
    {"name": "Albumin", "value": 4.5, "unit": "g/dL",
     "ref_low": 3.4, "ref_high": 5.4, "date": "2026-02-20"},
    {"name": "Calcium", "value": 9.2, "unit": "mg/dL",
     "ref_low": 8.5, "ref_high": 10.5, "date": "2026-02-20"},
]

_JUN_2025_LABS: list[dict] = [
    {"name": "A1c", "value": 5.7, "unit": "%",
     "ref_low": 4.3, "ref_high": 5.6, "date": "2025-06-09"},
    {"name": "TSH", "value": 1.38, "unit": "mIU/L",
     "ref_low": 0.27, "ref_high": 4.2, "date": "2025-06-09"},
    {"name": "Total Cholesterol", "value": 143.0, "unit": "mg/dL",
     "ref_low": None, "ref_high": 200.0, "date": "2025-06-09"},
    {"name": "HDL", "value": 36.0, "unit": "mg/dL",
     "ref_low": 39.0, "ref_high": None, "date": "2025-06-09"},
    {"name": "LDL", "value": 93.0, "unit": "mg/dL",
     "ref_low": None, "ref_high": 100.0, "date": "2025-06-09"},
    {"name": "Triglycerides", "value": 72.0, "unit": "mg/dL",
     "ref_low": None, "ref_high": 150.0, "date": "2025-06-09"},
    {"name": "PSA", "value": 1.76, "unit": "ng/mL",
     "ref_low": None, "ref_high": 2.6, "date": "2025-06-09"},
]


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _build_emg_observations() -> list[Observation]:
    """Two EMG observations: Jun 2025 (outside) and Mar 2026 (CC)."""
    emg_jun = EMGFinding(
        study_date=date(2025, 6, 9),
        summary=(
            "Positive sharp waves in L vastus lateralis and tibialis anterior, "
            "reduced recruitment. Low CMAPs: L peroneal 0.6mV, L tibial 1.6mV. "
            "Interpreted as L4-L5 radiculopathy."
        ),
        performing_physician="Precision Orthopaedic Specialties",
        regions_with_active_denervation=[
            "left vastus lateralis",
            "left tibialis anterior",
        ],
        nerve_conduction_abnormalities=[
            "L peroneal CMAP 0.6mV (low)",
            "L tibial CMAP 1.6mV (low)",
        ],
        supports_als=False,
    )
    emg_mar = EMGFinding(
        study_date=date(2026, 3, 6),
        summary=(
            "Widespread active and chronic motor axon loss changes supportive "
            "of ALS."
        ),
        performing_physician="Georgette Dib, MD",
        regions_with_active_denervation=[
            "widespread — multiple limbs and regions",
        ],
        regions_with_chronic_denervation=[
            "widespread — multiple limbs and regions",
        ],
        supports_als=True,
    )
    return [
        Observation(
            id="obs:emg:outside_emg:2025-06-09",
            subject_ref=SUBJECT_REF,
            observation_kind=ObservationKind.emg_feature,
            name="EMG — Precision Orthopaedic",
            emg_finding=emg_jun,
            provenance=_EHR,
        ),
        Observation(
            id="obs:emg:cleveland_clinic:2026-03-06",
            subject_ref=SUBJECT_REF,
            observation_kind=ObservationKind.emg_feature,
            name="EMG — Cleveland Clinic (ALS confirmation)",
            emg_finding=emg_mar,
            provenance=_EHR,
        ),
    ]


def _build_spirometry_observation() -> Observation:
    rm = RespiratoryMetric(
        measurement_date=date(2026, 3, 9),
        fvc_percent_predicted=100.0,
        fvc_liters_sitting=5.0,
        fvc_liters_supine=4.8,
    )
    return Observation(
        id="obs:respiratory:spirometry:2026-03-09",
        subject_ref=SUBJECT_REF,
        observation_kind=ObservationKind.respiratory_metric,
        name="Spirometry — FVC",
        respiratory_metric=rm,
        provenance=_EHR,
    )


def _build_imaging_observations() -> list[Observation]:
    brain = ImagingFinding(
        study_date=date(2026, 2, 21),
        modality="MRI Brain",
        summary=(
            "No motor pathway signal changes. Incidental 1.8x1.1cm fourth "
            "ventricle lesion (possible ependymoma). Microvascular changes. "
            "Mild volume loss."
        ),
        findings=[
            "No motor pathway signal abnormality",
            "Microvascular ischemic changes",
            "Mild volume loss",
        ],
        incidental_findings=[
            "1.8x1.1cm fourth ventricle lesion — possible ependymoma",
        ],
        als_relevant=False,
    )
    cervical = ImagingFinding(
        study_date=date(2026, 2, 21),
        modality="MRI Cervical Spine",
        summary=(
            "Normal cord signal. No myelopathy. Multi-level degenerative "
            "changes, up to severe foraminal stenoses. Multinodular thyroid "
            "incidental finding."
        ),
        findings=[
            "Normal spinal cord signal intensity",
            "No myelopathy",
            "Multi-level degenerative disc disease",
            "Severe foraminal stenosis at multiple levels",
        ],
        incidental_findings=[
            "Multinodular thyroid",
        ],
        als_relevant=False,
    )
    return [
        Observation(
            id="obs:imaging:mri_brain:2026-02-21",
            subject_ref=SUBJECT_REF,
            observation_kind=ObservationKind.imaging_finding,
            name="MRI Brain",
            imaging_finding=brain,
            provenance=_EHR,
        ),
        Observation(
            id="obs:imaging:mri_cervical:2026-02-21",
            subject_ref=SUBJECT_REF,
            observation_kind=ObservationKind.imaging_finding,
            name="MRI Cervical Spine",
            imaging_finding=cervical,
            provenance=_EHR,
        ),
    ]


def _build_physical_exam_observations() -> list[Observation]:
    """Physical exam findings from Feb 6, 2026 neuromuscular evaluation."""
    exam_date = date(2026, 2, 6)
    findings: list[Observation] = []

    # Motor findings
    motor_items = [
        ("left", "hip flexion", "4+"),
        ("left", "knee extension", "4"),
        ("left", "ankle dorsiflexion", "4+"),
        ("right", "hip flexion", "5-"),
        ("right", "knee extension", "4+"),
        ("bilateral", "upper extremity", "5/5"),
    ]
    for lat, region, val in motor_items:
        slug = region.lower().replace(" ", "_")
        findings.append(Observation(
            id=f"obs:exam:motor_{slug}_{lat}:{exam_date.isoformat()}",
            subject_ref=SUBJECT_REF,
            observation_kind=ObservationKind.physical_exam_finding,
            name=f"Motor — {region} ({lat})",
            physical_exam_finding=PhysicalExamFinding(
                exam_date=exam_date,
                category="motor",
                region=region,
                finding=f"Strength {val}",
                laterality=lat,
                value=val,
            ),
            provenance=_EHR,
        ))

    # Reflex findings
    reflex_items = [
        ("bilateral", "plantars", "upgoing (Babinski positive)"),
        ("left", "biceps/triceps/brachioradialis", "3+"),
        ("right", "Achilles", "4"),
        ("right", "ankle", "non-sustained clonus"),
    ]
    for lat, region, val in reflex_items:
        slug = region.lower().replace("/", "_").replace(" ", "_")
        findings.append(Observation(
            id=f"obs:exam:reflex_{slug}_{lat}:{exam_date.isoformat()}",
            subject_ref=SUBJECT_REF,
            observation_kind=ObservationKind.physical_exam_finding,
            name=f"Reflex — {region} ({lat})",
            physical_exam_finding=PhysicalExamFinding(
                exam_date=exam_date,
                category="reflexes",
                region=region,
                finding=val,
                laterality=lat,
                value=val,
            ),
            provenance=_EHR,
        ))

    # Tone
    tone_items = [
        ("bilateral", "lower extremity", "A2 (moderate spasticity)"),
        ("left", "upper extremity", "A1 (mild spasticity)"),
    ]
    for lat, region, val in tone_items:
        slug = region.lower().replace(" ", "_")
        findings.append(Observation(
            id=f"obs:exam:tone_{slug}_{lat}:{exam_date.isoformat()}",
            subject_ref=SUBJECT_REF,
            observation_kind=ObservationKind.physical_exam_finding,
            name=f"Tone — {region} ({lat})",
            physical_exam_finding=PhysicalExamFinding(
                exam_date=exam_date,
                category="tone",
                region=region,
                finding=val,
                laterality=lat,
                value=val,
            ),
            provenance=_EHR,
        ))

    # Gait
    findings.append(Observation(
        id=f"obs:exam:gait:{exam_date.isoformat()}",
        subject_ref=SUBJECT_REF,
        observation_kind=ObservationKind.physical_exam_finding,
        name="Gait assessment",
        physical_exam_finding=PhysicalExamFinding(
            exam_date=exam_date,
            category="gait",
            region="general",
            finding="Wide-base spastic gait, left foot drop, uses 2 hiking poles",
            laterality="bilateral",
        ),
        provenance=_EHR,
    ))

    # Sensation
    findings.append(Observation(
        id=f"obs:exam:sensation_vibration_left_foot:{exam_date.isoformat()}",
        subject_ref=SUBJECT_REF,
        observation_kind=ObservationKind.physical_exam_finding,
        name="Sensation — vibration (left foot)",
        physical_exam_finding=PhysicalExamFinding(
            exam_date=exam_date,
            category="sensation",
            region="left foot",
            finding="Vibration reduced; absent at great toe",
            laterality="left",
        ),
        provenance=_EHR,
    ))

    return findings


def _build_weight_observations() -> list[Observation]:
    return [
        Observation(
            id="obs:weight:2025-06-09",
            subject_ref=SUBJECT_REF,
            observation_kind=ObservationKind.weight_measurement,
            name="Body weight",
            value=117.0,
            unit="kg",
            provenance=_EHR,
        ),
        Observation(
            id="obs:weight:2026-02-06",
            subject_ref=SUBJECT_REF,
            observation_kind=ObservationKind.weight_measurement,
            name="Body weight",
            value=111.1,
            unit="kg",
            provenance=_EHR,
        ),
        Observation(
            id="obs:weight:2026-03-09",
            subject_ref=SUBJECT_REF,
            observation_kind=ObservationKind.weight_measurement,
            name="Body weight",
            value=118.2,
            unit="kg",
            provenance=_EHR,
        ),
    ]


def _build_vitals_observation() -> Observation:
    return Observation(
        id="obs:vitals:2026-02-06",
        subject_ref=SUBJECT_REF,
        observation_kind=ObservationKind.vital_sign,
        name="Vital signs",
        value_str="BP 129/95, Pulse 95, SpO2 94%",
        provenance=_EHR,
    )


def _build_medication_observation() -> Observation:
    return Observation(
        id="obs:medication:riluzole_start:2026-03-06",
        subject_ref=SUBJECT_REF,
        observation_kind=ObservationKind.medication_event,
        name="Riluzole initiation",
        value_str="Riluzole started 2026-03-06",
        provenance=_EHR,
    )


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_erik_draper() -> tuple[Patient, ALSTrajectory, list[Observation]]:
    """Construct the full Erik Draper clinical record.

    Returns
    -------
    tuple[Patient, ALSTrajectory, list[Observation]]
        - Patient with demographics, medications, comorbidities, family hx
        - ALSTrajectory with onset, diagnosis, ALSFRS-R
        - All clinical observations (~48 total)
    """
    # ---------------------------------------------------------------- Patient
    patient = Patient(
        id=SUBJECT_REF,
        patient_key="erik_draper",
        birth_year=1958,
        sex_at_birth="male",
        family_history_of_als=False,
        family_history_notes=(
            "Mother and maternal family with Alzheimer's disease. "
            "No family history of MND/ALS."
        ),
        allergies=[],
        medications=[
            "amlodipine-atorvastatin 10-20mg daily",
            "ramipril 10mg daily",
            "riluzole (started 2026-03-06)",
            "vitamin C 500mg",
            "calcium-D3",
            "glucosamine 2000mg",
            "lysine 500mg",
            "magnesium oxide 400mg",
            "multivitamin",
            "potassium gluconate 600mg",
        ],
        comorbidities=[
            "hypertension",
            "prediabetes (A1c 5.7%)",
            "elevated cholesterol (on statin)",
            "cervical spinal stenosis (C3-C7, no myelopathy)",
            "basal cell carcinoma history",
        ],
    )

    # ------------------------------------------------------------ ALSFRS-R
    alsfrs = ALSFRSRScore(
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
        assessment_date=date(2026, 2, 6),
    )

    # --------------------------------------------------------- Observations
    observations: list[Observation] = []

    # Labs
    observations.extend(parse_lab_panel(_FEB_2026_LABS, SUBJECT_REF))
    observations.extend(parse_lab_panel(_JUN_2025_LABS, SUBJECT_REF))

    # EMG
    observations.extend(_build_emg_observations())

    # Spirometry
    observations.append(_build_spirometry_observation())

    # Imaging
    observations.extend(_build_imaging_observations())

    # Physical exam
    observations.extend(_build_physical_exam_observations())

    # Weight
    observations.extend(_build_weight_observations())

    # Vitals
    observations.append(_build_vitals_observation())

    # Medication event
    observations.append(_build_medication_observation())

    # ---------------------------------------------------------- Trajectory
    trajectory = ALSTrajectory(
        id="trajectory:erik_draper",
        patient_ref=patient.id,
        onset_date=date(2025, 1, 15),
        diagnosis_date=date(2026, 3, 6),
        onset_region=ALSOnsetRegion.lower_limb,
        alsfrs_r_scores=[alsfrs],
        linked_observation_refs=[o.id for o in observations],
    )

    return patient, trajectory, observations
