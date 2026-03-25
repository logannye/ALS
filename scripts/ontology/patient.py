"""Patient-centric domain models for the Erik ALS causal research engine.

Includes:
- ALSFRSRScore  — 12-item ALS Functional Rating Scale – Revised
- Patient       — PHI-bearing patient envelope
- ALSTrajectory — Longitudinal disease trajectory for one patient
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import Field, computed_field, model_validator

from ontology.base import BaseEnvelope, Privacy
from ontology.enums import ALSOnsetRegion, PrivacyClass
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# ALSFRSRScore
# ---------------------------------------------------------------------------

class ALSFRSRScore(BaseModel):
    """ALS Functional Rating Scale – Revised (ALSFRS-R).

    12 items each scored 0–4 (4 = normal, 0 = maximal impairment).
    Maximum total = 48.
    """

    # Bulbar
    speech: int = Field(..., ge=0, le=4)
    salivation: int = Field(..., ge=0, le=4)
    swallowing: int = Field(..., ge=0, le=4)

    # Fine motor
    handwriting: int = Field(..., ge=0, le=4)
    cutting_food: int = Field(..., ge=0, le=4)
    dressing_hygiene: int = Field(..., ge=0, le=4)

    # Gross motor
    turning_in_bed: int = Field(..., ge=0, le=4)
    walking: int = Field(..., ge=0, le=4)
    climbing_stairs: int = Field(..., ge=0, le=4)

    # Respiratory
    dyspnea: int = Field(..., ge=0, le=4)
    orthopnea: int = Field(..., ge=0, le=4)
    respiratory_insufficiency: int = Field(..., ge=0, le=4)

    assessment_date: date

    # ------------------------------------------------------------------
    # Computed subscores
    # ------------------------------------------------------------------

    @computed_field  # type: ignore[misc]
    @property
    def bulbar_subscore(self) -> int:
        return self.speech + self.salivation + self.swallowing

    @computed_field  # type: ignore[misc]
    @property
    def fine_motor_subscore(self) -> int:
        return self.handwriting + self.cutting_food + self.dressing_hygiene

    @computed_field  # type: ignore[misc]
    @property
    def gross_motor_subscore(self) -> int:
        return self.turning_in_bed + self.walking + self.climbing_stairs

    @computed_field  # type: ignore[misc]
    @property
    def respiratory_subscore(self) -> int:
        return self.dyspnea + self.orthopnea + self.respiratory_insufficiency

    @computed_field  # type: ignore[misc]
    @property
    def total(self) -> int:
        return (
            self.bulbar_subscore
            + self.fine_motor_subscore
            + self.gross_motor_subscore
            + self.respiratory_subscore
        )

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def decline_rate_from_onset(self, onset_date: date) -> float:
        """Return negative decline rate in ALSFRS-R points per month.

        Formula: -(48 - total) / months_since_onset

        A lower (more negative) value indicates faster decline.
        Returns 0.0 if onset_date equals assessment_date to avoid division
        by zero.
        """
        days = (self.assessment_date - onset_date).days
        if days <= 0:
            return 0.0
        months = days / 30.4375  # average days per month
        points_lost = 48 - self.total
        return -(points_lost / months)


# ---------------------------------------------------------------------------
# Patient
# ---------------------------------------------------------------------------

class Patient(BaseEnvelope):
    """A single patient record.  Always PHI-classified.

    ``type`` is locked to ``"Patient"`` and ``privacy`` is locked to
    ``PrivacyClass.phi`` via model_validator so callers cannot accidentally
    downgrade sensitivity.
    """

    # Lock type with a default so callers need not pass it
    type: str = Field(default="Patient", min_length=1)

    # Domain fields
    patient_key: str = ""
    birth_year: Optional[int] = None
    sex_at_birth: Optional[str] = None
    family_history_of_als: bool = False
    family_history_notes: Optional[str] = None
    consent_profiles: list[str] = Field(default_factory=list)
    preference_profile_ref: Optional[str] = None
    allergies: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    comorbidities: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _enforce_patient_invariants(self) -> "Patient":
        object.__setattr__(self, "type", "Patient")
        self.privacy = Privacy(classification=PrivacyClass.phi)
        return self


# ---------------------------------------------------------------------------
# ALSTrajectory
# ---------------------------------------------------------------------------

class ALSTrajectory(BaseEnvelope):
    """Longitudinal ALS disease trajectory for one patient.

    Always PHI-classified.  Links a patient to their clinical timeline,
    ALSFRS-R assessments, observations, interventions, and outcomes.
    """

    # Lock type with a default so callers need not pass it
    type: str = Field(default="ALSTrajectory", min_length=1)

    # Core linkage
    patient_ref: str
    onset_date: Optional[date] = None
    diagnosis_date: Optional[date] = None
    onset_region: ALSOnsetRegion = ALSOnsetRegion.unknown
    episode_status: str = "active"

    # Care / driver references
    site_of_care_refs: list[str] = Field(default_factory=list)
    etiologic_driver_profile_ref: Optional[str] = None
    current_state_snapshot_ref: Optional[str] = None

    # ALSFRS-R history
    alsfrs_r_scores: list[ALSFRSRScore] = Field(default_factory=list)

    # Cross-links
    linked_observation_refs: list[str] = Field(default_factory=list)
    linked_intervention_refs: list[str] = Field(default_factory=list)
    linked_outcome_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _enforce_trajectory_invariants(self) -> "ALSTrajectory":
        object.__setattr__(self, "type", "ALSTrajectory")
        self.privacy = Privacy(classification=PrivacyClass.phi)
        return self
