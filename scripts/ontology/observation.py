"""Observation domain models for the Erik ALS causal research engine.

Includes:
- LabResult            — quantitative or qualitative lab measurement
- EMGFinding           — electromyography study result
- RespiratoryMetric    — pulmonary function measurements
- ImagingFinding       — radiology / neuroimaging result
- PhysicalExamFinding  — clinician examination finding
- Observation          — BaseEnvelope wrapping any of the above
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, computed_field, model_validator

from ontology.base import BaseEnvelope
from ontology.enums import ObservationKind


# ---------------------------------------------------------------------------
# LabResult
# ---------------------------------------------------------------------------

class LabResult(BaseModel):
    """A single quantitative or semi-quantitative laboratory result."""

    name: str
    value: float
    unit: str
    reference_low: Optional[float] = None
    reference_high: Optional[float] = None
    collection_date: date
    method: str
    notes: str = ""

    @computed_field  # type: ignore[misc]
    @property
    def is_high(self) -> bool:
        """True if value strictly exceeds reference_high."""
        if self.reference_high is None:
            return False
        return self.value > self.reference_high

    @computed_field  # type: ignore[misc]
    @property
    def is_low(self) -> bool:
        """True if value is strictly below reference_low."""
        if self.reference_low is None:
            return False
        return self.value < self.reference_low

    @computed_field  # type: ignore[misc]
    @property
    def is_abnormal(self) -> bool:
        """True if is_high or is_low."""
        return self.is_high or self.is_low


# ---------------------------------------------------------------------------
# EMGFinding
# ---------------------------------------------------------------------------

class EMGFinding(BaseModel):
    """Electromyography (EMG) and nerve conduction study result."""

    study_date: date
    summary: str
    performing_physician: str
    regions_with_active_denervation: list[str] = Field(default_factory=list)
    regions_with_chronic_denervation: list[str] = Field(default_factory=list)
    regions_with_reinnervation: list[str] = Field(default_factory=list)
    fasciculation_potentials: list[str] = Field(default_factory=list)
    nerve_conduction_abnormalities: list[str] = Field(default_factory=list)
    supports_als: bool = False
    raw_report_ref: Optional[str] = None


# ---------------------------------------------------------------------------
# RespiratoryMetric
# ---------------------------------------------------------------------------

class RespiratoryMetric(BaseModel):
    """Pulmonary function test (PFT) and respiratory muscle strength metrics."""

    measurement_date: date
    fvc_percent_predicted: Optional[float] = None
    fvc_liters_sitting: Optional[float] = None
    fvc_liters_supine: Optional[float] = None
    fev1_liters: Optional[float] = None
    snip: Optional[float] = None   # Sniff Nasal Inspiratory Pressure (cmH2O)
    mip: Optional[float] = None    # Maximal Inspiratory Pressure (cmH2O)
    notes: str = ""

    @computed_field  # type: ignore[misc]
    @property
    def supine_drop_percent(self) -> Optional[float]:
        """Percentage drop in FVC from sitting to supine.

        Formula: ((sitting - supine) / sitting) * 100

        Returns None if either value is absent or sitting is zero.
        """
        if self.fvc_liters_sitting is None or self.fvc_liters_supine is None:
            return None
        if self.fvc_liters_sitting == 0:
            return None
        return (
            (self.fvc_liters_sitting - self.fvc_liters_supine)
            / self.fvc_liters_sitting
        ) * 100


# ---------------------------------------------------------------------------
# ImagingFinding
# ---------------------------------------------------------------------------

class ImagingFinding(BaseModel):
    """Radiological or neuroimaging study finding."""

    study_date: date
    modality: str
    summary: str
    findings: list[str] = Field(default_factory=list)
    incidental_findings: list[str] = Field(default_factory=list)
    als_relevant: bool = False
    raw_report_ref: Optional[str] = None


# ---------------------------------------------------------------------------
# PhysicalExamFinding
# ---------------------------------------------------------------------------

class PhysicalExamFinding(BaseModel):
    """A single finding from a structured physical examination."""

    exam_date: date
    category: str
    region: str
    finding: str
    laterality: str
    value: Optional[str] = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Observation (envelope)
# ---------------------------------------------------------------------------

class Observation(BaseEnvelope):
    """Universal observation envelope.

    ``type`` is locked to ``"Observation"``.  Exactly one or zero of the
    typed sub-object fields should be populated; the generic ``value`` /
    ``value_str`` / ``unit`` fields can complement any sub-object.
    """

    # Lock type with a default so callers need not pass it
    type: str = Field(default="Observation", min_length=1)

    # Required linkage
    subject_ref: str
    observation_kind: ObservationKind
    name: str

    # Optional context
    measurement_method: Optional[str] = None
    specimen_or_context: Optional[str] = None
    source_ref: Optional[str] = None

    # Typed sub-objects — at most one should be set per observation
    lab_result: Optional[LabResult] = None
    emg_finding: Optional[EMGFinding] = None
    respiratory_metric: Optional[RespiratoryMetric] = None
    imaging_finding: Optional[ImagingFinding] = None
    physical_exam_finding: Optional[PhysicalExamFinding] = None

    # Generic scalar fields (usable with or without sub-object)
    value: Optional[float] = None
    value_str: Optional[str] = None
    unit: Optional[str] = None

    @model_validator(mode="after")
    def _enforce_observation_type(self) -> "Observation":
        object.__setattr__(self, "type", "Observation")
        return self
