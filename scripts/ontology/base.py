"""Base envelope and supporting sub-models for all Erik canonical objects.

Every domain object (Patient, Observation, Intervention, etc.) is a
``BaseEnvelope`` with a typed ``body`` dict.  This keeps the envelope
layer lightweight and schema-forward without coupling it to any single
domain.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from ontology.enums import (
    ConfidenceBand,
    ObjectStatus,
    PrivacyClass,
    SourceSystem,
)


# ---------------------------------------------------------------------------
# TimeFields
# ---------------------------------------------------------------------------

class TimeFields(BaseModel):
    """Temporal metadata for when an observation was made and when it is valid."""

    observed_at: Optional[datetime] = None
    effective_at: Optional[datetime] = None
    recorded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class Provenance(BaseModel):
    """Where did this assertion come from and who made it?"""

    source_system: SourceSystem = SourceSystem.manual
    source_artifact_id: Optional[str] = None
    asserted_by: Optional[str] = None
    trace_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------

class Uncertainty(BaseModel):
    """Epistemic and aleatoric uncertainty metadata."""

    confidence: Optional[float] = None
    confidence_band: Optional[ConfidenceBand] = None
    sources: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------

class Privacy(BaseModel):
    """Privacy classification for HIPAA / governance compliance."""

    classification: PrivacyClass = PrivacyClass.restricted


# ---------------------------------------------------------------------------
# BaseEnvelope
# ---------------------------------------------------------------------------

class BaseEnvelope(BaseModel):
    """Universal object envelope — every canonical Erik object inherits from this.

    The ``body`` field holds domain-specific payload as an open dict so that
    specialised models can extend without modifying the envelope schema.
    """

    id: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1)
    schema_version: str = "1.0"
    tenant_id: str = "erik_default"
    status: ObjectStatus = ObjectStatus.active
    time: TimeFields = Field(default_factory=TimeFields)
    provenance: Provenance = Field(default_factory=Provenance)
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)
    privacy: Privacy = Field(default_factory=Privacy)
    body: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", mode="before")
    @classmethod
    def _validate_id(cls, v: Any) -> Any:
        if isinstance(v, str) and not v.strip():
            raise ValueError("id must not be blank")
        return v

    @field_validator("type", mode="before")
    @classmethod
    def _validate_type(cls, v: Any) -> Any:
        if isinstance(v, str) and not v.strip():
            raise ValueError("type must not be blank")
        return v
