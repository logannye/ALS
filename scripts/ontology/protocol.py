"""Protocol models for the Erik ALS causal research engine.

Includes:
- ProtocolLayerEntry      — A single layer within a multi-layer protocol (NOT a BaseEnvelope)
- CureProtocolCandidate   — A complete candidate cure protocol with evidence and approval state
- MonitoringPlan          — Scheduled checks, success criteria, and failure triggers
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ontology.base import BaseEnvelope
from ontology.enums import ApprovalState, ProtocolLayer


# ---------------------------------------------------------------------------
# ProtocolLayerEntry  (NOT a BaseEnvelope — lightweight sub-object)
# ---------------------------------------------------------------------------

class ProtocolLayerEntry(BaseModel):
    """A single treatment layer entry within a CureProtocolCandidate.

    Not an envelope — this is a value-object embedded inside CureProtocolCandidate.
    """

    layer: ProtocolLayer
    intervention_refs: list[str] = Field(default_factory=list)
    start_offset_days: int = 0
    notes: str = ""


# ---------------------------------------------------------------------------
# CureProtocolCandidate
# ---------------------------------------------------------------------------

class CureProtocolCandidate(BaseEnvelope):
    """A complete candidate cure / treatment protocol for a specific patient.

    ``type`` is locked to ``"CureProtocolCandidate"``.
    ``approval_state`` defaults to ``ApprovalState.pending``.
    """

    type: str = Field(default="CureProtocolCandidate", min_length=1)

    subject_ref: str
    objective: str
    eligibility_constraints: list[str] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
    assumed_active_programs: list[str] = Field(default_factory=list)
    layers: list[ProtocolLayerEntry] = Field(default_factory=list)
    monitoring_plan_ref: Optional[str] = None
    expected_state_shift_summary: dict[str, float] = Field(default_factory=dict)
    dominant_failure_modes: list[str] = Field(default_factory=list)
    approval_state: ApprovalState = ApprovalState.pending
    required_approval_refs: list[str] = Field(default_factory=list)
    evidence_bundle_refs: list[str] = Field(default_factory=list)
    uncertainty_ref: Optional[str] = None


# ---------------------------------------------------------------------------
# MonitoringPlan
# ---------------------------------------------------------------------------

class MonitoringPlan(BaseEnvelope):
    """A monitoring plan specifying scheduled checks, success criteria, and failure triggers.

    ``type`` is locked to ``"MonitoringPlan"``.
    """

    type: str = Field(default="MonitoringPlan", min_length=1)

    subject_ref: str
    scheduled_checks: list[dict] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    failure_triggers: list[str] = Field(default_factory=list)
