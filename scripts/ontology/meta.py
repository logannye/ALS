"""Meta-loop models for the Erik ALS causal research engine.

These models represent the autonomous learning cycle: episodes, errors,
improvement proposals, and experimental branches.

Includes:
- LearningEpisode      — One pass of the observe → predict → act → evaluate loop
- ErrorRecord          — A recorded prediction or reasoning error
- ImprovementProposal  — A proposal to improve a model component
- Branch               — An experimental model branch for safe evaluation
"""
from __future__ import annotations

from typing import Optional

from pydantic import Field

from ontology.base import BaseEnvelope


# ---------------------------------------------------------------------------
# LearningEpisode
# ---------------------------------------------------------------------------

class LearningEpisode(BaseEnvelope):
    """One complete pass of the observe → predict → act → evaluate learning loop.

    ``type`` is locked to ``"LearningEpisode"``.
    """

    type: str = Field(default="LearningEpisode", min_length=1)

    subject_ref: str
    trigger: str
    state_snapshot_ref: Optional[str] = None
    protocol_ref: Optional[str] = None
    expected_outcome_ref: Optional[str] = None
    actual_outcome_ref: Optional[str] = None
    error_record_refs: list[str] = Field(default_factory=list)
    replay_trace_ref: Optional[str] = None


# ---------------------------------------------------------------------------
# ErrorRecord
# ---------------------------------------------------------------------------

class ErrorRecord(BaseEnvelope):
    """A recorded prediction or reasoning error with root cause candidates.

    ``type`` is locked to ``"ErrorRecord"``.
    """

    type: str = Field(default="ErrorRecord", min_length=1)

    category: str
    severity: str
    description: str
    affected_components: list[str] = Field(default_factory=list)
    candidate_root_causes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# ImprovementProposal
# ---------------------------------------------------------------------------

class ImprovementProposal(BaseEnvelope):
    """A proposal to improve a specific model or system component.

    ``type`` is locked to ``"ImprovementProposal"``.
    """

    type: str = Field(default="ImprovementProposal", min_length=1)

    proposal_kind: str
    target_component: str
    description: str
    justification_refs: list[str] = Field(default_factory=list)
    evaluation_plan_ref: Optional[str] = None
    branch_ref: Optional[str] = None


# ---------------------------------------------------------------------------
# Branch
# ---------------------------------------------------------------------------

class Branch(BaseEnvelope):
    """An experimental model branch created for safe isolated evaluation.

    ``type`` is locked to ``"Branch"``.
    ``deployment_rights`` defaults to ``"none"`` — branches cannot be
    promoted to production without explicit approval.
    """

    type: str = Field(default="Branch", min_length=1)

    parent_model_ref: str
    branch_purpose: str
    created_from_snapshot_ref: Optional[str] = None
    deployment_rights: str = "none"
