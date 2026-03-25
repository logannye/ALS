"""Evidence models for the Erik ALS causal research engine.

Includes:
- EvidenceItem   — A single piece of evidence with direction, source, and strength
- EvidenceBundle — Aggregated evidence for a subject/topic with coverage scores
"""
from __future__ import annotations

from typing import Optional

from pydantic import Field

from ontology.base import BaseEnvelope
from ontology.enums import EvidenceDirection, EvidenceStrength


# ---------------------------------------------------------------------------
# EvidenceItem
# ---------------------------------------------------------------------------

class EvidenceItem(BaseEnvelope):
    """A single piece of evidence supporting, refuting, or remaining mixed on a claim.

    ``type`` is locked to ``"EvidenceItem"``.
    """

    type: str = Field(default="EvidenceItem", min_length=1)

    claim: str
    direction: EvidenceDirection
    source_refs: list[str] = Field(default_factory=list)
    strength: EvidenceStrength = EvidenceStrength.unknown
    supersedes_ref: Optional[str] = None
    notes: str = ""


# ---------------------------------------------------------------------------
# EvidenceBundle
# ---------------------------------------------------------------------------

class EvidenceBundle(BaseEnvelope):
    """Aggregated evidence items for a given subject and topic.

    ``type`` is locked to ``"EvidenceBundle"``.
    """

    type: str = Field(default="EvidenceBundle", min_length=1)

    subject_ref: str
    topic: str
    evidence_item_refs: list[str] = Field(default_factory=list)
    contradiction_refs: list[str] = Field(default_factory=list)
    coverage_score: float
    grounding_score: float
