"""Interpretation and EtiologicDriverProfile models for the Erik ALS engine.

Includes:
- Interpretation          — A clinician or model-generated interpretation of observations
- EtiologicDriverProfile  — Bayesian posterior over ALS subtypes for a patient
"""
from __future__ import annotations

from typing import Optional

from pydantic import Field, computed_field

from ontology.base import BaseEnvelope
from ontology.enums import InterpretationKind, SubtypeClass


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

class Interpretation(BaseEnvelope):
    """A clinician or model-generated interpretation of one or more observations.

    ``type`` is locked to ``"Interpretation"``.
    """

    type: str = Field(default="Interpretation", min_length=1)

    subject_ref: str
    interpretation_kind: InterpretationKind
    value: str
    supporting_observation_refs: list[str] = Field(default_factory=list)
    evidence_bundle_ref: Optional[str] = None
    supersedes_ref: Optional[str] = None
    notes: str = ""


# ---------------------------------------------------------------------------
# EtiologicDriverProfile
# ---------------------------------------------------------------------------

class EtiologicDriverProfile(BaseEnvelope):
    """Bayesian posterior distribution over ALS subtypes for a single patient.

    ``type`` is locked to ``"EtiologicDriverProfile"``.
    The ``dominant_subtype`` computed field returns the SubtypeClass key with
    the highest posterior probability.
    """

    type: str = Field(default="EtiologicDriverProfile", min_length=1)

    subject_ref: str
    posterior: dict[SubtypeClass, float]
    supporting_evidence_refs: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[misc]
    @property
    def dominant_subtype(self) -> SubtypeClass:
        """Return the SubtypeClass with the highest posterior probability."""
        return max(self.posterior, key=lambda k: self.posterior[k])
