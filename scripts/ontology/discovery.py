"""Discovery models for the Erik ALS causal research engine.

Includes:
- MechanismHypothesis — A testable mechanistic hypothesis about ALS biology
- ExperimentProposal  — A proposed experiment to test a hypothesis
"""
from __future__ import annotations

from pydantic import Field

from ontology.base import BaseEnvelope
from ontology.enums import EvidenceDirection


# ---------------------------------------------------------------------------
# MechanismHypothesis
# ---------------------------------------------------------------------------

class MechanismHypothesis(BaseEnvelope):
    """A testable mechanistic hypothesis about ALS biology.

    ``type`` is locked to ``"MechanismHypothesis"``.
    """

    type: str = Field(default="MechanismHypothesis", min_length=1)

    statement: str
    subject_scope: str
    predicted_observables: list[str] = Field(default_factory=list)
    candidate_tests: list[str] = Field(default_factory=list)
    current_support_direction: EvidenceDirection


# ---------------------------------------------------------------------------
# ExperimentProposal
# ---------------------------------------------------------------------------

class ExperimentProposal(BaseEnvelope):
    """A proposed experiment to generate information relevant to a hypothesis.

    ``type`` is locked to ``"ExperimentProposal"``.
    """

    type: str = Field(default="ExperimentProposal", min_length=1)

    objective: str
    modality: str
    required_inputs: list[str] = Field(default_factory=list)
    expected_information_gain: float
    estimated_cost_band: str
    estimated_duration_days: int
    linked_hypothesis_refs: list[str] = Field(default_factory=list)
