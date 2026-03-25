"""Type registry mapping canonical type names to Pydantic model classes.

Every concrete ``BaseEnvelope`` subclass implemented in the Erik ontology is
registered here under its canonical ``type`` string.  The registry is used for
generic deserialization, schema introspection, and validation.
"""
from __future__ import annotations

from typing import Optional, Type

from ontology.base import BaseEnvelope
from ontology.patient import Patient, ALSTrajectory
from ontology.observation import Observation
from ontology.interpretation import Interpretation, EtiologicDriverProfile
from ontology.state import (
    TDP43FunctionalState,
    SplicingState,
    GlialState,
    NMJIntegrityState,
    RespiratoryReserveState,
    FunctionalState,
    ReversibilityWindowEstimate,
    UncertaintyState,
    DiseaseStateSnapshot,
)
from ontology.evidence import EvidenceItem, EvidenceBundle
from ontology.intervention import Intervention
from ontology.protocol import CureProtocolCandidate, MonitoringPlan
from ontology.discovery import MechanismHypothesis, ExperimentProposal
from ontology.meta import LearningEpisode, ErrorRecord, ImprovementProposal, Branch


# ---------------------------------------------------------------------------
# Internal registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Type[BaseEnvelope]] = {
    # Patient
    "Patient":                    Patient,
    "ALSTrajectory":              ALSTrajectory,
    # Observation
    "Observation":                Observation,
    # Interpretation
    "Interpretation":             Interpretation,
    "EtiologicDriverProfile":     EtiologicDriverProfile,
    # State
    "TDP43FunctionalState":       TDP43FunctionalState,
    "SplicingState":              SplicingState,
    "GlialState":                 GlialState,
    "NMJIntegrityState":          NMJIntegrityState,
    "RespiratoryReserveState":    RespiratoryReserveState,
    "FunctionalState":            FunctionalState,
    "ReversibilityWindowEstimate": ReversibilityWindowEstimate,
    "UncertaintyState":           UncertaintyState,
    "DiseaseStateSnapshot":       DiseaseStateSnapshot,
    # Evidence
    "EvidenceItem":               EvidenceItem,
    "EvidenceBundle":             EvidenceBundle,
    # Intervention
    "Intervention":               Intervention,
    # Protocol
    "CureProtocolCandidate":      CureProtocolCandidate,
    "MonitoringPlan":             MonitoringPlan,
    # Discovery
    "MechanismHypothesis":        MechanismHypothesis,
    "ExperimentProposal":         ExperimentProposal,
    # Meta
    "LearningEpisode":            LearningEpisode,
    "ErrorRecord":                ErrorRecord,
    "ImprovementProposal":        ImprovementProposal,
    "Branch":                     Branch,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_model_class(type_name: str) -> Optional[Type[BaseEnvelope]]:
    """Return the Pydantic model class for *type_name*, or ``None`` if unknown.

    The lookup is case-sensitive; ``"patient"`` will not match ``"Patient"``.
    """
    return _REGISTRY.get(type_name)


def list_types() -> list[str]:
    """Return a sorted list of all registered canonical type names."""
    return sorted(_REGISTRY.keys())
