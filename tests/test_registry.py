"""Tests for ontology.registry — canonical type registry."""
import pytest

from ontology.registry import get_model_class, list_types
from ontology.base import BaseEnvelope
from ontology.patient import Patient, ALSTrajectory
from ontology.observation import Observation
from ontology.interpretation import Interpretation, EtiologicDriverProfile
from ontology.state import (
    DiseaseStateSnapshot,
    TDP43FunctionalState,
    SplicingState,
    GlialState,
    NMJIntegrityState,
    RespiratoryReserveState,
    FunctionalState,
    ReversibilityWindowEstimate,
    UncertaintyState,
)
from ontology.evidence import EvidenceItem, EvidenceBundle
from ontology.intervention import Intervention
from ontology.protocol import CureProtocolCandidate, MonitoringPlan
from ontology.discovery import MechanismHypothesis, ExperimentProposal
from ontology.meta import LearningEpisode, ErrorRecord, ImprovementProposal, Branch


# ---------------------------------------------------------------------------
# get_model_class — known types
# ---------------------------------------------------------------------------

class TestGetModelClassKnownTypes:
    def test_patient_returns_patient_class(self):
        assert get_model_class("Patient") is Patient

    def test_als_trajectory_returns_class(self):
        assert get_model_class("ALSTrajectory") is ALSTrajectory

    def test_observation_returns_class(self):
        assert get_model_class("Observation") is Observation

    def test_interpretation_returns_class(self):
        assert get_model_class("Interpretation") is Interpretation

    def test_etiologic_driver_profile_returns_class(self):
        assert get_model_class("EtiologicDriverProfile") is EtiologicDriverProfile

    def test_disease_state_snapshot_returns_class(self):
        assert get_model_class("DiseaseStateSnapshot") is DiseaseStateSnapshot

    def test_tdp43_functional_state_returns_class(self):
        assert get_model_class("TDP43FunctionalState") is TDP43FunctionalState

    def test_splicing_state_returns_class(self):
        assert get_model_class("SplicingState") is SplicingState

    def test_glial_state_returns_class(self):
        assert get_model_class("GlialState") is GlialState

    def test_nmj_integrity_state_returns_class(self):
        assert get_model_class("NMJIntegrityState") is NMJIntegrityState

    def test_respiratory_reserve_state_returns_class(self):
        assert get_model_class("RespiratoryReserveState") is RespiratoryReserveState

    def test_functional_state_returns_class(self):
        assert get_model_class("FunctionalState") is FunctionalState

    def test_reversibility_window_returns_class(self):
        assert get_model_class("ReversibilityWindowEstimate") is ReversibilityWindowEstimate

    def test_uncertainty_state_returns_class(self):
        assert get_model_class("UncertaintyState") is UncertaintyState

    def test_evidence_item_returns_class(self):
        assert get_model_class("EvidenceItem") is EvidenceItem

    def test_evidence_bundle_returns_class(self):
        assert get_model_class("EvidenceBundle") is EvidenceBundle

    def test_intervention_returns_class(self):
        assert get_model_class("Intervention") is Intervention

    def test_cure_protocol_candidate_returns_class(self):
        assert get_model_class("CureProtocolCandidate") is CureProtocolCandidate

    def test_monitoring_plan_returns_class(self):
        assert get_model_class("MonitoringPlan") is MonitoringPlan

    def test_mechanism_hypothesis_returns_class(self):
        assert get_model_class("MechanismHypothesis") is MechanismHypothesis

    def test_experiment_proposal_returns_class(self):
        assert get_model_class("ExperimentProposal") is ExperimentProposal

    def test_learning_episode_returns_class(self):
        assert get_model_class("LearningEpisode") is LearningEpisode

    def test_error_record_returns_class(self):
        assert get_model_class("ErrorRecord") is ErrorRecord

    def test_improvement_proposal_returns_class(self):
        assert get_model_class("ImprovementProposal") is ImprovementProposal

    def test_branch_returns_class(self):
        assert get_model_class("Branch") is Branch


# ---------------------------------------------------------------------------
# get_model_class — all returned classes are BaseEnvelope subclasses
# ---------------------------------------------------------------------------

class TestAllClassesAreBaseEnvelopes:
    def test_all_registered_classes_inherit_base_envelope(self):
        for type_name in list_types():
            cls = get_model_class(type_name)
            assert cls is not None, f"get_model_class({type_name!r}) returned None"
            assert issubclass(cls, BaseEnvelope), (
                f"{type_name} → {cls} does not inherit from BaseEnvelope"
            )


# ---------------------------------------------------------------------------
# get_model_class — unknown types
# ---------------------------------------------------------------------------

class TestGetModelClassUnknownTypes:
    def test_unknown_type_returns_none(self):
        assert get_model_class("NonExistentType") is None

    def test_empty_string_returns_none(self):
        assert get_model_class("") is None

    def test_lowercase_patient_returns_none(self):
        # Registry is case-sensitive
        assert get_model_class("patient") is None


# ---------------------------------------------------------------------------
# list_types — completeness
# ---------------------------------------------------------------------------

class TestListTypes:
    EXPECTED_CORE_TYPES = {
        "Patient",
        "ALSTrajectory",
        "Observation",
        "Interpretation",
        "DiseaseStateSnapshot",
        "CureProtocolCandidate",
        "EvidenceBundle",
        "LearningEpisode",
        "ErrorRecord",
        "Branch",
    }

    def test_returns_list(self):
        result = list_types()
        assert isinstance(result, list)

    def test_all_core_types_present(self):
        registered = set(list_types())
        missing = self.EXPECTED_CORE_TYPES - registered
        assert not missing, f"Missing core types: {missing}"

    def test_list_has_at_least_25_types(self):
        assert len(list_types()) >= 25

    def test_no_duplicates(self):
        types = list_types()
        assert len(types) == len(set(types)), "list_types() contains duplicates"
