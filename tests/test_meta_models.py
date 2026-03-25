"""Tests for ontology.discovery and ontology.meta — learning loop models."""
import pytest

from ontology.discovery import ExperimentProposal, MechanismHypothesis
from ontology.enums import EvidenceDirection
from ontology.meta import Branch, ErrorRecord, ImprovementProposal, LearningEpisode


# ---------------------------------------------------------------------------
# MechanismHypothesis
# ---------------------------------------------------------------------------

class TestMechanismHypothesis:
    def _make(self) -> MechanismHypothesis:
        return MechanismHypothesis(
            id="hypothesis:tdp43_stmn2_v1",
            statement="TDP-43 loss of nuclear function drives cryptic exon inclusion in STMN2, preventing axon regeneration",
            subject_scope="sporadic_tdp43_als",
            predicted_observables=["stmn2_mrna_truncation", "axon_regeneration_failure"],
            candidate_tests=["rna_seq_stmn2_splicing", "stmn2_protein_immunoblot"],
            current_support_direction=EvidenceDirection.supports,
        )

    def test_type_is_mechanism_hypothesis(self):
        h = self._make()
        assert h.type == "MechanismHypothesis"

    def test_statement(self):
        h = self._make()
        assert "TDP-43" in h.statement

    def test_subject_scope(self):
        h = self._make()
        assert h.subject_scope == "sporadic_tdp43_als"

    def test_predicted_observables(self):
        h = self._make()
        assert "stmn2_mrna_truncation" in h.predicted_observables

    def test_candidate_tests(self):
        h = self._make()
        assert "rna_seq_stmn2_splicing" in h.candidate_tests

    def test_current_support_direction(self):
        h = self._make()
        assert h.current_support_direction == EvidenceDirection.supports


# ---------------------------------------------------------------------------
# ExperimentProposal
# ---------------------------------------------------------------------------

class TestExperimentProposal:
    def _make(self) -> ExperimentProposal:
        return ExperimentProposal(
            id="experiment:stmn2_aso_correction_v1",
            objective="Test whether STMN2 cryptic splicing is corrected by TDP-43 restoration ASO",
            modality="rna_seq_perturbation",
            required_inputs=["patient_ipsc_motor_neurons", "tdp43_aso_compound"],
            expected_information_gain=0.72,
            estimated_cost_band="$10k–$50k",
            estimated_duration_days=90,
            linked_hypothesis_refs=["hypothesis:tdp43_stmn2_v1"],
        )

    def test_type_is_experiment_proposal(self):
        e = self._make()
        assert e.type == "ExperimentProposal"

    def test_objective(self):
        e = self._make()
        assert "STMN2" in e.objective

    def test_modality(self):
        e = self._make()
        assert e.modality == "rna_seq_perturbation"

    def test_required_inputs(self):
        e = self._make()
        assert "patient_ipsc_motor_neurons" in e.required_inputs

    def test_expected_information_gain(self):
        e = self._make()
        assert e.expected_information_gain == pytest.approx(0.72)

    def test_estimated_cost_band(self):
        e = self._make()
        assert "$10k" in e.estimated_cost_band

    def test_estimated_duration_days(self):
        e = self._make()
        assert e.estimated_duration_days == 90

    def test_linked_hypothesis_refs(self):
        e = self._make()
        assert "hypothesis:tdp43_stmn2_v1" in e.linked_hypothesis_refs


# ---------------------------------------------------------------------------
# LearningEpisode
# ---------------------------------------------------------------------------

class TestLearningEpisode:
    def _make(self) -> LearningEpisode:
        return LearningEpisode(
            id="episode:erik_draper_cycle_001",
            subject_ref="patient:erik_draper",
            trigger="new_observation_received",
            state_snapshot_ref="snapshot:erik_draper_latest",
            protocol_ref="protocol:erik_draper_phase1_candidate",
            expected_outcome_ref="outcome:expected_v1",
            actual_outcome_ref="outcome:actual_v1",
            error_record_refs=["error:prediction_miss_001"],
            replay_trace_ref="trace:cycle_001",
        )

    def test_type_is_learning_episode(self):
        e = self._make()
        assert e.type == "LearningEpisode"

    def test_subject_ref(self):
        e = self._make()
        assert e.subject_ref == "patient:erik_draper"

    def test_trigger(self):
        e = self._make()
        assert e.trigger == "new_observation_received"

    def test_state_snapshot_ref(self):
        e = self._make()
        assert e.state_snapshot_ref == "snapshot:erik_draper_latest"

    def test_protocol_ref(self):
        e = self._make()
        assert e.protocol_ref == "protocol:erik_draper_phase1_candidate"

    def test_expected_and_actual_outcome_refs(self):
        e = self._make()
        assert e.expected_outcome_ref == "outcome:expected_v1"
        assert e.actual_outcome_ref == "outcome:actual_v1"

    def test_error_record_refs(self):
        e = self._make()
        assert "error:prediction_miss_001" in e.error_record_refs

    def test_replay_trace_ref(self):
        e = self._make()
        assert e.replay_trace_ref == "trace:cycle_001"

    def test_minimal_creation(self):
        e = LearningEpisode(
            id="episode:minimal",
            subject_ref="patient:x",
            trigger="manual",
            state_snapshot_ref=None,
            protocol_ref=None,
            expected_outcome_ref=None,
            actual_outcome_ref=None,
            error_record_refs=[],
            replay_trace_ref=None,
        )
        assert e.type == "LearningEpisode"


# ---------------------------------------------------------------------------
# ErrorRecord
# ---------------------------------------------------------------------------

class TestErrorRecord:
    def _make(self) -> ErrorRecord:
        return ErrorRecord(
            id="error:prediction_miss_001",
            category="prediction_error",
            severity="moderate",
            description="Predicted ALSFRS-R stable but observed 3-point decline in 4 weeks",
            affected_components=["functional_state_model", "progression_predictor"],
            candidate_root_causes=[
                "underweighted_respiratory_decline",
                "missing_nmj_occupancy_data",
            ],
        )

    def test_type_is_error_record(self):
        e = self._make()
        assert e.type == "ErrorRecord"

    def test_category(self):
        e = self._make()
        assert e.category == "prediction_error"

    def test_severity(self):
        e = self._make()
        assert e.severity == "moderate"

    def test_description(self):
        e = self._make()
        assert "ALSFRS-R" in e.description

    def test_affected_components(self):
        e = self._make()
        assert "functional_state_model" in e.affected_components

    def test_candidate_root_causes(self):
        e = self._make()
        assert "missing_nmj_occupancy_data" in e.candidate_root_causes

    def test_minimal_creation(self):
        e = ErrorRecord(
            id="error:minimal",
            category="unknown",
            severity="low",
            description="",
            affected_components=[],
            candidate_root_causes=[],
        )
        assert e.type == "ErrorRecord"


# ---------------------------------------------------------------------------
# ImprovementProposal
# ---------------------------------------------------------------------------

class TestImprovementProposal:
    def _make(self) -> ImprovementProposal:
        return ImprovementProposal(
            id="proposal:add_nmj_features_v1",
            proposal_kind="model_feature_addition",
            target_component="functional_state_model",
            description="Add NMJ occupancy estimate as input feature to ALSFRS-R trajectory predictor",
            justification_refs=["error:prediction_miss_001", "hypothesis:tdp43_stmn2_v1"],
            evaluation_plan_ref="experiment:nmj_feature_ablation_v1",
            branch_ref="branch:nmj_feature_test_v1",
        )

    def test_type_is_improvement_proposal(self):
        p = self._make()
        assert p.type == "ImprovementProposal"

    def test_proposal_kind(self):
        p = self._make()
        assert p.proposal_kind == "model_feature_addition"

    def test_target_component(self):
        p = self._make()
        assert p.target_component == "functional_state_model"

    def test_description(self):
        p = self._make()
        assert "NMJ" in p.description

    def test_justification_refs(self):
        p = self._make()
        assert "error:prediction_miss_001" in p.justification_refs

    def test_evaluation_plan_ref(self):
        p = self._make()
        assert p.evaluation_plan_ref == "experiment:nmj_feature_ablation_v1"

    def test_branch_ref(self):
        p = self._make()
        assert p.branch_ref == "branch:nmj_feature_test_v1"


# ---------------------------------------------------------------------------
# Branch
# ---------------------------------------------------------------------------

class TestBranch:
    def _make(self) -> Branch:
        return Branch(
            id="branch:nmj_feature_test_v1",
            parent_model_ref="model:functional_state_v3",
            branch_purpose="Test NMJ occupancy feature addition to trajectory predictor",
            created_from_snapshot_ref="snapshot:erik_draper_latest",
        )

    def test_type_is_branch(self):
        b = self._make()
        assert b.type == "Branch"

    def test_parent_model_ref(self):
        b = self._make()
        assert b.parent_model_ref == "model:functional_state_v3"

    def test_branch_purpose(self):
        b = self._make()
        assert "NMJ" in b.branch_purpose

    def test_created_from_snapshot_ref(self):
        b = self._make()
        assert b.created_from_snapshot_ref == "snapshot:erik_draper_latest"

    def test_deployment_rights_default_none(self):
        b = self._make()
        assert b.deployment_rights == "none"

    def test_minimal_creation(self):
        b = Branch(
            id="branch:minimal",
            parent_model_ref="model:x",
            branch_purpose="Test",
            created_from_snapshot_ref=None,
        )
        assert b.type == "Branch"
        assert b.deployment_rights == "none"
