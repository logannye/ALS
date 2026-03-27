"""Tests for hypothesis generation, lifecycle, and action planning."""
from __future__ import annotations
import pytest
from research.hypotheses import create_hypothesis, plan_validation_actions, resolve_hypothesis, HypothesisStatus

class TestCreateHypothesis:
    def test_creates_valid_hypothesis(self):
        hyp = create_hypothesis(
            statement="Erik's TDP-43 pathology may respond to sigma-1R agonism",
            subject_ref="traj:draper_001",
            topic="pathology_reversal",
            cited_evidence=["evi:sigma1r_biology"],
        )
        assert hyp.type == "MechanismHypothesis"
        assert hyp.statement.startswith("Erik")
        assert hyp.current_support_direction.value == "insufficient"

    def test_hypothesis_has_id(self):
        hyp = create_hypothesis(
            statement="Test hypothesis",
            subject_ref="traj:draper_001",
            topic="root_cause_suppression",
        )
        assert hyp.id.startswith("hyp:")

class TestPlanValidationActions:
    def test_returns_action_list(self):
        actions = plan_validation_actions(
            statement="TDP-43 nuclear import may be restored by pridopidine",
            topic="pathology_reversal",
        )
        assert len(actions) >= 1
        assert all(isinstance(a, dict) for a in actions)
        assert all("action" in a for a in actions)

    def test_pubmed_search_included(self):
        actions = plan_validation_actions(
            statement="Sigma-1R agonism reduces TDP-43 aggregation",
            topic="pathology_reversal",
        )
        action_types = [a["action"] for a in actions]
        assert "search_pubmed" in action_types

class TestResolveHypothesis:
    def test_resolve_supported(self):
        from ontology.enums import EvidenceDirection
        from ontology.discovery import MechanismHypothesis
        hyp = MechanismHypothesis(
            id="hyp:test", statement="Test", subject_scope="traj:draper_001",
            current_support_direction=EvidenceDirection.insufficient,
        )
        resolved = resolve_hypothesis(hyp, evidence_for=["evi:a", "evi:b", "evi:c"], evidence_against=[])
        assert resolved.current_support_direction == EvidenceDirection.supports

    def test_resolve_refuted(self):
        from ontology.enums import EvidenceDirection
        from ontology.discovery import MechanismHypothesis
        hyp = MechanismHypothesis(
            id="hyp:test", statement="Test", subject_scope="traj:draper_001",
            current_support_direction=EvidenceDirection.insufficient,
        )
        resolved = resolve_hypothesis(hyp, evidence_for=[], evidence_against=["evi:x", "evi:y"])
        assert resolved.current_support_direction == EvidenceDirection.refutes

    def test_resolve_mixed(self):
        from ontology.enums import EvidenceDirection
        from ontology.discovery import MechanismHypothesis
        hyp = MechanismHypothesis(
            id="hyp:test", statement="Test", subject_scope="traj:draper_001",
            current_support_direction=EvidenceDirection.insufficient,
        )
        resolved = resolve_hypothesis(hyp, evidence_for=["evi:a"], evidence_against=["evi:b"])
        assert resolved.current_support_direction == EvidenceDirection.mixed
