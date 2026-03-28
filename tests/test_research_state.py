"""Tests for ResearchState — point-in-time snapshot of the research loop."""
from __future__ import annotations
import pytest
from research.state import ResearchState, initial_state

class TestResearchState:
    def test_initial_state_defaults(self):
        state = initial_state(subject_ref="traj:draper_001")
        assert state.step_count == 0
        assert state.protocol_version == 0
        assert state.protocol_stable_cycles == 0
        assert state.total_evidence_items == 0
        assert state.converged is False
        assert state.subject_ref == "traj:draper_001"

    def test_initial_state_has_empty_collections(self):
        state = initial_state(subject_ref="traj:draper_001")
        assert state.active_hypotheses == []
        assert state.resolved_hypotheses == 0
        assert state.action_values == {}
        assert state.action_counts == {}
        assert state.causal_chains == {}

    def test_evidence_by_layer_defaults(self):
        state = initial_state(subject_ref="traj:draper_001")
        assert "root_cause_suppression" in state.evidence_by_layer
        assert all(v == 0 for v in state.evidence_by_layer.values())

    def test_state_serializes_to_dict(self):
        state = initial_state(subject_ref="traj:draper_001")
        d = state.to_dict()
        assert isinstance(d, dict)
        assert d["step_count"] == 0
        assert d["subject_ref"] == "traj:draper_001"

    def test_state_roundtrip(self):
        state = initial_state(subject_ref="traj:draper_001")
        state.step_count = 42
        state.total_evidence_items = 100
        d = state.to_dict()
        restored = ResearchState.from_dict(d)
        assert restored.step_count == 42
        assert restored.total_evidence_items == 100


class TestUncertaintyFields:
    def test_state_has_uncertainty_fields(self):
        state = initial_state(subject_ref="traj:draper_001")
        assert hasattr(state, 'uncertainty_score')
        assert hasattr(state, 'uncertainty_history')
        assert state.uncertainty_score == 1.0
        assert state.uncertainty_history == []

    def test_uncertainty_serializes(self):
        state = initial_state(subject_ref="traj:draper_001")
        state.uncertainty_score = 0.75
        state.uncertainty_history = [1.0, 0.9, 0.75]
        d = state.to_dict()
        assert d["uncertainty_score"] == 0.75
        restored = ResearchState.from_dict(d)
        assert restored.uncertainty_score == 0.75
        assert restored.uncertainty_history == [1.0, 0.9, 0.75]
