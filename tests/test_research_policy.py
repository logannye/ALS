"""Tests for action selection policy."""
from __future__ import annotations
import pytest
from research.actions import ActionType
from research.state import ResearchState, initial_state
from research.policy import select_action

class TestSelectAction:
    def _state(self, **overrides) -> ResearchState:
        s = initial_state(subject_ref="traj:draper_001")
        for k, v in overrides.items():
            setattr(s, k, v)
        return s

    def test_regen_when_enough_new_evidence(self):
        state = self._state(new_evidence_since_regen=15, protocol_version=1)
        action, params = select_action(state, regen_threshold=10)
        assert action == ActionType.REGENERATE_PROTOCOL

    def test_validate_when_pending_hypotheses(self):
        state = self._state(active_hypotheses=["hyp:test1"], new_evidence_since_regen=0)
        action, params = select_action(state, regen_threshold=10)
        assert action == ActionType.VALIDATE_HYPOTHESIS

    def test_deepen_when_shallow_chains(self):
        state = self._state(
            causal_chains={"int:pridopidine": 2},
            active_hypotheses=[],
            new_evidence_since_regen=0,
        )
        action, params = select_action(state, regen_threshold=10, target_depth=5)
        assert action == ActionType.DEEPEN_CAUSAL_CHAIN

    def test_search_when_uncertainty_high(self):
        state = self._state(
            top_uncertainties=["genetic_testing_pending"],
            active_hypotheses=[],
            new_evidence_since_regen=0,
            causal_chains={},
        )
        action, params = select_action(state, regen_threshold=10)
        assert action in {ActionType.SEARCH_PUBMED, ActionType.GENERATE_HYPOTHESIS}

    def test_generate_hypothesis_when_converging(self):
        state = self._state(
            protocol_stable_cycles=2,
            active_hypotheses=[],
            new_evidence_since_regen=0,
            causal_chains={},
            top_uncertainties=["subtype_ambiguity"],
        )
        action, params = select_action(state, regen_threshold=10)
        assert action == ActionType.GENERATE_HYPOTHESIS

    def test_layer_rotation_fallback(self):
        state = self._state(
            active_hypotheses=[],
            new_evidence_since_regen=0,
            causal_chains={},
            top_uncertainties=[],
            step_count=5,
        )
        action, params = select_action(state, regen_threshold=10)
        assert action == ActionType.SEARCH_PUBMED
