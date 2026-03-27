"""Tests for action selection policy — interleaved diversity + hypothesis-guided."""
from __future__ import annotations

import pytest

from research.actions import ActionType
from research.state import ResearchState, initial_state
from research.policy import select_action, _ACQUISITION_INTERVAL


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
        state = self._state(active_hypotheses=["hyp:test1"], new_evidence_since_regen=0, step_count=1)
        action, params = select_action(state, regen_threshold=10)
        assert action == ActionType.VALIDATE_HYPOTHESIS

    def test_deepen_when_shallow_chains(self):
        state = self._state(
            causal_chains={"int:pridopidine": 2},
            active_hypotheses=[],
            new_evidence_since_regen=0,
            step_count=1,
        )
        action, params = select_action(state, regen_threshold=10, target_depth=5)
        assert action == ActionType.DEEPEN_CAUSAL_CHAIN

    def test_forced_acquisition_every_n_steps(self):
        """Every _ACQUISITION_INTERVAL steps, policy forces evidence acquisition."""
        state = self._state(
            causal_chains={"int:pridopidine": 2},
            step_count=_ACQUISITION_INTERVAL,
        )
        action, params = select_action(state, regen_threshold=10)
        assert action in {
            ActionType.SEARCH_PUBMED, ActionType.SEARCH_TRIALS,
            ActionType.QUERY_PATHWAYS, ActionType.QUERY_PPI_NETWORK,
        }

    def test_generate_hypothesis_when_uncertainties_exist(self):
        state = self._state(
            active_hypotheses=[],
            causal_chains={},
            top_uncertainties=["subtype_ambiguity"],
            step_count=1,
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
        assert action in {
            ActionType.SEARCH_PUBMED, ActionType.SEARCH_TRIALS,
            ActionType.QUERY_PATHWAYS, ActionType.QUERY_PPI_NETWORK,
        }

    def test_diverse_actions_over_10_steps(self):
        """Verify multiple action types are selected over a 10-step window."""
        state = self._state(
            causal_chains={"int:pridopidine": 2, "int:vtx002": 0},
            top_uncertainties=["genetic_testing_pending"],
            active_hypotheses=[],
        )
        actions_seen = set()
        for i in range(10):
            state.step_count = i
            action, _ = select_action(state, regen_threshold=100)
            actions_seen.add(action)
        assert len(actions_seen) >= 2
