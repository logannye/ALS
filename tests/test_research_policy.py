"""Tests for action selection policy — balanced cycle with validation budget."""
from __future__ import annotations

import pytest

from research.actions import ActionType
from research.state import ResearchState, initial_state
from research.policy import select_action, _CYCLE_LENGTH, _ACQUIRE_STEPS


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

    def test_acquisition_on_acquire_steps(self):
        """Cycle positions 0, 2, 4 should be acquisition actions."""
        for step in [0, 2, 4, 5, 7, 9]:  # positions 0,2,4 in first two cycles
            state = self._state(step_count=step)
            action, _ = select_action(state, regen_threshold=100)
            assert action in {
                ActionType.SEARCH_PUBMED, ActionType.SEARCH_TRIALS,
                ActionType.QUERY_PATHWAYS, ActionType.QUERY_PPI_NETWORK,
                ActionType.CHECK_PHARMACOGENOMICS,
            }, f"Step {step} (cycle pos {step % _CYCLE_LENGTH}) should be acquisition, got {action}"

    def test_reasoning_on_reason_step(self):
        """Cycle position 1 should be reasoning (chain or hypothesis)."""
        state = self._state(
            step_count=1,
            causal_chains={"int:pridopidine": 2},
            top_uncertainties=["test"],
        )
        action, _ = select_action(state, regen_threshold=100, target_depth=5)
        assert action in {ActionType.DEEPEN_CAUSAL_CHAIN, ActionType.GENERATE_HYPOTHESIS}

    def test_validation_on_validate_step_with_hypotheses(self):
        """Cycle position 3 with active hypotheses should validate."""
        state = self._state(
            step_count=3,
            active_hypotheses=["hyp:test1"],
            action_counts={"validate_hypothesis": 1},
        )
        action, _ = select_action(state, regen_threshold=100)
        assert action == ActionType.VALIDATE_HYPOTHESIS

    def test_validation_budget_prevents_over_validation(self):
        """When validation ratio is too high, should acquire instead."""
        state = self._state(
            step_count=3,
            active_hypotheses=["hyp:test1"],
            last_action="validate_hypothesis",
            action_counts={"validate_hypothesis": 50, "search_pubmed": 10},
        )
        action, _ = select_action(state, regen_threshold=100)
        # Should NOT be validate_hypothesis due to budget exhaustion
        assert action != ActionType.VALIDATE_HYPOTHESIS

    def test_diverse_actions_over_cycle(self):
        """A full cycle should include both acquisition and reasoning."""
        state = self._state(
            causal_chains={"int:pridopidine": 2},
            top_uncertainties=["test_uncertainty"],
            active_hypotheses=["hyp:test1"],
            action_counts={"validate_hypothesis": 1},
        )
        actions_seen = set()
        for i in range(_CYCLE_LENGTH * 2):
            state.step_count = i
            action, _ = select_action(state, regen_threshold=100)
            actions_seen.add(action)
        assert len(actions_seen) >= 3, f"Expected 3+ action types in 2 cycles, got {actions_seen}"

    def test_hypothesis_expiry(self):
        """Hypotheses validated too many times should be expired."""
        state = self._state(
            active_hypotheses=["hyp:old1", "hyp:old2"],
            action_counts={"validate_hypothesis": 30},
            resolved_hypotheses=2,
        )
        # Trigger a selection which calls _maybe_expire_hypotheses
        select_action(state, regen_threshold=100)
        # At least one hypothesis should have been expired
        assert len(state.active_hypotheses) < 2 or state.resolved_hypotheses > 2
