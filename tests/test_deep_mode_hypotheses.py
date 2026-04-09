"""Tests for hypothesis lifecycle in deep mode (_select_deep_action)."""
from __future__ import annotations

import pytest
from dataclasses import replace

from research.state import ResearchState, initial_state


class TestSelectDeepAction:
    """Tests for _select_deep_action() hypothesis scheduling."""

    def test_every_10th_step_is_hypothesis_action(self):
        """step_count=10, active_hypotheses non-empty -> validate or generate."""
        from run_loop import _select_deep_action

        state = initial_state(subject_ref="traj:test")
        state = replace(
            state,
            step_count=10,
            active_hypotheses=["TDP-43 aggregation drives motor neuron death"],
        )
        action_name, params = _select_deep_action(state, is_stagnated=False)
        assert action_name in ("validate_hypothesis", "generate_hypothesis")

    def test_validates_when_hypotheses_pending(self):
        """step_count=10, active_hypotheses=["TDP-43 causes..."] -> validate_hypothesis."""
        from run_loop import _select_deep_action

        state = initial_state(subject_ref="traj:test")
        state = replace(
            state,
            step_count=10,
            active_hypotheses=["TDP-43 causes motor neuron degeneration via aggregation"],
        )
        action_name, params = _select_deep_action(state, is_stagnated=False)
        assert action_name == "validate_hypothesis"
        assert params["hypothesis_id"] == "TDP-43 causes motor neuron degeneration via aggregation"

    def test_generates_when_no_hypotheses(self):
        """step_count=10, active_hypotheses=[] -> generate_hypothesis."""
        from run_loop import _select_deep_action

        state = initial_state(subject_ref="traj:test")
        state = replace(state, step_count=10, active_hypotheses=[])
        action_name, params = _select_deep_action(state, is_stagnated=False)
        assert action_name == "generate_hypothesis"
        assert "topic" in params
        assert params["topic"] == "ALS mechanism"

    def test_non_hypothesis_step_returns_empty(self):
        """step_count=7 (not divisible by 10) -> returns ("", {})."""
        from run_loop import _select_deep_action

        state = initial_state(subject_ref="traj:test")
        state = replace(state, step_count=7)
        action_name, params = _select_deep_action(state, is_stagnated=False)
        assert action_name == ""
        assert params == {}

    def test_step_0_is_hypothesis_step(self):
        """step_count=0 (divisible by 10) -> hypothesis action."""
        from run_loop import _select_deep_action

        state = initial_state(subject_ref="traj:test")
        state = replace(state, step_count=0, active_hypotheses=[])
        action_name, params = _select_deep_action(state, is_stagnated=False)
        assert action_name == "generate_hypothesis"

    def test_step_20_is_hypothesis_step(self):
        """step_count=20 (divisible by 10) -> hypothesis action."""
        from run_loop import _select_deep_action

        state = initial_state(subject_ref="traj:test")
        state = replace(state, step_count=20, active_hypotheses=["SOD1 misfolding"])
        action_name, params = _select_deep_action(state, is_stagnated=False)
        assert action_name == "validate_hypothesis"
        assert params["hypothesis_id"] == "SOD1 misfolding"

    def test_stagnated_returns_expanded_action(self):
        """When stagnated, returns an expanded action regardless of step_count."""
        from run_loop import _select_deep_action

        state = initial_state(subject_ref="traj:test")
        state = replace(state, step_count=10)
        action_name, params = _select_deep_action(state, is_stagnated=True)
        # Should return an expanded action, not a hypothesis action
        assert action_name != ""
        assert action_name != "validate_hypothesis" or action_name != "generate_hypothesis"

    def test_stagnated_non_hypothesis_step_returns_expanded(self):
        """When stagnated on a non-10th step, returns expanded action."""
        from run_loop import _select_deep_action

        state = initial_state(subject_ref="traj:test")
        state = replace(state, step_count=7)
        action_name, params = _select_deep_action(state, is_stagnated=True)
        assert action_name != ""
        assert isinstance(params, dict)

    def test_validates_first_hypothesis_in_list(self):
        """When multiple hypotheses pending, validates the first one."""
        from run_loop import _select_deep_action

        state = initial_state(subject_ref="traj:test")
        state = replace(
            state,
            step_count=30,
            active_hypotheses=[
                "FUS nuclear import defect",
                "C9orf72 DPR toxicity",
                "SOD1 gain-of-function",
            ],
        )
        action_name, params = _select_deep_action(state, is_stagnated=False)
        assert action_name == "validate_hypothesis"
        assert params["hypothesis_id"] == "FUS nuclear import defect"
