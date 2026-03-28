"""Tests for Thompson sampling policy — Beta posteriors, decay, diversity floor."""
from __future__ import annotations

import pytest

from research.actions import ActionType
from research.state import ResearchState, initial_state
from research.policy import (
    _update_posteriors,
    _apply_decay,
    select_action_thompson,
)


class TestUpdatePosteriors:
    def test_success_increments_alpha(self):
        posteriors = {"search_pubmed:layer_a": (1.0, 1.0)}
        updated = _update_posteriors(posteriors, "search_pubmed:layer_a", success=True)
        assert updated["search_pubmed:layer_a"] == (2.0, 1.0)

    def test_failure_increments_beta(self):
        posteriors = {"search_pubmed:layer_a": (1.0, 1.0)}
        updated = _update_posteriors(posteriors, "search_pubmed:layer_a", success=False)
        assert updated["search_pubmed:layer_a"] == (1.0, 2.0)

    def test_new_key_starts_uniform(self):
        posteriors = {}
        updated = _update_posteriors(posteriors, "new:ctx", success=True)
        assert updated["new:ctx"] == (2.0, 1.0)

    def test_does_not_mutate_original(self):
        posteriors = {"k": (3.0, 2.0)}
        _ = _update_posteriors(posteriors, "k", success=True)
        assert posteriors["k"] == (3.0, 2.0)


class TestApplyDecay:
    def test_decay_reduces(self):
        posteriors = {"a": (10.0, 5.0)}
        decayed = _apply_decay(posteriors, rate=0.95)
        assert decayed["a"][0] == pytest.approx(9.5)
        assert decayed["a"][1] == pytest.approx(4.75)

    def test_decay_floor(self):
        posteriors = {"a": (1.0, 1.0)}
        decayed = _apply_decay(posteriors, rate=0.5)
        assert decayed["a"][0] >= 1.0
        assert decayed["a"][1] >= 1.0

    def test_decay_does_not_mutate_original(self):
        posteriors = {"a": (10.0, 5.0)}
        _ = _apply_decay(posteriors, rate=0.5)
        assert posteriors["a"] == (10.0, 5.0)

    def test_decay_floor_prevents_below_one(self):
        posteriors = {"a": (1.05, 1.02)}
        decayed = _apply_decay(posteriors, rate=0.01)
        assert decayed["a"][0] >= 1.0
        assert decayed["a"][1] >= 1.0


class TestThompsonPolicy:
    def test_regeneration_preempts(self):
        state = initial_state("traj:test")
        state = ResearchState(**{**state.to_dict(), "new_evidence_since_regen": 15, "protocol_version": 1})
        action, _ = select_action_thompson(state, regen_threshold=10)
        assert action == ActionType.REGENERATE_PROTOCOL

    def test_no_regeneration_below_threshold(self):
        state = initial_state("traj:test")
        state = ResearchState(**{**state.to_dict(), "new_evidence_since_regen": 5, "protocol_version": 1})
        action, _ = select_action_thompson(state, regen_threshold=10)
        assert action != ActionType.REGENERATE_PROTOCOL

    def test_selects_valid_action(self):
        state = initial_state("traj:test")
        state = ResearchState(**{**state.to_dict(), "protocol_version": 1})
        action, params = select_action_thompson(state)
        assert isinstance(action, ActionType)
        assert isinstance(params, dict)

    def test_diversity_forces_starved_action(self):
        """Diversity floor: action types not used in 30+ steps must be forced.
        When search_pubmed was used at step 50 (current step=50, so delta=0 < 30)
        but all others have last_used=0 (delta=50 >= 30), the floor fires and
        forces a non-search_pubmed action."""
        state = initial_state("traj:test")
        # search_pubmed was used at step 50 (just now), all others at step 0
        last_per_type = {at.value: 0 for at in ActionType}
        last_per_type["search_pubmed"] = 50
        state = ResearchState(**{
            **state.to_dict(),
            "step_count": 50,
            "protocol_version": 1,
            "last_action_per_type": last_per_type,
        })
        # With all non-pubmed types starved, the diversity floor must fire
        # and return something other than search_pubmed
        action, _ = select_action_thompson(state)
        assert action.value != "search_pubmed", (
            f"Expected diversity floor to force a non-pubmed action, got {action}"
        )

    def test_returns_tuple(self):
        state = initial_state("traj:test")
        result = select_action_thompson(state)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_params_include_action_key(self):
        state = initial_state("traj:test")
        _, params = select_action_thompson(state)
        assert "action" in params

    def test_high_alpha_posterior_biases_toward_action(self):
        """An action with very high alpha should be selected most often."""
        state = initial_state("traj:test")
        # Give search_pubmed a very strong positive posterior
        posteriors = {at.value: (1.0, 100.0) for at in ActionType}
        posteriors["search_pubmed"] = (1000.0, 1.0)
        state = ResearchState(**{
            **state.to_dict(),
            "protocol_version": 1,
            "step_count": 1000,  # large step_count so diversity floor won't fire
            "last_action_per_type": {at.value: 990 for at in ActionType},  # all used recently
            "action_posteriors": posteriors,
        })
        counts: dict[str, int] = {}
        for _ in range(50):
            action, _ = select_action_thompson(state)
            counts[action.value] = counts.get(action.value, 0) + 1
        assert counts.get("search_pubmed", 0) > 20, (
            f"Expected search_pubmed to dominate with high alpha posterior, got {counts}"
        )

    def test_deepen_causal_chain_uses_shallow_chain(self):
        """When DEEPEN_CAUSAL_CHAIN wins Thompson, it should use the shallow chain."""
        state = initial_state("traj:test")
        # Force DEEPEN_CAUSAL_CHAIN to win by giving it the highest posterior
        posteriors = {at.value: (1.0, 100.0) for at in ActionType}
        posteriors["deepen_causal_chain"] = (1000.0, 1.0)
        state = ResearchState(**{
            **state.to_dict(),
            "protocol_version": 1,
            "step_count": 1000,
            "last_action_per_type": {at.value: 990 for at in ActionType},
            "action_posteriors": posteriors,
            "causal_chains": {"int:pridopidine": 2},
        })
        found_deepen = False
        for _ in range(30):
            action, params = select_action_thompson(state, target_depth=5)
            if action == ActionType.DEEPEN_CAUSAL_CHAIN:
                assert params.get("intervention_id") == "int:pridopidine"
                found_deepen = True
                break
        assert found_deepen, "Expected DEEPEN_CAUSAL_CHAIN to be selected at least once"

    def test_deepen_falls_back_when_no_shallow(self):
        """When DEEPEN_CAUSAL_CHAIN wins but all chains are deep, fallback to GENERATE_HYPOTHESIS."""
        state = initial_state("traj:test")
        posteriors = {at.value: (1.0, 100.0) for at in ActionType}
        posteriors["deepen_causal_chain"] = (1000.0, 1.0)
        state = ResearchState(**{
            **state.to_dict(),
            "protocol_version": 1,
            "step_count": 1000,
            "last_action_per_type": {at.value: 990 for at in ActionType},
            "action_posteriors": posteriors,
            "causal_chains": {"int:pridopidine": 10},  # already at depth 10 >= target_depth 5
        })
        for _ in range(30):
            action, _ = select_action_thompson(state, target_depth=5)
            # Should not pick DEEPEN when all chains are deep
            if action == ActionType.DEEPEN_CAUSAL_CHAIN:
                pytest.fail("Should have fallen back when all chains are at target depth")
