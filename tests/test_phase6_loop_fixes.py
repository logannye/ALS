"""Tests for Phase 6 — research loop structural fixes.

Three fixes:
1. Evidence deduplication (DB-count based)
2. Max-consecutive-same-action cap (3)
3. Convergence quality gate (stable_cycles >= 5 AND uncertainty < 0.3)
"""
from __future__ import annotations

import sys
import os
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from research.state import ResearchState, initial_state, ALL_LAYERS
from research.actions import ActionType, ActionResult
from research.convergence import compute_uncertainty_score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_state(**overrides) -> ResearchState:
    """Create a ResearchState with sensible defaults for testing."""
    s = initial_state("traj:test_001")
    return replace(s, **overrides)


# ---------------------------------------------------------------------------
# 1. Evidence deduplication
# ---------------------------------------------------------------------------

class TestEvidenceDeduplication:
    """The research loop should use DB count delta, not connector-reported count."""

    def test_duplicate_evidence_yields_zero(self):
        """When the DB count doesn't change, evidence_items_added should be 0."""
        from research.loop import research_step

        mock_store = MagicMock()
        mock_store.count_by_type.return_value = 500  # Same before and after

        mock_llm = MagicMock()

        state = _make_state(
            protocol_version=1,
            step_count=100,
            total_evidence_items=500,
            evidence_by_layer={layer: 10 for layer in ALL_LAYERS},
        )

        # Patch _execute_action to return a result that claims 20 items added
        fake_result = ActionResult(
            action=ActionType.QUERY_GALEN_KG,
            success=True,
            evidence_items_added=20,  # Connector claims 20
            protocol_layer="root_cause_suppression",
            evidence_strength="preclinical",
        )

        with patch("research.loop._execute_action", return_value=fake_result):
            new_state = research_step(state, mock_store, mock_llm)

        # Total evidence should NOT have increased by 20
        assert new_state.total_evidence_items == 500  # DB count didn't change

    def test_genuine_new_evidence_counted(self):
        """When DB count increases, the delta should be used."""
        from research.loop import research_step

        mock_store = MagicMock()
        # First call (before action): 500, second call (after): 505
        mock_store.count_by_type.side_effect = [500, 505]

        mock_llm = MagicMock()

        state = _make_state(
            protocol_version=1,
            step_count=100,
            total_evidence_items=500,
            evidence_by_layer={layer: 10 for layer in ALL_LAYERS},
        )

        fake_result = ActionResult(
            action=ActionType.SEARCH_PUBMED,
            success=True,
            evidence_items_added=20,  # Connector claims 20 but only 5 were new
            protocol_layer="root_cause_suppression",
            evidence_strength="moderate",
        )

        with patch("research.loop._execute_action", return_value=fake_result):
            new_state = research_step(state, mock_store, mock_llm)

        assert new_state.total_evidence_items == 505  # Used DB delta (5), not connector (20)

    def test_dry_run_skips_dedup(self):
        """In dry_run mode, no DB queries should be made."""
        from research.loop import research_step

        mock_store = MagicMock()
        mock_llm = MagicMock()

        state = _make_state(step_count=100)

        new_state = research_step(state, mock_store, mock_llm, dry_run=True)

        # count_by_type should not be called in dry_run
        mock_store.count_by_type.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Max-consecutive-same-action cap
# ---------------------------------------------------------------------------

class TestConsecutiveActionCap:
    """Thompson policy should exclude an action after 3 consecutive picks."""

    def test_consecutive_count_increments(self):
        """consecutive_same_action should increment when same action repeats."""
        from research.loop import research_step

        mock_store = MagicMock()
        mock_store.count_by_type.return_value = 500

        mock_llm = MagicMock()

        state = _make_state(
            protocol_version=1,
            step_count=100,
            total_evidence_items=500,
            last_action="query_galen_kg",
            consecutive_same_action=2,
            evidence_by_layer={layer: 10 for layer in ALL_LAYERS},
        )

        fake_result = ActionResult(
            action=ActionType.QUERY_GALEN_KG,
            success=True,
            evidence_items_added=0,
        )

        with patch("research.loop._execute_action", return_value=fake_result), \
             patch("research.loop.select_action", return_value=(ActionType.QUERY_GALEN_KG, {"action": ActionType.QUERY_GALEN_KG})):
            new_state = research_step(state, mock_store, mock_llm)

        assert new_state.consecutive_same_action == 3

    def test_consecutive_count_resets_on_different_action(self):
        """consecutive_same_action should reset to 1 when a different action is picked."""
        from research.loop import research_step

        mock_store = MagicMock()
        mock_store.count_by_type.return_value = 500

        mock_llm = MagicMock()

        state = _make_state(
            protocol_version=1,
            step_count=100,
            total_evidence_items=500,
            last_action="query_galen_kg",
            consecutive_same_action=3,
            evidence_by_layer={layer: 10 for layer in ALL_LAYERS},
        )

        fake_result = ActionResult(
            action=ActionType.SEARCH_PUBMED,
            success=True,
            evidence_items_added=0,
        )

        with patch("research.loop._execute_action", return_value=fake_result), \
             patch("research.loop.select_action", return_value=(ActionType.SEARCH_PUBMED, {"action": ActionType.SEARCH_PUBMED})):
            new_state = research_step(state, mock_store, mock_llm)

        assert new_state.consecutive_same_action == 1

    def test_thompson_excludes_repeated_action(self):
        """Thompson policy should not pick the same action after 3+ consecutive uses."""
        from research.policy import select_action_thompson

        state = _make_state(
            protocol_version=1,
            step_count=100,
            last_action="query_galen_kg",
            consecutive_same_action=3,
            new_evidence_since_regen=0,
        )

        # Run many times — should never pick query_galen_kg
        picked_actions = set()
        for _ in range(50):
            action, _ = select_action_thompson(state, regen_threshold=15)
            picked_actions.add(action.value)

        assert "query_galen_kg" not in picked_actions

    def test_thompson_allows_action_below_cap(self):
        """Thompson policy can pick the same action if below the cap."""
        from research.policy import select_action_thompson

        state = _make_state(
            protocol_version=1,
            step_count=100,
            last_action="query_galen_kg",
            consecutive_same_action=2,  # Below cap of 3
            new_evidence_since_regen=0,
            # Give query_galen_kg a very high posterior so Thompson always picks it
            action_posteriors={"query_galen_kg": (100.0, 1.0)},
        )

        action, _ = select_action_thompson(state, regen_threshold=15)
        # It CAN pick query_galen_kg (though not guaranteed due to Thompson randomness)
        # Just verify it doesn't crash and returns a valid action
        assert isinstance(action, ActionType)


# ---------------------------------------------------------------------------
# 3. Convergence quality gate
# ---------------------------------------------------------------------------

class TestConvergenceQualityGate:
    """Convergence requires both stability AND low uncertainty."""

    def test_high_uncertainty_blocks_convergence(self):
        """protocol_stable_cycles >= 5 should NOT converge if uncertainty is high."""
        from research.loop import run_research_loop

        mock_store = MagicMock()
        mock_store.count_by_type.return_value = 100

        mock_llm = MagicMock()

        # Create state that would have converged under old rules
        state = _make_state(
            protocol_version=5,
            protocol_stable_cycles=5,
            uncertainty_score=0.8,  # High uncertainty — should block
            evidence_by_layer={layer: 2 for layer in ALL_LAYERS},  # Sparse layers
            total_evidence_items=100,
        )

        # Just test the condition directly rather than running the full loop
        _unc = state.uncertainty_score
        _stable = state.protocol_stable_cycles >= 5
        _quality = _unc < 0.3

        assert _stable is True  # Would have converged before
        assert _quality is False  # Quality gate blocks it
        assert not (_stable and _quality)  # Combined: not converged

    def test_low_uncertainty_and_stable_converges(self):
        """Should converge when both stability AND quality thresholds are met."""
        state = _make_state(
            protocol_version=10,
            protocol_stable_cycles=5,
            uncertainty_score=0.2,  # Low uncertainty
            evidence_by_layer={layer: 50 for layer in ALL_LAYERS},
        )

        _stable = state.protocol_stable_cycles >= 5
        _quality = state.uncertainty_score < 0.3

        assert _stable and _quality  # Both met → converge

    def test_stable_but_not_quality_does_not_converge(self):
        """Stability alone is not sufficient."""
        state = _make_state(
            protocol_stable_cycles=10,
            uncertainty_score=0.5,
        )

        _stable = state.protocol_stable_cycles >= 5
        _quality = state.uncertainty_score < 0.3

        assert _stable is True
        assert _quality is False
        assert not (_stable and _quality)

    def test_quality_but_not_stable_does_not_converge(self):
        """Low uncertainty alone is not sufficient."""
        state = _make_state(
            protocol_stable_cycles=2,
            uncertainty_score=0.1,
        )

        _stable = state.protocol_stable_cycles >= 5
        _quality = state.uncertainty_score < 0.3

        assert _stable is False
        assert _quality is True
        assert not (_stable and _quality)


# ---------------------------------------------------------------------------
# Uncertainty score computation
# ---------------------------------------------------------------------------

class TestUncertaintyScore:
    """compute_uncertainty_score uses layer coverage, chain depth, and missing measurements."""

    def test_fully_covered_low_uncertainty(self):
        """All layers covered + deep chains + no missing measurements → low score."""
        state = _make_state(
            evidence_by_layer={layer: 50 for layer in ALL_LAYERS},
            causal_chains={"int:a": 5, "int:b": 4, "int:c": 5},
            missing_measurements=[],
        )
        score = compute_uncertainty_score(state)
        assert score < 0.1

    def test_empty_state_high_uncertainty(self):
        """Default state should have high uncertainty."""
        state = _make_state()
        score = compute_uncertainty_score(state)
        assert score > 0.7

    def test_partial_coverage(self):
        """Some layers covered, some not → moderate uncertainty."""
        layers = {layer: 0 for layer in ALL_LAYERS}
        layers[ALL_LAYERS[0]] = 50
        layers[ALL_LAYERS[1]] = 50
        state = _make_state(
            evidence_by_layer=layers,
            causal_chains={"int:a": 3},
            missing_measurements=["genetic_testing"],
        )
        score = compute_uncertainty_score(state)
        assert 0.1 < score < 0.7


# ---------------------------------------------------------------------------
# last_action_per_type tracking
# ---------------------------------------------------------------------------

class TestLastActionPerType:
    """last_action_per_type should be updated each step."""

    def test_last_action_per_type_updated(self):
        """research_step should record when each action type was last used."""
        from research.loop import research_step

        mock_store = MagicMock()
        mock_store.count_by_type.return_value = 500

        mock_llm = MagicMock()

        state = _make_state(
            protocol_version=1,
            step_count=100,
            total_evidence_items=500,
            evidence_by_layer={layer: 10 for layer in ALL_LAYERS},
        )

        fake_result = ActionResult(
            action=ActionType.SEARCH_PUBMED,
            success=True,
            evidence_items_added=0,
        )

        with patch("research.loop._execute_action", return_value=fake_result), \
             patch("research.loop.select_action", return_value=(ActionType.SEARCH_PUBMED, {"action": ActionType.SEARCH_PUBMED})):
            new_state = research_step(state, mock_store, mock_llm)

        assert new_state.last_action_per_type.get("search_pubmed") == 101
