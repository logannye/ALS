"""Tests for production deadlock fixes (2026-03-30).

Five fixes to break Erik's hypothesis deadlock and restore evidence flow:
1. Thompson path includes VALIDATE_HYPOTHESIS and calls hypothesis expiry
2. Dedup zeroes causal_depth_added when DB delta is 0 (Galen SCM inflation)
3. DrugBank encoding fix (errors="replace")
4. CHALLENGE_INTERVENTION executor wired up
5. Evidence stagnation detection and recovery
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from research.actions import ActionResult, ActionType, build_action_params
from research.state import ResearchState, initial_state
from research.policy import (
    _action_is_feasible,
    _maybe_expire_hypotheses,
    _build_thompson_params,
    select_action_thompson,
)


# ---------------------------------------------------------------------------
# Fix 1: Hypothesis deadlock — Thompson path VALIDATE_HYPOTHESIS
# ---------------------------------------------------------------------------

class TestThompsonValidationIntegration:
    """Thompson path must include VALIDATE_HYPOTHESIS to prevent deadlock."""

    def test_validate_in_thompson_when_hypotheses_exist(self):
        """Thompson should sometimes select VALIDATE_HYPOTHESIS when hypotheses exist."""
        state = initial_state("traj:test")
        state = ResearchState(**{
            **state.to_dict(),
            "protocol_version": 1,
            "step_count": 100,
            "active_hypotheses": ["TDP-43 aggregation causes motor neuron death"],
            "last_action_per_type": {at.value: 99 for at in ActionType},
        })
        actions = set()
        for i in range(200):
            test_state = ResearchState(**{
                **state.to_dict(),
                "step_count": 100 + i,
                "last_action_per_type": {at.value: 100 + i - 1 for at in ActionType},
            })
            action, _ = select_action_thompson(test_state, regen_threshold=999)
            actions.add(action.value)
        assert "validate_hypothesis" in actions, (
            f"Expected validate_hypothesis in Thompson selections, got: {actions}"
        )

    def test_validate_not_selected_when_no_hypotheses(self):
        """VALIDATE_HYPOTHESIS should be infeasible with no active hypotheses."""
        state = initial_state("traj:test")
        assert not _action_is_feasible(ActionType.VALIDATE_HYPOTHESIS, state, target_depth=5)

    def test_validate_feasible_when_hypotheses_exist(self):
        """VALIDATE_HYPOTHESIS should be feasible when hypotheses exist."""
        state = initial_state("traj:test")
        state = ResearchState(**{
            **state.to_dict(),
            "active_hypotheses": ["TDP-43 aggregation causes motor neuron death"],
        })
        assert _action_is_feasible(ActionType.VALIDATE_HYPOTHESIS, state, target_depth=5)

    def test_build_thompson_params_validate(self):
        """_build_thompson_params should handle VALIDATE_HYPOTHESIS."""
        state = initial_state("traj:test")
        state = ResearchState(**{
            **state.to_dict(),
            "active_hypotheses": ["SOD1 misfolding drives oxidative stress"],
        })
        action, params = _build_thompson_params(
            ActionType.VALIDATE_HYPOTHESIS, state, target_depth=5,
        )
        assert action == ActionType.VALIDATE_HYPOTHESIS
        assert params.get("hypothesis_id") == "SOD1 misfolding drives oxidative stress"

    def test_build_thompson_params_validate_fallback_no_hyps(self):
        """When VALIDATE_HYPOTHESIS wins but no hypotheses exist, fallback to acquisition."""
        state = initial_state("traj:test")
        action, params = _build_thompson_params(
            ActionType.VALIDATE_HYPOTHESIS, state, target_depth=5,
        )
        # Should fall back to an acquisition action
        assert action != ActionType.VALIDATE_HYPOTHESIS


# ---------------------------------------------------------------------------
# Fix 1: Hypothesis expiry at max_active
# ---------------------------------------------------------------------------

class TestHypothesisExpiryAtMaxActive:
    """_maybe_expire_hypotheses must force-expire at max_active to prevent deadlock."""

    def test_force_expire_at_max_active(self):
        """When hypotheses are at max_active (10), oldest should be expired."""
        state = initial_state("traj:test")
        state = ResearchState(**{
            **state.to_dict(),
            "active_hypotheses": [f"hypothesis {i}" for i in range(10)],
        })
        _maybe_expire_hypotheses(state)
        assert len(state.active_hypotheses) == 9
        assert state.active_hypotheses[0] == "hypothesis 1"  # oldest expired
        assert state.resolved_hypotheses == 1

    def test_no_expire_below_max_active(self):
        """Below max_active, force-expire should not fire (only ratio-based)."""
        state = initial_state("traj:test")
        state = ResearchState(**{
            **state.to_dict(),
            "active_hypotheses": [f"hypothesis {i}" for i in range(5)],
            "action_counts": {"validate_hypothesis": 0},
        })
        _maybe_expire_hypotheses(state)
        # No expiry: below max_active and validate_hypothesis count is 0
        assert len(state.active_hypotheses) == 5

    def test_expire_empty_hypotheses_is_noop(self):
        """_maybe_expire_hypotheses should be a no-op with empty list."""
        state = initial_state("traj:test")
        _maybe_expire_hypotheses(state)
        assert len(state.active_hypotheses) == 0

    def test_thompson_calls_expire(self):
        """select_action_thompson must call _maybe_expire_hypotheses.

        If hypotheses are at max_active before the call, after Thompson
        runs they should be reduced (proving expiry was called).
        """
        state = initial_state("traj:test")
        state = ResearchState(**{
            **state.to_dict(),
            "protocol_version": 1,
            "step_count": 100,
            "active_hypotheses": [f"hypothesis {i}" for i in range(10)],
            "last_action_per_type": {at.value: 99 for at in ActionType},
        })
        # select_action_thompson mutates state.active_hypotheses via _maybe_expire_hypotheses
        select_action_thompson(state, regen_threshold=999)
        assert len(state.active_hypotheses) <= 9, (
            "Expected _maybe_expire_hypotheses to fire from Thompson path"
        )


# ---------------------------------------------------------------------------
# Fix 2: Galen SCM causal_depth_added dedup correction
# ---------------------------------------------------------------------------

class TestCausalDepthDedup:
    """causal_depth_added must be zeroed when DB delta shows no new evidence."""

    def test_causal_depth_zeroed_on_zero_dedup(self):
        """When evidence is all duplicates, causal_depth_added should be 0."""
        result = ActionResult(
            action=ActionType.QUERY_GALEN_SCM,
            success=True,
            evidence_items_added=100,  # Pre-dedup: connector reported 100
            causal_depth_added=1,
        )
        # Simulate dedup: DB delta = 0 (all upserts)
        _true_new = 0
        result.evidence_items_added = _true_new
        if _true_new == 0 and result.causal_depth_added > 0:
            result.causal_depth_added = 0

        assert result.evidence_items_added == 0
        assert result.causal_depth_added == 0

    def test_thompson_success_false_after_dedup(self):
        """After dedup correction, Thompson success should be False."""
        result = ActionResult(
            action=ActionType.QUERY_GALEN_SCM,
            success=True,
            evidence_items_added=0,
            causal_depth_added=0,
        )
        _success = (
            result.evidence_items_added > 0
            or result.hypothesis_generated is not None
            or result.hypothesis_resolved
            or result.causal_depth_added > 0
            or result.protocol_regenerated
        )
        assert _success is False

    def test_causal_depth_preserved_when_new_evidence(self):
        """When there IS new evidence, causal_depth_added should be preserved."""
        result = ActionResult(
            action=ActionType.QUERY_GALEN_SCM,
            success=True,
            evidence_items_added=5,
            causal_depth_added=1,
        )
        # Simulate dedup: DB delta = 3 (some new)
        _true_new = 3
        result.evidence_items_added = _true_new
        if _true_new == 0 and result.causal_depth_added > 0:
            result.causal_depth_added = 0

        assert result.evidence_items_added == 3
        assert result.causal_depth_added == 1  # preserved


# ---------------------------------------------------------------------------
# Fix 3: DrugBank encoding
# ---------------------------------------------------------------------------

class TestDrugBankEncoding:
    """DrugBank connector must handle non-UTF-8 bytes."""

    def test_vocabulary_with_latin1_bytes(self, tmp_path):
        """File with non-UTF-8 bytes (0xb0 = degree symbol) should not crash."""
        import connectors.drugbank_local as dbl

        vocab = tmp_path / "vocab.csv"
        vocab.write_bytes(
            b"DrugBank ID,Common name,CAS,UNII,Synonyms\n"
            b"DB00001,Test Drug 50\xb0C,123-45-6,ABC123,syn1\n"
        )
        original_vocab = dbl._VOCAB_PATH
        original_links = dbl._TARGET_LINKS_PATH
        try:
            dbl._VOCAB_PATH = str(vocab)
            # Create empty target links file
            links = tmp_path / "links.csv"
            links.write_text("UniProt ID,DrugBank ID\n")
            dbl._TARGET_LINKS_PATH = str(links)

            connector = dbl.DrugBankLocalConnector()
            connector._load()  # Should not raise
            assert "DB00001" in connector._vocab
        finally:
            dbl._VOCAB_PATH = original_vocab
            dbl._TARGET_LINKS_PATH = original_links

    def test_target_links_with_latin1_bytes(self, tmp_path):
        """Target links with non-UTF-8 bytes should not crash."""
        import connectors.drugbank_local as dbl

        vocab = tmp_path / "vocab.csv"
        vocab.write_text("DrugBank ID,Common name,CAS,UNII,Synonyms\nDB00001,Test,123,ABC,syn\n")
        links = tmp_path / "links.csv"
        links.write_bytes(b"UniProt ID,DrugBank ID\nQ13148,DB00001\xb0\n")

        original_vocab = dbl._VOCAB_PATH
        original_links = dbl._TARGET_LINKS_PATH
        try:
            dbl._VOCAB_PATH = str(vocab)
            dbl._TARGET_LINKS_PATH = str(links)

            connector = dbl.DrugBankLocalConnector()
            connector._load()
            assert "Q13148" in connector._target_links
        finally:
            dbl._VOCAB_PATH = original_vocab
            dbl._TARGET_LINKS_PATH = original_links


# ---------------------------------------------------------------------------
# Fix 4: CHALLENGE_INTERVENTION executor
# ---------------------------------------------------------------------------

class TestChallengeInterventionDispatch:
    """CHALLENGE_INTERVENTION must be in the dispatch table."""

    def test_challenge_in_dispatch(self):
        """The dispatch table should include CHALLENGE_INTERVENTION."""
        from research.loop import _execute_action

        state = initial_state("traj:test")
        state = ResearchState(**{
            **state.to_dict(),
            "causal_chains": {"int:riluzole": 3},
        })

        class MockStore:
            def upsert_object(self, obj):
                pass
            def count_by_type(self, t):
                return 0
            def query_by_intervention_ref(self, ref):
                return []

        result = _execute_action(
            ActionType.CHALLENGE_INTERVENTION, {},
            state, MockStore(), None,
        )
        assert "Unknown action" not in (result.error or ""), (
            f"CHALLENGE_INTERVENTION should be in dispatch, got error: {result.error}"
        )


# ---------------------------------------------------------------------------
# Fix 5: Stagnation detection
# ---------------------------------------------------------------------------

class TestStagnationDetection:
    """Stagnation should be detected and recovery triggered."""

    def test_state_has_stagnation_fields(self):
        """ResearchState should have evidence_at_step and stagnation_resets."""
        state = initial_state("traj:test")
        assert hasattr(state, "evidence_at_step")
        assert hasattr(state, "stagnation_resets")
        assert state.evidence_at_step == {}
        assert state.stagnation_resets == 0

    def test_stagnation_detected_when_flat(self):
        """Stagnation logic: no growth over window should trigger recovery."""
        # Simulate the stagnation detection logic
        evidence_at_step = {0: 100, 50: 100, 100: 100, 150: 100, 200: 100}
        current_step = 250
        window = 200
        min_growth = 5

        # Find reference point
        ref_candidates = [k for k in evidence_at_step if k <= current_step - window]
        ref_step = max(ref_candidates) if ref_candidates else None
        growth = 100 - evidence_at_step[ref_step] if ref_step is not None else 999

        assert ref_step == 50
        assert growth == 0
        assert growth < min_growth  # Stagnation detected

    def test_no_stagnation_with_growth(self):
        """When evidence is growing, stagnation should NOT be detected."""
        evidence_at_step = {0: 50, 50: 75, 100: 100, 150: 130, 200: 170}
        current_step = 250
        window = 200

        ref_candidates = [k for k in evidence_at_step if k <= current_step - window]
        ref_step = max(ref_candidates) if ref_candidates else None
        growth = 170 - evidence_at_step[ref_step] if ref_step is not None else 999

        assert ref_step == 50
        assert growth == 95
        assert growth >= 5  # No stagnation

    def test_stagnation_fields_in_to_dict(self):
        """New stagnation fields should survive serialization round-trip."""
        state = initial_state("traj:test")
        state = replace(state, evidence_at_step={100: 500}, stagnation_resets=2)
        d = state.to_dict()
        restored = ResearchState.from_dict(d)
        assert restored.evidence_at_step == {100: 500}
        assert restored.stagnation_resets == 2
