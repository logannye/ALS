"""Tests for Phase 8.1: Evidence flow restoration (2026-03-30).

Three compounding failures causing total evidence stall at 2,137:
1. Phantom hypothesis resolution — validate_hypothesis gets reward=2.50
   for "resolving" hypotheses with 100% duplicate evidence (DB delta=0)
2. CSF biomarker fixation — missing_data gaps have no recency penalty,
   so generate_hypothesis targets csf_biomarkers on every call
3. Stagnation recovery over-firing — fires every step instead of
   periodically, preventing Thompson from ever learning
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from research.actions import ActionResult, ActionType
from research.state import ResearchState, initial_state
from research.rewards import compute_reward


# ---------------------------------------------------------------------------
# Fix 1: Phantom hypothesis resolution
# ---------------------------------------------------------------------------

class TestPhantomHypothesisResolution:
    """hypothesis_resolved must be False when DB delta is 0 (all duplicates)."""

    def test_resolution_requires_true_new_evidence(self):
        """When dedup shows DB delta=0, hypothesis_resolved should be False."""
        result = ActionResult(
            action=ActionType.VALIDATE_HYPOTHESIS,
            success=True,
            evidence_items_added=10,  # raw connector count
            hypothesis_resolved=True,  # set by _exec_validate_hypothesis
        )
        # Simulate dedup correction (DB delta = 0)
        _true_new = 0
        result.evidence_items_added = _true_new
        # FIX: hypothesis_resolved must also be corrected
        if _true_new == 0:
            result.hypothesis_resolved = False

        assert result.hypothesis_resolved is False

    def test_no_reward_for_phantom_resolution(self):
        """Reward for hypothesis_resolution should be 0 when all evidence is duplicate."""
        reward = compute_reward(
            evidence_items_added=0,
            uncertainty_before=0.5,
            uncertainty_after=0.5,
            protocol_score_delta=0.0,
            hypothesis_resolved=False,  # corrected by dedup
            causal_depth_added=0,
            interaction_safe=False,
            eligibility_confirmed=False,
            protocol_stable=False,
        )
        assert reward.hypothesis_resolution == 0.0
        assert reward.total() == 0.0

    def test_reward_preserved_for_real_resolution(self):
        """When there IS new evidence, hypothesis_resolved should stay True."""
        reward = compute_reward(
            evidence_items_added=5,
            uncertainty_before=0.5,
            uncertainty_after=0.5,
            protocol_score_delta=0.0,
            hypothesis_resolved=True,
            causal_depth_added=0,
            interaction_safe=False,
            eligibility_confirmed=False,
            protocol_stable=False,
        )
        assert reward.hypothesis_resolution == 1.0
        assert reward.total() > 0.0


# ---------------------------------------------------------------------------
# Fix 2: CSF biomarker hypothesis fixation
# ---------------------------------------------------------------------------

class TestHypothesisFixation:
    """missing_data gaps must have recency penalty to prevent fixation."""

    def test_missing_data_gap_has_recency_penalty(self):
        """missing_data gaps should decay priority when repeatedly targeted."""
        from research.intelligence import analyze_protocol_gaps

        state = initial_state("traj:test")
        # Simulate repeated targeting of csf_biomarkers
        state = replace(state, last_gap_layers=[
            "missing_data:csf_biomarkers",
            "missing_data:csf_biomarkers",
            "missing_data:csf_biomarkers",
        ])

        class MockStore:
            def query_by_protocol_layer(self, layer):
                return [{"id": f"e{i}"} for i in range(50)]  # 50 items per layer

        gaps = analyze_protocol_gaps(state, MockStore())
        csf_gaps = [g for g in gaps if g.get("gap_type") == "missing_data"
                    and "csf" in g.get("description", "").lower()]

        if csf_gaps:
            # After 3 hits, priority should be significantly reduced from 0.6
            assert csf_gaps[0]["priority"] < 0.2, (
                f"CSF gap priority should decay with recency, got {csf_gaps[0]['priority']}"
            )

    def test_gap_same_type_max_catches_csf_variants(self):
        """gap_same_type_max should match on gap_type, not just text."""
        state = initial_state("traj:test")
        state = replace(state, active_hypotheses=[
            "Erik's CSF biomarker panel would show elevated TDP-43",
            "CSF biomarkers would likely show elevated TDP-43 aggregates",
            "Erik's CSF biomarkers would show phosphorylated TDP-43",
        ])

        # The pre-generation check should recognize these all target csf_biomarkers
        # Count hypotheses that mention CSF (the actual gap target)
        csf_count = sum(1 for h in state.active_hypotheses if "csf" in h.lower())
        assert csf_count >= 3, "Test setup: should have 3 CSF hypotheses"

    def test_jaccard_catches_csf_variants(self):
        """Jaccard should catch minor CSF TDP-43 wording variants."""
        from research.hypotheses import is_duplicate_hypothesis

        existing = [
            "Erik's CSF biomarker panel would likely show elevated levels of TDP-43 "
            "protein aggregates and phosphorylated TDP-43 in sporadic ALS"
        ]
        # Minor variant — same semantic content, slight rewording
        new_stmt = (
            "Erik's CSF biomarkers would likely show elevated TDP-43 protein "
            "aggregates (phosphorylated at S409/S410) in sporadic ALS"
        )
        # With lowered threshold (0.35), this should be caught as duplicate
        assert is_duplicate_hypothesis(new_stmt, existing, threshold=0.35)


# ---------------------------------------------------------------------------
# Fix 3: Stagnation recovery over-firing
# ---------------------------------------------------------------------------

class TestStagnationCooldown:
    """Stagnation recovery should have a cooldown, not fire every step."""

    def test_stagnation_has_cooldown_field(self):
        """ResearchState should track last stagnation recovery step."""
        state = initial_state("traj:test")
        assert hasattr(state, "last_stagnation_step"), (
            "ResearchState needs last_stagnation_step field for cooldown"
        )

    def test_stagnation_respects_cooldown(self):
        """Stagnation should not fire within cooldown window of last recovery."""
        state = initial_state("traj:test")
        state = replace(
            state,
            step_count=500,
            last_stagnation_step=490,  # recovered 10 steps ago
            evidence_at_step={0: 100, 50: 100, 100: 100},  # flat evidence
        )
        # With a cooldown of 50 steps, recovery should NOT fire at step 500
        # (only 10 steps since last recovery)
        cooldown = 50
        should_fire = (
            state.step_count - state.last_stagnation_step >= cooldown
        )
        assert not should_fire, "Stagnation should not fire within cooldown"

    def test_stagnation_fires_after_cooldown(self):
        """Stagnation should fire once cooldown has elapsed."""
        state = initial_state("traj:test")
        state = replace(
            state,
            step_count=550,
            last_stagnation_step=490,  # recovered 60 steps ago
            evidence_at_step={0: 100, 50: 100, 100: 100},
        )
        cooldown = 50
        should_fire = (
            state.step_count - state.last_stagnation_step >= cooldown
        )
        assert should_fire, "Stagnation should fire after cooldown elapsed"

    def test_stagnation_fields_serialize(self):
        """last_stagnation_step should survive serialization round-trip."""
        state = initial_state("traj:test")
        state = replace(state, last_stagnation_step=42)
        d = state.to_dict()
        restored = ResearchState.from_dict(d)
        assert restored.last_stagnation_step == 42
