"""Tests for 8-component reward computation."""
from __future__ import annotations
import pytest
from research.rewards import compute_reward, RewardComponents

class TestRewardComponents:
    def test_model_construction(self):
        rc = RewardComponents(evidence_gain=3.0, uncertainty_reduction=2.0)
        assert rc.evidence_gain == 3.0
        assert rc.uncertainty_reduction == 2.0

    def test_defaults_are_zero(self):
        rc = RewardComponents()
        assert rc.evidence_gain == 0.0
        assert rc.total() == 0.0

    def test_total_is_weighted_sum(self):
        rc = RewardComponents(evidence_gain=1.0)
        total = rc.total()
        assert total > 0.0  # evidence_gain weight is 3.0

    def test_to_dict(self):
        rc = RewardComponents(evidence_gain=1.0, hypothesis_resolution=0.5)
        d = rc.to_dict()
        assert "evidence_gain" in d
        assert "total" in d

class TestComputeReward:
    def test_no_new_evidence_zero_gain(self):
        rc = compute_reward(
            evidence_items_added=0, uncertainty_before=0.5, uncertainty_after=0.5,
            protocol_score_delta=0.0, hypothesis_resolved=False, causal_depth_added=0,
            interaction_safe=False, eligibility_confirmed=False, protocol_stable=False,
        )
        assert rc.evidence_gain == 0.0

    def test_evidence_gain_diminishing(self):
        rc1 = compute_reward(evidence_items_added=1, uncertainty_before=0.5, uncertainty_after=0.5,
                             protocol_score_delta=0.0, hypothesis_resolved=False, causal_depth_added=0,
                             interaction_safe=False, eligibility_confirmed=False, protocol_stable=False)
        rc5 = compute_reward(evidence_items_added=5, uncertainty_before=0.5, uncertainty_after=0.5,
                             protocol_score_delta=0.0, hypothesis_resolved=False, causal_depth_added=0,
                             interaction_safe=False, eligibility_confirmed=False, protocol_stable=False)
        assert rc5.evidence_gain > rc1.evidence_gain
        assert rc5.evidence_gain < rc1.evidence_gain * 5

    def test_uncertainty_reduction_rewarded(self):
        rc = compute_reward(evidence_items_added=0, uncertainty_before=0.5, uncertainty_after=0.3,
                            protocol_score_delta=0.0, hypothesis_resolved=False, causal_depth_added=0,
                            interaction_safe=False, eligibility_confirmed=False, protocol_stable=False)
        assert rc.uncertainty_reduction > 0.0

    def test_hypothesis_resolution_rewarded(self):
        rc = compute_reward(evidence_items_added=0, uncertainty_before=0.5, uncertainty_after=0.5,
                            protocol_score_delta=0.0, hypothesis_resolved=True, causal_depth_added=0,
                            interaction_safe=False, eligibility_confirmed=False, protocol_stable=False)
        assert rc.hypothesis_resolution > 0.0
