"""Tests for Phase 2: break hypothesis fixation.

Covers: weighted gap selection, recency penalty, prior hypothesis injection,
hypothesis deduplication, and max_active enforcement.
"""
from __future__ import annotations

import pytest
from research.state import initial_state, ResearchState
from research.hypotheses import create_hypothesis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(**overrides) -> ResearchState:
    s = initial_state(subject_ref="traj:draper_001")
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


class _MockEvidenceStore:
    """Returns configurable evidence counts per layer."""
    def __init__(self, layer_counts: dict[str, int] | None = None):
        self._counts = layer_counts or {}

    def query_by_protocol_layer(self, layer: str) -> list[dict]:
        n = self._counts.get(layer, 0)
        return [{"id": f"evi:{layer}_{i}"} for i in range(n)]

    def query_by_intervention_ref(self, int_id):
        return []

    def query_by_mechanism_target(self, target):
        return []

    def query_all_interventions(self):
        return []

    def upsert_object(self, obj):
        pass


# ===========================================================================
# 2A. Gap analysis returns multiple sparse layers (not just the minimum)
# ===========================================================================

class TestGapAnalysisMultipleLayers:

    def test_multiple_sparse_layers_returned(self):
        """When multiple layers have < 30 evidence, all should appear as gaps."""
        from research.intelligence import analyze_protocol_gaps
        state = _state()
        # All layers sparse but at different counts
        store = _MockEvidenceStore({
            "root_cause_suppression": 10,
            "pathology_reversal": 15,
            "circuit_stabilization": 20,
            "regeneration_reinnervation": 5,
            "adaptive_maintenance": 25,
        })
        gaps = analyze_protocol_gaps(state, store)
        sparse_gaps = [g for g in gaps if g["gap_type"] == "sparse_layer"]
        # Should have ALL 5 layers as sparse (all < 30)
        assert len(sparse_gaps) == 5
        # Layers should be present
        gap_layers = {g["layer"] for g in sparse_gaps}
        assert "regeneration_reinnervation" in gap_layers
        assert "root_cause_suppression" in gap_layers

    def test_recency_penalty_reduces_priority(self):
        """A layer targeted recently should have lower priority."""
        from research.intelligence import analyze_protocol_gaps
        state = _state()
        # Set last_gap_layers to penalize regeneration_reinnervation
        state.last_gap_layers = [
            "regeneration_reinnervation",
            "regeneration_reinnervation",
            "regeneration_reinnervation",
        ]
        store = _MockEvidenceStore({
            "root_cause_suppression": 10,
            "regeneration_reinnervation": 5,  # Fewer items but penalized by recency
        })
        gaps = analyze_protocol_gaps(state, store)
        sparse_gaps = [g for g in gaps if g["gap_type"] == "sparse_layer"]
        # regeneration should exist but with lower priority than root_cause
        regen_gap = next((g for g in sparse_gaps if g["layer"] == "regeneration_reinnervation"), None)
        root_gap = next((g for g in sparse_gaps if g["layer"] == "root_cause_suppression"), None)
        assert regen_gap is not None
        assert root_gap is not None
        # Recency penalty: 3 hits = 0.5^3 = 0.125x multiplier
        assert regen_gap["priority"] < root_gap["priority"]


# ===========================================================================
# 2B. last_gap_layers field exists on ResearchState
# ===========================================================================

class TestLastGapLayers:

    def test_field_exists_with_default(self):
        """ResearchState should have last_gap_layers defaulting to empty list."""
        state = initial_state(subject_ref="traj:draper_001")
        assert hasattr(state, "last_gap_layers")
        assert state.last_gap_layers == []

    def test_serialization_roundtrip(self):
        """last_gap_layers should survive to_dict/from_dict."""
        state = _state(last_gap_layers=["root_cause_suppression", "pathology_reversal"])
        d = state.to_dict()
        restored = ResearchState.from_dict(d)
        assert restored.last_gap_layers == ["root_cause_suppression", "pathology_reversal"]

    def test_backward_compat_missing_field(self):
        """Old state dicts without last_gap_layers should deserialize fine."""
        old_dict = initial_state(subject_ref="traj:draper_001").to_dict()
        del old_dict["last_gap_layers"]  # Simulate old format
        restored = ResearchState.from_dict(old_dict)
        assert restored.last_gap_layers == []


# ===========================================================================
# 2C. Prior hypothesis injection in prompt
# ===========================================================================

class TestPriorHypothesisInjection:

    def test_prompt_includes_prior_hypotheses(self):
        """When active_hypotheses is non-empty, prompt should include them."""
        from research.intelligence import build_hypothesis_prompt
        state = _state(active_hypotheses=[
            "STMN2 restoration enables axonal regeneration",
            "Sigma-1R agonism reduces ER stress in motor neurons",
        ])
        gap = {
            "gap_type": "sparse_layer",
            "layer": "regeneration_reinnervation",
            "description": "Layer has only 14 evidence items",
        }
        prompt = build_hypothesis_prompt(gap, state, [])
        assert "PRIOR HYPOTHESES" in prompt
        assert "STMN2 restoration" in prompt
        assert "DIFFERENT" in prompt

    def test_prompt_without_hypotheses_has_no_prior_section(self):
        """When active_hypotheses is empty, no PRIOR HYPOTHESES section."""
        from research.intelligence import build_hypothesis_prompt
        state = _state(active_hypotheses=[])
        gap = {
            "gap_type": "sparse_layer",
            "layer": "regeneration_reinnervation",
            "description": "Layer has only 14 evidence items",
        }
        prompt = build_hypothesis_prompt(gap, state, [])
        assert "PRIOR HYPOTHESES" not in prompt


# ===========================================================================
# 2D. Hypothesis deduplication
# ===========================================================================

class TestHypothesisDedup:

    def test_identical_statement_is_duplicate(self):
        from research.hypotheses import is_duplicate_hypothesis
        existing = ["STMN2 restoration enables axonal regeneration in ALS motor neurons"]
        assert is_duplicate_hypothesis(
            "STMN2 restoration enables axonal regeneration in ALS motor neurons",
            existing,
        ) is True

    def test_high_overlap_is_duplicate(self):
        from research.hypotheses import is_duplicate_hypothesis
        existing = ["STMN2 restoration enables axonal regeneration in ALS motor neurons"]
        # Minor rewording — still high Jaccard overlap
        assert is_duplicate_hypothesis(
            "Restoring STMN2 levels enables axonal regeneration for ALS motor neurons",
            existing,
            threshold=0.5,
        ) is True

    def test_different_topic_is_not_duplicate(self):
        from research.hypotheses import is_duplicate_hypothesis
        existing = ["STMN2 restoration enables axonal regeneration in ALS motor neurons"]
        assert is_duplicate_hypothesis(
            "Sigma-1R agonists reduce ER stress proteostasis dysfunction in spinal cord",
            existing,
        ) is False

    def test_empty_existing_is_not_duplicate(self):
        from research.hypotheses import is_duplicate_hypothesis
        assert is_duplicate_hypothesis(
            "Any hypothesis statement here",
            [],
        ) is False

    def test_threshold_respected(self):
        from research.hypotheses import is_duplicate_hypothesis
        existing = ["STMN2 restoration enables axonal regeneration in motor neurons"]
        # With very high threshold, even similar statements aren't duplicates
        assert is_duplicate_hypothesis(
            "STMN2 protein restoration promotes axonal regeneration in motor neurons",
            existing,
            threshold=0.95,
        ) is False


# ===========================================================================
# 2E. Max active hypotheses enforced
# ===========================================================================

class TestMaxActiveHypotheses:

    def test_hypothesis_not_appended_at_capacity(self):
        """When active_hypotheses is at max_active, new ones should not be appended."""
        from research.loop import research_step
        state = _state(
            active_hypotheses=[f"hyp:test_{i}" for i in range(10)],
            protocol_version=1,
        )

        class _MockStore(_MockEvidenceStore):
            def count_by_type(self, t):
                return 100

        new_state = research_step(
            state=state,
            evidence_store=_MockStore(),
            llm_manager=type("M", (), {
                "get_research_engine": lambda s: None,
                "get_protocol_engine": lambda s: None,
                "unload_protocol_model": lambda s: None,
            })(),
            dry_run=True,
        )
        # Should not exceed 10
        assert len(new_state.active_hypotheses) <= 10
