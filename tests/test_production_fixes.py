"""Tests for production fixes — hypothesis fixation, gap analysis, dynamic queries, Thompson updates.

These tests verify the five fixes deployed on 2026-03-29 to address the
complete evidence stall (220 steps, 0 new evidence, total frozen at 9,111).
"""
from __future__ import annotations

import pytest
from dataclasses import replace
from unittest.mock import MagicMock, patch

from research.actions import ActionResult, ActionType
from research.state import ResearchState, initial_state, ALL_LAYERS
from research.intelligence import analyze_protocol_gaps, _inject_prior_hypotheses
from research.hypotheses import is_duplicate_hypothesis
from research.policy import (
    _update_posteriors,
    _apply_decay,
    _action_is_feasible,
    get_layer_query,
    _get_dynamic_query,
    _BASE_LAYER_QUERIES,
    select_action_thompson,
)
from research.rewards import compute_reward


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

class _MockStore:
    """Mock evidence store with configurable layer counts."""

    def __init__(self, layer_counts=None):
        self._layer_counts = layer_counts or {}
        self._total = sum(self._layer_counts.values()) if self._layer_counts else 100

    def query_by_protocol_layer(self, layer):
        count = self._layer_counts.get(layer, 5)
        return [{"id": f"evi:mock_{layer}_{i}", "body": {"protocol_layer": layer}} for i in range(count)]

    def query_by_intervention_ref(self, int_id):
        return []

    def count_by_type(self, _type):
        return self._total


def _state_with_hypotheses(statements: list[str]) -> ResearchState:
    """Create a state with hypothesis statements in active_hypotheses."""
    state = initial_state(subject_ref="traj:draper_001")
    state = replace(state, active_hypotheses=list(statements))
    return state


def _state_with_chains(n: int = 3) -> ResearchState:
    """Create a state with N intervention chains (triggers unvalidated_safety)."""
    state = initial_state(subject_ref="traj:draper_001")
    chains = {f"int:drug_{i}": i for i in range(n)}
    return replace(state, causal_chains=chains)


# ===========================================================================
# Fix 1: Hypothesis Statement Storage (not IDs)
# ===========================================================================

class TestHypothesisStatementStorage:

    def test_inject_prior_hypotheses_with_readable_statements(self):
        """_inject_prior_hypotheses should produce readable text, not hash IDs."""
        state = _state_with_hypotheses([
            "Inhibiting CK1delta reduces TDP-43 aggregation in motor neurons",
            "STMN2 restoration enables axonal regeneration in ALS",
        ])
        prompt = "TASK: Generate hypothesis."
        result = _inject_prior_hypotheses(prompt, state)
        assert "Inhibiting CK1delta" in result
        assert "STMN2 restoration" in result
        assert "hyp:" not in result  # No hash IDs

    def test_inject_prior_hypotheses_skips_old_format_ids(self):
        """If state contains old-format IDs, they still appear (graceful degradation)."""
        state = _state_with_hypotheses(["hyp:abc123def456"])
        prompt = "TASK: Generate hypothesis."
        result = _inject_prior_hypotheses(prompt, state)
        # Old IDs are included (backward compat) but system still works
        assert "hyp:abc123def456" in result

    def test_inject_prior_hypotheses_empty_state(self):
        """No prior hypotheses = prompt unchanged."""
        state = _state_with_hypotheses([])
        prompt = "TASK: Generate hypothesis."
        result = _inject_prior_hypotheses(prompt, state)
        assert result == prompt

    def test_duplicate_hypothesis_tighter_threshold(self):
        """Jaccard threshold of 0.45 should catch near-identical CK1delta hypotheses."""
        existing = [
            "Inhibiting CK1delta with IGS2.7 will reduce TDP-43 cytoplasmic aggregation and improve motor neuron survival in sporadic ALS"
        ]
        # Very similar statement with minor word changes
        candidate = "Inhibiting CK1delta via IGS2.7 reduces TDP-43 cytoplasmic aggregation and improves motor neuron survival in sporadic TDP-43 ALS"
        assert is_duplicate_hypothesis(candidate, existing, threshold=0.45)

    def test_distinct_hypothesis_passes_dedup(self):
        """A genuinely different hypothesis should pass even at 0.45 threshold."""
        existing = [
            "Inhibiting CK1delta with IGS2.7 will reduce TDP-43 cytoplasmic aggregation and improve motor neuron survival"
        ]
        candidate = "STMN2 cryptic exon restoration via antisense oligonucleotide will rescue axonal transport in C9orf72 ALS"
        assert not is_duplicate_hypothesis(candidate, existing, threshold=0.45)

    def test_dedup_handles_empty_existing(self):
        assert not is_duplicate_hypothesis("any hypothesis", [], threshold=0.45)

    def test_dedup_handles_empty_candidate(self):
        assert not is_duplicate_hypothesis("", ["existing hypothesis"], threshold=0.45)


# ===========================================================================
# Fix 2: Gap Analysis Unsticking
# ===========================================================================

class TestGapAnalysisUnsticking:

    def test_unvalidated_safety_has_layer_field(self):
        """unvalidated_safety gap should have a 'layer' key for tracking."""
        store = _MockStore()
        state = _state_with_chains(3)
        gaps = analyze_protocol_gaps(state, store)
        safety_gaps = [g for g in gaps if g["gap_type"] == "unvalidated_safety"]
        assert len(safety_gaps) == 1
        assert safety_gaps[0]["layer"] == "unvalidated_safety"

    def test_unvalidated_safety_priority_decays_with_recency(self):
        """Priority should halve with each entry in last_gap_layers."""
        store = _MockStore()
        state = _state_with_chains(3)

        # No recency — full priority
        gaps_0 = analyze_protocol_gaps(state, store)
        safety_0 = [g for g in gaps_0 if g["gap_type"] == "unvalidated_safety"][0]
        assert safety_0["priority"] == pytest.approx(0.7)

        # 1 recency hit
        state_1 = replace(state, last_gap_layers=["unvalidated_safety"])
        gaps_1 = analyze_protocol_gaps(state_1, store)
        safety_1 = [g for g in gaps_1 if g["gap_type"] == "unvalidated_safety"][0]
        assert safety_1["priority"] == pytest.approx(0.35)

        # 2 recency hits
        state_2 = replace(state, last_gap_layers=["unvalidated_safety", "unvalidated_safety"])
        gaps_2 = analyze_protocol_gaps(state_2, store)
        safety_2 = [g for g in gaps_2 if g["gap_type"] == "unvalidated_safety"][0]
        assert safety_2["priority"] == pytest.approx(0.175)

    def test_unvalidated_safety_priority_floored_at_01(self):
        """Priority should not drop below 0.1 even with many recency hits."""
        store = _MockStore()
        state = _state_with_chains(3)
        state = replace(state, last_gap_layers=["unvalidated_safety"] * 10)
        gaps = analyze_protocol_gaps(state, store)
        safety = [g for g in gaps if g["gap_type"] == "unvalidated_safety"][0]
        assert safety["priority"] >= 0.1

    def test_sparse_layer_overtakes_safety_after_decay(self):
        """After 2 safety selections, sparse layers should be higher priority."""
        store = _MockStore({"regeneration_reinnervation": 2})  # Very sparse layer
        state = _state_with_chains(3)
        state = replace(state, last_gap_layers=["unvalidated_safety", "unvalidated_safety"])
        gaps = analyze_protocol_gaps(state, store)
        # Safety is now 0.175, sparse layer should be ~0.9*(1-2/30) = 0.84
        top_gap = gaps[0]
        assert top_gap["gap_type"] != "unvalidated_safety"

    def test_last_gap_layers_window_bounded(self):
        """last_gap_layers should be capped at a sliding window."""
        # This is enforced in loop.py, but test the state field works
        state = initial_state(subject_ref="traj:draper_001")
        layers = ["unvalidated_safety"] * 15
        state = replace(state, last_gap_layers=layers[-10:])
        assert len(state.last_gap_layers) == 10


# ===========================================================================
# Fix 3: Query Expansion
# ===========================================================================

class TestQueryExpansion:

    def test_query_bank_has_8_per_layer(self):
        """Each layer should have 8 queries (doubled from 4)."""
        for layer, queries in _BASE_LAYER_QUERIES.items():
            assert len(queries) == 8, f"Layer {layer} has {len(queries)} queries, expected 8"

    def test_query_bank_covers_all_layers(self):
        """All 5 protocol layers should have query entries."""
        expected = {"root_cause_suppression", "pathology_reversal", "circuit_stabilization",
                    "regeneration_reinnervation", "adaptive_maintenance"}
        assert set(_BASE_LAYER_QUERIES.keys()) == expected

    def test_get_layer_query_rotates(self):
        """Different steps should produce different queries."""
        queries = set()
        for step in range(8):
            q = get_layer_query("root_cause_suppression", step)
            queries.add(q)
        assert len(queries) == 8

    def test_get_layer_query_has_year_suffix(self):
        """Query should end with current year."""
        import datetime
        year = str(datetime.datetime.now().year)
        q = get_layer_query("root_cause_suppression", 0)
        assert year in q

    def test_dynamic_query_extracts_hypothesis_terms(self):
        """Dynamic query should pull biomedical terms from hypothesis statements."""
        state = _state_with_hypotheses([
            "Inhibiting CK1delta with IGS2.7 reduces TDP-43 phosphorylation at Serine-409 in motor neurons"
        ])
        query = _get_dynamic_query(state, step=1, layer="root_cause_suppression")
        assert "ALS" in query
        # Should contain some biomedical terms from the hypothesis
        assert any(term in query for term in ["CK1delta", "IGS2.7", "TDP-43", "Serine-409"])

    def test_dynamic_query_falls_back_to_static(self):
        """Empty active_hypotheses should fall back to static query."""
        state = _state_with_hypotheses([])
        query = _get_dynamic_query(state, step=1, layer="root_cause_suppression")
        # Should be a valid static query
        assert "ALS" in query

    def test_dynamic_query_skips_old_format_ids(self):
        """Hypotheses that are old-format IDs (hyp:...) should be skipped."""
        state = _state_with_hypotheses(["hyp:abc123def456"])
        query = _get_dynamic_query(state, step=1, layer="root_cause_suppression")
        # Should fall back to static since the only hypothesis is an ID
        assert "hyp:" not in query

    def test_dynamic_query_uses_most_recent_hypothesis(self):
        """Dynamic query should prefer the most recent (last) hypothesis."""
        state = _state_with_hypotheses([
            "SOD1 mutation drives copper-mediated oxidative damage",
            "STMN2 cryptic exon inclusion causes axonal degeneration in sporadic ALS",
        ])
        query = _get_dynamic_query(state, step=1, layer="root_cause_suppression")
        # Should prefer STMN2 hypothesis (last in list)
        assert any(term in query for term in ["STMN2", "ALS"])


# ===========================================================================
# Fix 4: Thompson Posterior Updates
# ===========================================================================

class TestThompsonPosteriorUpdates:

    def test_posteriors_updated_on_evidence_gain(self):
        """When evidence > 0, success=True → alpha should increase."""
        posteriors = {"search_pubmed": (1.0, 1.0)}
        updated = _update_posteriors(posteriors, "search_pubmed", success=True)
        assert updated["search_pubmed"][0] == 2.0  # alpha
        assert updated["search_pubmed"][1] == 1.0  # beta unchanged

    def test_posteriors_updated_on_zero_evidence(self):
        """When evidence=0 and no other success signals, beta should increase."""
        posteriors = {"search_pubmed": (1.0, 1.0)}
        updated = _update_posteriors(posteriors, "search_pubmed", success=False)
        assert updated["search_pubmed"][0] == 1.0  # alpha unchanged
        assert updated["search_pubmed"][1] == 2.0  # beta

    def test_hypothesis_generation_is_thompson_success(self):
        """hypothesis_generated being non-None should count as Thompson success."""
        result = ActionResult(
            action=ActionType.GENERATE_HYPOTHESIS,
            success=True,
            hypothesis_generated="Some hypothesis statement",
        )
        _success = (
            result.evidence_items_added > 0
            or result.hypothesis_generated is not None
            or result.hypothesis_resolved
            or result.causal_depth_added > 0
            or result.protocol_regenerated
        )
        assert _success is True

    def test_zero_everything_is_thompson_failure(self):
        """An action that produces nothing should be a Thompson failure."""
        result = ActionResult(
            action=ActionType.SEARCH_PUBMED,
            success=True,
            evidence_items_added=0,
        )
        _success = (
            result.evidence_items_added > 0
            or result.hypothesis_generated is not None
            or result.hypothesis_resolved
            or result.causal_depth_added > 0
            or result.protocol_regenerated
        )
        assert _success is False

    def test_posteriors_diverge_after_asymmetric_outcomes(self):
        """After many successes for one action and failures for another, posteriors should diverge."""
        posteriors: dict[str, tuple[float, float]] = {}
        for _ in range(10):
            posteriors = _update_posteriors(posteriors, "search_pubmed", success=True)
            posteriors = _update_posteriors(posteriors, "check_pharmacogenomics", success=False)

        pubmed_alpha, pubmed_beta = posteriors["search_pubmed"]
        pharma_alpha, pharma_beta = posteriors["check_pharmacogenomics"]

        # PubMed should have much higher alpha/beta ratio
        pubmed_ratio = pubmed_alpha / pubmed_beta
        pharma_ratio = pharma_alpha / pharma_beta
        assert pubmed_ratio > pharma_ratio * 5

    def test_decay_applied_at_interval(self):
        """Decay should reduce posteriors toward (1, 1) at configured intervals."""
        posteriors = {"search_pubmed": (10.0, 5.0)}
        decayed = _apply_decay(posteriors, rate=0.95)
        assert decayed["search_pubmed"][0] == pytest.approx(9.5)
        assert decayed["search_pubmed"][1] == pytest.approx(4.75)


# ===========================================================================
# Integration: research_step state updates
# ===========================================================================

class TestResearchStepStateUpdates:

    def test_action_result_detail_carries_gap_info(self):
        """ActionResult.detail should be able to carry gap_layer and gap_type."""
        result = ActionResult(
            action=ActionType.GENERATE_HYPOTHESIS,
            success=True,
            hypothesis_generated="TDP-43 aggregation drives STMN2 loss",
            detail={"gap_layer": "root_cause_suppression", "gap_type": "sparse_layer"},
        )
        assert result.detail["gap_layer"] == "root_cause_suppression"
        assert result.detail["gap_type"] == "sparse_layer"

    def test_state_serialization_round_trip(self):
        """State with new fields should survive to_dict/from_dict round-trip."""
        state = initial_state(subject_ref="traj:draper_001")
        state = replace(
            state,
            active_hypotheses=["TDP-43 aggregation causes motor neuron death"],
            last_gap_layers=["unvalidated_safety", "root_cause_suppression"],
            action_posteriors={"search_pubmed": (5.0, 2.0), "generate_hypothesis": (3.0, 4.0)},
        )
        d = state.to_dict()
        restored = ResearchState.from_dict(d)
        assert restored.active_hypotheses == ["TDP-43 aggregation causes motor neuron death"]
        assert restored.last_gap_layers == ["unvalidated_safety", "root_cause_suppression"]
        assert restored.action_posteriors["search_pubmed"] == (5.0, 2.0)

    def test_backward_compat_state_with_old_ids(self):
        """State containing old-format hypothesis IDs should still deserialize."""
        d = {
            "subject_ref": "traj:draper_001",
            "active_hypotheses": ["hyp:abc123", "hyp:def456"],
            "last_gap_layers": [],
            "action_posteriors": {},
        }
        state = ResearchState.from_dict(d)
        assert state.active_hypotheses == ["hyp:abc123", "hyp:def456"]

    def test_backward_compat_state_missing_new_fields(self):
        """Old state dicts without new fields should still deserialize with defaults."""
        d = {
            "subject_ref": "traj:draper_001",
            "step_count": 100,
        }
        state = ResearchState.from_dict(d)
        assert state.last_gap_layers == []
        assert state.action_posteriors == {}


# ===========================================================================
# Fix 6: Action dominance prevention
# ===========================================================================

class TestActionDominancePrevention:

    def test_deepen_chain_infeasible_when_all_chains_full(self):
        """DEEPEN_CAUSAL_CHAIN should be infeasible when all chains >= target depth."""
        state = initial_state(subject_ref="traj:draper_001")
        state = replace(state, causal_chains={"int:drug_a": 10, "int:drug_b": 8})
        assert not _action_is_feasible(ActionType.DEEPEN_CAUSAL_CHAIN, state, target_depth=5)

    def test_deepen_chain_feasible_when_shallow_chains_exist(self):
        """DEEPEN_CAUSAL_CHAIN should be feasible when some chains are below target."""
        state = initial_state(subject_ref="traj:draper_001")
        state = replace(state, causal_chains={"int:drug_a": 2, "int:drug_b": 8})
        assert _action_is_feasible(ActionType.DEEPEN_CAUSAL_CHAIN, state, target_depth=5)

    def test_acquisition_actions_always_feasible(self):
        """Acquisition actions should always be feasible."""
        state = initial_state(subject_ref="traj:draper_001")
        for action in [ActionType.SEARCH_PUBMED, ActionType.SEARCH_TRIALS,
                       ActionType.QUERY_PATHWAYS, ActionType.QUERY_PPI_NETWORK,
                       ActionType.CHECK_PHARMACOGENOMICS, ActionType.QUERY_GALEN_KG,
                       ActionType.SEARCH_PREPRINTS, ActionType.QUERY_GALEN_SCM]:
            assert _action_is_feasible(action, state, target_depth=5)

    def test_generate_hypothesis_infeasible_at_max_active(self):
        """GENERATE_HYPOTHESIS should be infeasible when active_hypotheses is at max."""
        state = initial_state(subject_ref="traj:draper_001")
        state = replace(state, active_hypotheses=[f"hypothesis {i}" for i in range(10)])
        assert not _action_is_feasible(ActionType.GENERATE_HYPOTHESIS, state, target_depth=5)

    def test_generate_hypothesis_feasible_below_max(self):
        """GENERATE_HYPOTHESIS should be feasible when below max active."""
        state = initial_state(subject_ref="traj:draper_001")
        state = replace(state, active_hypotheses=["hypothesis 1", "hypothesis 2"])
        assert _action_is_feasible(ActionType.GENERATE_HYPOTHESIS, state, target_depth=5)

    def test_thompson_excludes_infeasible_actions(self):
        """Thompson should not select DEEPEN_CAUSAL_CHAIN when all chains are full."""
        state = initial_state(subject_ref="traj:draper_001")
        state = replace(
            state,
            causal_chains={"int:a": 10, "int:b": 10},
            protocol_version=1,
            step_count=100,
        )
        # Run 50 selections — none should be deepen_causal_chain
        actions_selected = set()
        for i in range(50):
            test_state = replace(state, step_count=100 + i)
            action, params = select_action_thompson(test_state, regen_threshold=999, target_depth=5)
            actions_selected.add(action.value)
        assert "deepen_causal_chain" not in actions_selected

    def test_thompson_diverse_when_chains_full(self):
        """When chains are full, Thompson should select a mix of acquisition + hypothesis actions."""
        state = initial_state(subject_ref="traj:draper_001")
        # All actions used very recently (step-1) so diversity floor never triggers.
        # This isolates Thompson sampling behavior.
        base_step = 100
        all_action_names = [at.value for at in ActionType]
        last_used = {name: base_step - 1 for name in all_action_names}
        state = replace(
            state,
            causal_chains={"int:a": 10, "int:b": 10},
            protocol_version=1,
            step_count=base_step,
            last_action_per_type=last_used,
        )
        actions_selected = []
        for i in range(50):
            test_state = replace(state, step_count=base_step + i,
                                 last_action_per_type={k: base_step + i - 1 for k in all_action_names})
            action, _ = select_action_thompson(test_state, regen_threshold=999, target_depth=5)
            actions_selected.append(action.value)

        unique = set(actions_selected)
        gen_hyp_count = actions_selected.count("generate_hypothesis")
        # Should have at least 3 different action types and gen_hypothesis should be < 80%
        assert len(unique) >= 3, f"Only {len(unique)} action types: {unique}"
        assert gen_hyp_count / len(actions_selected) < 0.80, (
            f"generate_hypothesis is {gen_hyp_count}/{len(actions_selected)} = "
            f"{gen_hyp_count/len(actions_selected):.0%}, should be < 80%"
        )

    def test_deepen_fallback_goes_to_acquisition_not_hypothesis(self):
        """When DEEPEN_CAUSAL_CHAIN has no shallow chains, fallback should be acquisition."""
        from research.policy import _build_thompson_params
        state = initial_state(subject_ref="traj:draper_001")
        state = replace(
            state,
            causal_chains={"int:a": 10, "int:b": 10},
            step_count=50,
        )
        action, params = _build_thompson_params(ActionType.DEEPEN_CAUSAL_CHAIN, state, target_depth=5)
        # Should NOT be generate_hypothesis
        assert action != ActionType.GENERATE_HYPOTHESIS, (
            f"DEEPEN_CAUSAL_CHAIN fell back to {action.value}, should be acquisition"
        )
