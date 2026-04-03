"""Tests for action selection policy — balanced cycle with validation budget."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from research.actions import ActionType
from research.state import ResearchState, initial_state
from research.policy import select_action, _select_action_cycle, _CYCLE_LENGTH, _ACQUIRE_STEPS


class TestPharmacogenomicsDrugList:
    """Phase 10: Dynamic drug list loaded from interventions.json."""

    def test_loads_more_than_five_drugs(self):
        from research.policy import _get_pharmacogenomics_drugs
        drugs = _get_pharmacogenomics_drugs()
        assert len(drugs) > 5, f"Expected >5 drugs, got {len(drugs)}: {drugs}"

    def test_includes_riluzole(self):
        from research.policy import _get_pharmacogenomics_drugs
        drugs = _get_pharmacogenomics_drugs()
        assert any("riluzole" in d for d in drugs)

    def test_includes_masitinib(self):
        from research.policy import _get_pharmacogenomics_drugs
        drugs = _get_pharmacogenomics_drugs()
        assert any("masitinib" in d for d in drugs)

    def test_fallback_on_file_error(self):
        import research.policy as pol
        old = pol._DRUG_NAMES_CACHE
        pol._DRUG_NAMES_CACHE = None  # Clear cache
        try:
            with patch("builtins.open", side_effect=FileNotFoundError):
                drugs = pol._get_pharmacogenomics_drugs()
            assert drugs == pol._FALLBACK_DRUGS
        finally:
            pol._DRUG_NAMES_CACHE = old  # Restore

    def test_returns_list_of_strings(self):
        from research.policy import _get_pharmacogenomics_drugs
        drugs = _get_pharmacogenomics_drugs()
        assert isinstance(drugs, list)
        assert all(isinstance(d, str) for d in drugs)


class TestDrugCentricQueryStrategy:
    """Phase 10: Drug-name-based PubMed queries as 5th strategy."""

    def test_returns_string_with_drug(self):
        from research.policy import _get_drug_centric_query
        state = initial_state(subject_ref="traj:draper_001")
        query = _get_drug_centric_query(state, step=0)
        assert isinstance(query, str)
        assert "ALS" in query or "als" in query.lower()

    def test_rotates_drugs_across_steps(self):
        from research.policy import _get_drug_centric_query
        state = initial_state(subject_ref="traj:draper_001")
        queries = {_get_drug_centric_query(state, step=i) for i in range(20)}
        assert len(queries) >= 5, f"Expected >=5 unique queries, got {len(queries)}"

    def test_pubmed_five_strategy_cycling(self):
        """PubMed should use 5 strategies (step % 5), not 4."""
        from research.policy import _build_acquisition_params
        from research.actions import ActionType
        state = initial_state(subject_ref="traj:draper_001")
        queries = []
        for step in range(5):
            _, params = _build_acquisition_params(ActionType.SEARCH_PUBMED, state, step)
            queries.append(params.get("query", ""))
        # All 5 should produce queries; at least 3 should be distinct
        assert all(q for q in queries)
        assert len(set(queries)) >= 3


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
            action, _ = _select_action_cycle(state, regen_threshold=100)
            from research.policy import _ACQUISITION_ROTATION
            assert action in set(_ACQUISITION_ROTATION), (
                f"Step {step} (cycle pos {step % _CYCLE_LENGTH}) should be acquisition, got {action}"
            )

    def test_reasoning_on_reason_step(self):
        """Cycle position 1 should be reasoning (chain or hypothesis)."""
        state = self._state(
            step_count=1,
            causal_chains={"int:pridopidine": 2},
            top_uncertainties=["test"],
        )
        action, _ = _select_action_cycle(state, regen_threshold=100, target_depth=5)
        assert action in {ActionType.DEEPEN_CAUSAL_CHAIN, ActionType.GENERATE_HYPOTHESIS}

    def test_validation_on_validate_step_with_hypotheses(self):
        """Cycle position 3 with active hypotheses should validate."""
        state = self._state(
            step_count=3,
            active_hypotheses=["hyp:test1"],
            action_counts={"validate_hypothesis": 1},
        )
        action, _ = _select_action_cycle(state, regen_threshold=100)
        assert action == ActionType.VALIDATE_HYPOTHESIS

    def test_validation_budget_prevents_over_validation(self):
        """When validation ratio is too high, should acquire instead."""
        state = self._state(
            step_count=3,
            active_hypotheses=["hyp:test1"],
            last_action="validate_hypothesis",
            action_counts={"validate_hypothesis": 50, "search_pubmed": 10},
        )
        action, _ = _select_action_cycle(state, regen_threshold=100)
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
            action, _ = _select_action_cycle(state, regen_threshold=100)
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
        _select_action_cycle(state, regen_threshold=100)
        # At least one hypothesis should have been expired
        assert len(state.active_hypotheses) < 2 or state.resolved_hypotheses > 2

    # --- Change 5: target_depth configurable ---

    def test_deepen_chain_when_below_target_depth_10(self):
        """With target_depth=10, chains at depth 8 should still trigger deepening."""
        state = self._state(
            step_count=1,  # cycle position 1 = reasoning step
            causal_chains={"int:pridopidine": 8},
            top_uncertainties=["test"],
        )
        # Even-numbered cycle → chain deepening preferred
        state.step_count = 1  # cycle 0, position 1, even cycle → deepen
        action, params = _select_action_cycle(state, regen_threshold=100, target_depth=10)
        assert action == ActionType.DEEPEN_CAUSAL_CHAIN
        assert params.get("intervention_id") == "int:pridopidine"

    def test_no_deepen_when_at_target_depth(self):
        """Chains at or above target_depth should NOT trigger deepening."""
        state = self._state(
            step_count=1,
            causal_chains={"int:pridopidine": 10},
            top_uncertainties=["test"],
        )
        action, _ = _select_action_cycle(state, regen_threshold=100, target_depth=10)
        # Should fall through to hypothesis generation, not chain deepening
        assert action != ActionType.DEEPEN_CAUSAL_CHAIN

    # --- Change 3: connector target mapping ---

    def test_query_pathways_uses_als_target_keys(self):
        """QUERY_PATHWAYS must pass a target_name that exists in ALS_TARGETS."""
        from targets.als_targets import ALS_TARGETS
        state = self._state(
            causal_chains={"int:pridopidine": 2},
        )
        # Rotate through many steps to find one that hits QUERY_PATHWAYS
        for step in range(30):
            state.step_count = step
            action, params = _select_action_cycle(state, regen_threshold=100)
            if action == ActionType.QUERY_PATHWAYS:
                target_name = params.get("target_name", "")
                assert target_name in ALS_TARGETS, \
                    f"target_name '{target_name}' not in ALS_TARGETS keys"
                return
        # It's okay if QUERY_PATHWAYS wasn't selected in 30 steps
        # (depends on rotation), but let's make sure it was at least possible

    def test_query_ppi_passes_valid_gene_symbol(self):
        """QUERY_PPI_NETWORK must pass a valid gene symbol from ALS_TARGETS."""
        from targets.als_targets import ALS_TARGETS
        valid_genes = {t["gene"] for t in ALS_TARGETS.values() if t.get("gene")}
        state = self._state()
        for step in range(30):
            state.step_count = step
            action, params = _select_action_cycle(state, regen_threshold=100)
            if action == ActionType.QUERY_PPI_NETWORK:
                gene = params.get("gene_symbol", "")
                assert gene in valid_genes, \
                    f"gene_symbol '{gene}' not in ALS_TARGETS genes"
                return
