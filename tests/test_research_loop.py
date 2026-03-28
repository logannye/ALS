"""Tests for the main research loop orchestrator."""
from __future__ import annotations
import pytest
from research.loop import research_step, run_research_loop
from research.state import initial_state
from research.actions import ActionType

class TestResearchStep:
    class _MockEvidenceStore:
        def query_by_protocol_layer(self, layer):
            return [{"id": f"evi:mock_{layer}", "type": "EvidenceItem", "status": "active",
                     "body": {"protocol_layer": layer, "claim": "mock"}, "claim": "mock"}]
        def query_by_intervention_ref(self, int_id):
            return []
        def query_by_mechanism_target(self, target):
            return []
        def query_all_interventions(self):
            return []
        def upsert_object(self, obj):
            pass
        def count_by_type(self, obj_type):
            return 93

    class _MockLLMManager:
        def get_research_engine(self):
            return None
        def get_protocol_engine(self):
            return None
        def unload_protocol_model(self):
            pass

    def test_step_increments_count(self):
        state = initial_state(subject_ref="traj:draper_001")
        assert state.step_count == 0
        new_state = research_step(
            state=state, evidence_store=self._MockEvidenceStore(),
            llm_manager=self._MockLLMManager(), dry_run=True,
        )
        assert new_state.step_count == 1

    def test_step_records_last_action(self):
        state = initial_state(subject_ref="traj:draper_001")
        new_state = research_step(
            state=state, evidence_store=self._MockEvidenceStore(),
            llm_manager=self._MockLLMManager(), dry_run=True,
        )
        assert new_state.last_action != ""

class TestRunResearchLoop:
    class _MockStore:
        def query_by_protocol_layer(self, layer):
            return []
        def query_by_intervention_ref(self, int_id):
            return []
        def query_by_mechanism_target(self, target):
            return []
        def query_all_interventions(self):
            return []
        def upsert_object(self, obj):
            pass
        def count_by_type(self, obj_type):
            return 93

    class _MockLLM:
        def get_research_engine(self):
            return None
        def get_protocol_engine(self):
            return None
        def unload_protocol_model(self):
            pass

    def test_loop_runs_n_steps(self):
        final_state = run_research_loop(
            subject_ref="traj:draper_001", evidence_store=self._MockStore(),
            llm_manager=self._MockLLM(), max_steps=3, dry_run=True,
        )
        assert final_state.step_count == 3

    def test_loop_returns_state(self):
        final = run_research_loop(
            subject_ref="traj:draper_001", evidence_store=self._MockStore(),
            llm_manager=self._MockLLM(), max_steps=1, dry_run=True,
        )
        assert isinstance(final.step_count, int)


class TestEvidenceCounterUpdates:
    """Test that evidence_by_layer and evidence_by_strength are properly updated."""

    def test_evidence_by_layer_updates_from_action_result(self):
        """When research_step processes an ActionResult with protocol_layer set
        and evidence_items_added > 0, evidence_by_layer must increment."""
        from unittest.mock import patch, MagicMock
        from research.actions import ActionResult
        from dataclasses import replace

        state = initial_state(subject_ref="traj:draper_001")
        assert state.evidence_by_layer.get("root_cause_suppression") == 0

        # Mock _execute_action to return a result with protocol_layer
        mock_result = ActionResult(
            action=ActionType.SEARCH_PUBMED,
            evidence_items_added=5,
            protocol_layer="root_cause_suppression",
            evidence_strength="moderate",
        )

        with patch("research.loop._execute_action", return_value=mock_result):
            new_state = research_step(
                state=state,
                evidence_store=MagicMock(),
                llm_manager=MagicMock(),
            )

        assert new_state.evidence_by_layer["root_cause_suppression"] == 5

    def test_evidence_by_strength_updates(self):
        """evidence_by_strength must increment when evidence_strength is set."""
        from unittest.mock import patch, MagicMock
        from research.actions import ActionResult

        state = initial_state(subject_ref="traj:draper_001")
        assert state.evidence_by_strength.get("moderate") == 0

        mock_result = ActionResult(
            action=ActionType.SEARCH_PUBMED,
            evidence_items_added=3,
            protocol_layer="pathology_reversal",
            evidence_strength="moderate",
        )

        with patch("research.loop._execute_action", return_value=mock_result):
            new_state = research_step(
                state=state,
                evidence_store=MagicMock(),
                llm_manager=MagicMock(),
            )

        assert new_state.evidence_by_strength["moderate"] == 3

    def test_uncertainty_decreases_when_empty_layer_gets_evidence(self):
        """Reward uncertainty_reduction should be > 0 when previously empty layer gets evidence."""
        from unittest.mock import patch, MagicMock
        from research.actions import ActionResult
        from research.rewards import compute_reward

        state = initial_state(subject_ref="traj:draper_001")
        # All layers start at 0, so uncertainty_before = 5/5 = 1.0

        mock_result = ActionResult(
            action=ActionType.SEARCH_PUBMED,
            evidence_items_added=5,
            protocol_layer="root_cause_suppression",
        )

        # After adding evidence to root_cause_suppression, only 4/5 layers are empty
        # uncertainty_after = 4/5 = 0.8, uncertainty_before = 1.0
        # uncertainty_reduction weight is 4.0, so reward component = (1.0 - 0.8) * 4.0 = 0.8
        reward = compute_reward(
            evidence_items_added=5,
            uncertainty_before=1.0,
            uncertainty_after=0.8,
            protocol_score_delta=0.0,
            hypothesis_resolved=False,
            causal_depth_added=0,
            interaction_safe=False,
            eligibility_confirmed=False,
            protocol_stable=False,
        )
        assert reward.uncertainty_reduction > 0
