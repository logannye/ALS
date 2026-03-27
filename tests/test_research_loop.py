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
