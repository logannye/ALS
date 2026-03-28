"""Tests for Phase 1 improvements: missing executors, error logging, yield-aware skip.

TDD — these tests are written BEFORE the implementation.
"""
from __future__ import annotations

import pytest
from dataclasses import replace
from unittest.mock import patch, MagicMock

from research.actions import ActionType, ActionResult
from research.state import initial_state, ResearchState
from research.policy import _ACQUISITION_ROTATION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(**overrides) -> ResearchState:
    s = initial_state(subject_ref="traj:draper_001")
    for k, v in overrides.items():
        object.__setattr__(s, k, v) if not hasattr(s, k) else setattr(s, k, v)
    return s


class _MockStore:
    def query_by_protocol_layer(self, layer):
        return [{"id": f"evi:mock_{layer}"}]
    def query_by_intervention_ref(self, int_id):
        return []
    def query_by_mechanism_target(self, target):
        return []
    def query_all_interventions(self):
        return []
    def upsert_object(self, obj):
        pass
    def count_by_type(self, obj_type):
        return 100


class _MockLLM:
    def get_research_engine(self):
        return None
    def get_protocol_engine(self):
        return None
    def unload_protocol_model(self):
        pass


# ===========================================================================
# 1A. SEARCH_PREPRINTS executor exists in dispatch
# ===========================================================================

class TestSearchPreprintsExecutor:

    def test_dispatch_contains_search_preprints(self):
        """SEARCH_PREPRINTS must be present in the dispatch dict."""
        from research.loop import _execute_action
        # If the executor is missing, _execute_action returns success=False with "Unknown action"
        state = _state()
        result = _execute_action(
            ActionType.SEARCH_PREPRINTS,
            {"query": "ALS motor neuron", "protocol_layer": "root_cause_suppression"},
            state, _MockStore(), _MockLLM(),
        )
        # Should NOT be "Unknown action" error
        assert result.error is None or "Unknown action" not in result.error

    def test_search_preprints_returns_action_result(self):
        """Executor must return an ActionResult with correct action type."""
        from research.loop import _execute_action
        state = _state()
        result = _execute_action(
            ActionType.SEARCH_PREPRINTS,
            {"query": "ALS axonal regeneration", "protocol_layer": "regeneration_reinnervation"},
            state, _MockStore(), _MockLLM(),
        )
        assert isinstance(result, ActionResult)
        assert result.action == ActionType.SEARCH_PREPRINTS


# ===========================================================================
# 1B. QUERY_GALEN_SCM executor exists in dispatch
# ===========================================================================

class TestQueryGalenScmExecutor:

    def test_dispatch_contains_query_galen_scm(self):
        """QUERY_GALEN_SCM must be present in the dispatch dict."""
        from research.loop import _execute_action
        state = _state()
        result = _execute_action(
            ActionType.QUERY_GALEN_SCM,
            {"target_gene": "TARDBP", "protocol_layer": "root_cause_suppression"},
            state, _MockStore(), _MockLLM(),
        )
        assert result.error is None or "Unknown action" not in result.error

    def test_query_galen_scm_returns_action_result(self):
        """Executor must return an ActionResult with correct action type."""
        from research.loop import _execute_action
        state = _state()
        result = _execute_action(
            ActionType.QUERY_GALEN_SCM,
            {"target_gene": "TARDBP", "protocol_layer": "root_cause_suppression"},
            state, _MockStore(), _MockLLM(),
        )
        assert isinstance(result, ActionResult)
        assert result.action == ActionType.QUERY_GALEN_SCM


# ===========================================================================
# 1C. Error logging (silent catch must now print)
# ===========================================================================

class TestErrorLogging:

    def test_unknown_action_logs_warning(self, capsys):
        """An action not in dispatch dict must print a WARNING."""
        from research.loop import _execute_action
        state = _state()
        # UPDATE_TRAJECTORY is in ActionType but not in dispatch
        result = _execute_action(
            ActionType.UPDATE_TRAJECTORY,
            {},
            state, _MockStore(), _MockLLM(),
        )
        captured = capsys.readouterr()
        assert "WARNING" in captured.out or "No executor" in captured.out

    def test_exception_in_executor_logs_error(self, capsys):
        """An exception in an executor must print a failure message."""
        from research.loop import _execute_action
        state = _state()
        # SEARCH_PUBMED with a store that raises
        class _RaisingStore(_MockStore):
            def query_by_protocol_layer(self, layer):
                raise RuntimeError("test connection failure")
        # Force an executor that uses the store to fail
        result = _execute_action(
            ActionType.SEARCH_PUBMED,
            {"query": "ALS test"},
            state, _RaisingStore(), _MockLLM(),
        )
        captured = capsys.readouterr()
        assert "failed" in captured.out.lower() or result.error is not None


# ===========================================================================
# 1D. Yield-aware action skip
# ===========================================================================

class TestYieldAwareSkip:

    def test_low_yield_action_skipped(self):
        """Action with EMA < threshold after min_count uses should be skipped."""
        from research.policy import _select_acquisition_action
        # Rig the state so the first action in rotation has low yield
        first_action = _ACQUISITION_ROTATION[0]  # SEARCH_PUBMED at step 0
        state = _state(
            step_count=0,
            action_values={first_action.value: 0.01},
            action_counts={first_action.value: 10},
        )
        action, _ = _select_acquisition_action(state)
        # Should NOT select the low-yield action
        assert action != first_action

    def test_adequate_yield_action_not_skipped(self):
        """Action with EMA >= threshold should not be skipped."""
        from research.policy import _select_acquisition_action
        first_action = _ACQUISITION_ROTATION[0]
        state = _state(
            step_count=0,
            action_values={first_action.value: 5.0},
            action_counts={first_action.value: 10},
        )
        action, _ = _select_acquisition_action(state)
        assert action == first_action

    def test_new_action_not_skipped(self):
        """Action with < min_count uses should not be skipped (optimistic default)."""
        from research.policy import _select_acquisition_action
        first_action = _ACQUISITION_ROTATION[0]
        state = _state(
            step_count=0,
            action_values={first_action.value: 0.0},
            action_counts={first_action.value: 2},  # Below min_count threshold
        )
        action, _ = _select_acquisition_action(state)
        # Should still try it — not enough data to skip
        assert action == first_action

    def test_all_exhausted_falls_back_to_pubmed(self):
        """When all actions have low yield, must fall back to SEARCH_PUBMED."""
        from research.policy import _select_acquisition_action
        # Set ALL rotation actions to low yield
        low_values = {a.value: 0.01 for a in _ACQUISITION_ROTATION}
        high_counts = {a.value: 20 for a in _ACQUISITION_ROTATION}
        state = _state(
            step_count=0,
            action_values=low_values,
            action_counts=high_counts,
        )
        action, _ = _select_acquisition_action(state)
        # Must fall back to SEARCH_PUBMED (ultimate fallback)
        assert action == ActionType.SEARCH_PUBMED
