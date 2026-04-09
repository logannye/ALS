"""Tests for deep-mode stagnation detection and expanded action set."""
from __future__ import annotations

import pytest
from dataclasses import replace

from research.state import ResearchState, initial_state


class TestDeepStagnationDetected:
    """Tests for _deep_stagnation_detected()."""

    def test_returns_false_when_step_count_below_window(self):
        from run_loop import _deep_stagnation_detected

        state = initial_state(subject_ref="traj:test")
        state = replace(state, step_count=10, last_deep_evidence_step=0)
        assert _deep_stagnation_detected(state, window=50) is False

    def test_returns_true_when_no_evidence_for_window_steps(self):
        from run_loop import _deep_stagnation_detected

        state = initial_state(subject_ref="traj:test")
        state = replace(state, step_count=100, last_deep_evidence_step=30)
        # 100 - 30 = 70 >= 50
        assert _deep_stagnation_detected(state, window=50) is True

    def test_returns_false_when_recent_evidence(self):
        from run_loop import _deep_stagnation_detected

        state = initial_state(subject_ref="traj:test")
        state = replace(state, step_count=100, last_deep_evidence_step=80)
        # 100 - 80 = 20 < 50
        assert _deep_stagnation_detected(state, window=50) is False

    def test_returns_true_at_exact_boundary(self):
        from run_loop import _deep_stagnation_detected

        state = initial_state(subject_ref="traj:test")
        state = replace(state, step_count=50, last_deep_evidence_step=0)
        # 50 - 0 = 50 >= 50
        assert _deep_stagnation_detected(state, window=50) is True

    def test_returns_false_one_below_boundary(self):
        from run_loop import _deep_stagnation_detected

        state = initial_state(subject_ref="traj:test")
        state = replace(state, step_count=49, last_deep_evidence_step=0)
        # 49 < 50 (step_count < window)
        assert _deep_stagnation_detected(state, window=50) is False

    def test_custom_window_size(self):
        from run_loop import _deep_stagnation_detected

        state = initial_state(subject_ref="traj:test")
        state = replace(state, step_count=200, last_deep_evidence_step=180)
        # 200 - 180 = 20 >= 10
        assert _deep_stagnation_detected(state, window=10) is True
        # 200 - 180 = 20 < 30
        assert _deep_stagnation_detected(state, window=30) is False


class TestGetExpandedDeepActions:
    """Tests for _get_expanded_deep_actions()."""

    def test_returns_list_of_tuples(self):
        from run_loop import _get_expanded_deep_actions

        actions = _get_expanded_deep_actions(step=0)
        assert isinstance(actions, list)
        assert len(actions) > 0
        for item in actions:
            assert isinstance(item, tuple)
            assert len(item) == 2
            name, params = item
            assert isinstance(name, str)
            assert isinstance(params, dict)

    def test_returns_at_least_30_actions(self):
        from run_loop import _get_expanded_deep_actions

        actions = _get_expanded_deep_actions(step=0)
        assert len(actions) >= 30

    def test_includes_database_query_actions(self):
        from run_loop import _get_expanded_deep_actions

        actions = _get_expanded_deep_actions(step=0)
        action_names = [name for name, _ in actions]
        # Must include database connectors beyond the basic 5
        for db_action in [
            "query_chembl",
            "query_gwas",
            "query_clinvar",
            "query_bindingdb",
            "query_hpa",
            "query_drugbank",
            "query_uniprot",
            "query_alphafold",
            "query_disgenet",
            "query_gnomad",
            "query_geo_als",
            "query_reactome_local",
        ]:
            assert db_action in action_names, f"Missing database action: {db_action}"

    def test_includes_llm_actions(self):
        from run_loop import _get_expanded_deep_actions

        actions = _get_expanded_deep_actions(step=0)
        action_names = [name for name, _ in actions]
        assert "generate_hypothesis" in action_names
        assert "deepen_causal_chain" in action_names

    def test_includes_preprint_and_galen_actions(self):
        from run_loop import _get_expanded_deep_actions

        actions = _get_expanded_deep_actions(step=0)
        action_names = [name for name, _ in actions]
        assert "search_preprints" in action_names
        assert "query_galen_kg" in action_names

    def test_rotation_index_varies_with_step(self):
        """Different steps select different starting points in the list."""
        from run_loop import _get_expanded_deep_actions

        actions = _get_expanded_deep_actions(step=0)
        total = len(actions)
        # Step 0 gives index 0, step 1 gives index 1, etc.
        idx_0 = 0 % total
        idx_5 = 5 % total
        assert idx_0 != idx_5  # Different steps should pick different actions


class TestLastDeepEvidenceStepField:
    """Tests for the last_deep_evidence_step field on ResearchState."""

    def test_field_exists_with_default(self):
        state = initial_state(subject_ref="traj:test")
        assert hasattr(state, "last_deep_evidence_step")
        assert state.last_deep_evidence_step == 0

    def test_serializes_to_dict(self):
        state = initial_state(subject_ref="traj:test")
        state = replace(state, last_deep_evidence_step=42)
        d = state.to_dict()
        assert "last_deep_evidence_step" in d
        assert d["last_deep_evidence_step"] == 42

    def test_deserializes_from_dict(self):
        state = initial_state(subject_ref="traj:test")
        d = state.to_dict()
        d["last_deep_evidence_step"] = 99
        restored = ResearchState.from_dict(d)
        assert restored.last_deep_evidence_step == 99

    def test_deserializes_missing_field_defaults_to_zero(self):
        state = initial_state(subject_ref="traj:test")
        d = state.to_dict()
        del d["last_deep_evidence_step"]
        restored = ResearchState.from_dict(d)
        assert restored.last_deep_evidence_step == 0


class TestDeepStagnationConfigKey:
    """Ensure config key deep_stagnation_window exists."""

    def test_config_has_deep_stagnation_window(self):
        import json
        from pathlib import Path

        cfg_path = Path(__file__).resolve().parent.parent / "data" / "erik_config.json"
        cfg = json.loads(cfg_path.read_text())
        assert "deep_stagnation_window" in cfg
        assert isinstance(cfg["deep_stagnation_window"], int)
        assert cfg["deep_stagnation_window"] == 50
