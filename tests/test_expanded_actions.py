# tests/test_expanded_actions.py
"""Tests for expanded research actions (Phase 3B)."""
from __future__ import annotations
import pytest
from research.actions import ActionType, NETWORK_ACTIONS

class TestExpandedActionTypes:
    def test_total_15_actions(self):
        assert len(ActionType) == 15

    def test_new_evidence_actions(self):
        assert ActionType.QUERY_PATHWAYS.value == "query_pathways"
        assert ActionType.QUERY_PPI_NETWORK.value == "query_ppi_network"
        assert ActionType.MATCH_COHORT.value == "match_cohort"

    def test_new_reasoning_actions(self):
        assert ActionType.INTERPRET_VARIANT.value == "interpret_variant"
        assert ActionType.CHECK_PHARMACOGENOMICS.value == "check_pharmacogenomics"

    def test_network_actions_includes_new(self):
        assert ActionType.QUERY_PATHWAYS in NETWORK_ACTIONS
        assert ActionType.QUERY_PPI_NETWORK in NETWORK_ACTIONS
