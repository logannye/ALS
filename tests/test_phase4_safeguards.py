"""Tests for Phase 4: structural safeguards.

Covers: rotating search queries and Thompson sampling config.
"""
from __future__ import annotations

import pytest
from research.policy import get_layer_query, _BASE_LAYER_QUERIES


class TestRotatingQueries:

    def test_rotates_across_queries(self):
        """Different steps should produce different queries for the same layer."""
        q0 = get_layer_query("root_cause_suppression", step=0)
        q1 = get_layer_query("root_cause_suppression", step=1)
        q2 = get_layer_query("root_cause_suppression", step=2)
        # At least 2 of 3 should differ
        assert len({q0, q1, q2}) >= 2

    def test_year_suffix_present(self):
        """Query should contain the current year."""
        import datetime
        year = str(datetime.datetime.now().year)
        query = get_layer_query("pathology_reversal", step=0)
        assert year in query

    def test_all_layers_have_queries(self):
        """All 5 protocol layers should have base queries."""
        expected = {
            "root_cause_suppression",
            "pathology_reversal",
            "circuit_stabilization",
            "regeneration_reinnervation",
            "adaptive_maintenance",
        }
        assert set(_BASE_LAYER_QUERIES.keys()) == expected

    def test_each_layer_has_multiple_queries(self):
        """Each layer should have at least 3 query variants."""
        for layer, queries in _BASE_LAYER_QUERIES.items():
            assert len(queries) >= 3, f"{layer} has only {len(queries)} queries"

    def test_unknown_layer_has_fallback(self):
        """Unknown layer should produce a reasonable fallback query."""
        query = get_layer_query("unknown_layer", step=0)
        assert "ALS" in query
        assert "unknown" in query.lower()


class TestThompsonEnabled:

    def test_thompson_disabled_in_config(self):
        """Thompson sampling is disabled — depth-biased cycle policy is active."""
        import json
        with open("data/erik_config.json") as f:
            cfg = json.load(f)
        assert cfg.get("thompson_policy_enabled") is False
