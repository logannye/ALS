"""Tests for Phase 9: Query Expansion Engine (2026-03-30).

When gene x database combinations are exhausted (consecutive zero-delta queries),
the system expands to second-order queries derived from the KG's entities and
relationships — interacting genes, shared pathways, and mechanism neighbors.
"""
from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch, MagicMock

import pytest

from research.state import ResearchState, initial_state


# ---------------------------------------------------------------------------
# Step 1: State fields for exhaustion tracking
# ---------------------------------------------------------------------------

class TestExhaustionStateFields:
    """ResearchState must track target exhaustion and expansion history."""

    def test_state_has_target_exhaustion(self):
        state = initial_state("traj:test")
        assert hasattr(state, "target_exhaustion")
        assert state.target_exhaustion == {}

    def test_state_has_expansion_query_history(self):
        state = initial_state("traj:test")
        assert hasattr(state, "expansion_query_history")
        assert state.expansion_query_history == []

    def test_state_has_expansion_gene_history(self):
        state = initial_state("traj:test")
        assert hasattr(state, "expansion_gene_history")
        assert state.expansion_gene_history == {}

    def test_target_exhaustion_serializes(self):
        state = initial_state("traj:test")
        state = replace(state, target_exhaustion={"SOD1:query_clinvar": 5, "FUS:query_gtex": 2})
        d = state.to_dict()
        restored = ResearchState.from_dict(d)
        assert restored.target_exhaustion == {"SOD1:query_clinvar": 5, "FUS:query_gtex": 2}

    def test_expansion_query_history_serializes(self):
        state = initial_state("traj:test")
        state = replace(state, expansion_query_history=["als sod1 mechanism", "tdp43 aggregation"])
        d = state.to_dict()
        restored = ResearchState.from_dict(d)
        assert restored.expansion_query_history == ["als sod1 mechanism", "tdp43 aggregation"]

    def test_expansion_gene_history_serializes(self):
        state = initial_state("traj:test")
        state = replace(state, expansion_gene_history={"query_clinvar": ["OPTN", "TBK1"]})
        d = state.to_dict()
        restored = ResearchState.from_dict(d)
        assert restored.expansion_gene_history == {"query_clinvar": ["OPTN", "TBK1"]}


# ---------------------------------------------------------------------------
# Step 2: Exhaustion detection
# ---------------------------------------------------------------------------

class TestExhaustionDetection:
    """should_expand must correctly detect exhausted targets."""

    def test_below_threshold_no_expansion(self):
        from research.query_expansion import should_expand
        state = initial_state("traj:test")
        state = replace(state, target_exhaustion={"SOD1:query_clinvar": 2})
        assert not should_expand("SOD1:query_clinvar", state, threshold=3)

    def test_at_threshold_triggers_expansion(self):
        from research.query_expansion import should_expand
        state = initial_state("traj:test")
        state = replace(state, target_exhaustion={"SOD1:query_clinvar": 3})
        assert should_expand("SOD1:query_clinvar", state, threshold=3)

    def test_above_threshold_triggers_expansion(self):
        from research.query_expansion import should_expand
        state = initial_state("traj:test")
        state = replace(state, target_exhaustion={"SOD1:query_clinvar": 10})
        assert should_expand("SOD1:query_clinvar", state, threshold=3)

    def test_missing_key_no_expansion(self):
        from research.query_expansion import should_expand
        state = initial_state("traj:test")
        assert not should_expand("SOD1:query_clinvar", state, threshold=3)


# ---------------------------------------------------------------------------
# Step 3: KG neighbor lookup
# ---------------------------------------------------------------------------

class TestKGNeighborLookup:
    """get_gene_neighbors must query the KG for related genes."""

    def test_returns_list(self):
        from research.query_expansion import get_gene_neighbors
        # Should return a list even if KG is empty or unavailable
        result = get_gene_neighbors("NONEXISTENT_GENE_XYZ", max_neighbors=5)
        assert isinstance(result, list)

    def test_neighbors_have_required_fields(self):
        from research.query_expansion import get_gene_neighbors
        neighbors = get_gene_neighbors("SOD1", max_neighbors=5)
        for n in neighbors:
            assert "gene" in n
            assert "relationship" in n
            assert "confidence" in n


# ---------------------------------------------------------------------------
# Step 3: Expanded gene selection
# ---------------------------------------------------------------------------

class TestExpandedGeneSelection:
    """get_expanded_gene must pick unexplored neighbors and fall back correctly."""

    def test_falls_back_when_no_neighbors(self):
        from research.query_expansion import get_expanded_gene
        state = initial_state("traj:test")
        # With no KG data, should fall back to original gene
        result = get_expanded_gene(
            "NONEXISTENT_XYZ", "query_clinvar", state,
            max_neighbors=5, min_confidence=0.4,
        )
        assert result == "NONEXISTENT_XYZ"

    def test_skips_already_expanded_genes(self):
        from research.query_expansion import get_expanded_gene
        state = initial_state("traj:test")
        state = replace(state, expansion_gene_history={
            "query_clinvar": ["OPTN", "TBK1", "NEK1"],
        })
        # Mock KG to return exactly these 3 neighbors — all exhausted
        with patch("research.query_expansion.get_gene_neighbors") as mock_neighbors:
            mock_neighbors.return_value = [
                {"gene": "OPTN", "relationship": "associated_with", "confidence": 0.9},
                {"gene": "TBK1", "relationship": "associated_with", "confidence": 0.8},
                {"gene": "NEK1", "relationship": "associated_with", "confidence": 0.7},
            ]
            result = get_expanded_gene("SOD1", "query_clinvar", state)
            # All neighbors exhausted, should fall back to original
            assert result == "SOD1"

    def test_picks_highest_confidence_unexplored(self):
        from research.query_expansion import get_expanded_gene
        state = initial_state("traj:test")
        state = replace(state, expansion_gene_history={
            "query_clinvar": ["OPTN"],  # OPTN already used
        })
        with patch("research.query_expansion.get_gene_neighbors") as mock_neighbors:
            mock_neighbors.return_value = [
                {"gene": "OPTN", "relationship": "associated_with", "confidence": 0.9},
                {"gene": "TBK1", "relationship": "targets", "confidence": 0.8},
                {"gene": "NEK1", "relationship": "associated_with", "confidence": 0.7},
            ]
            result = get_expanded_gene("SOD1", "query_clinvar", state)
            # TBK1 is highest-confidence unexplored
            assert result == "TBK1"


# ---------------------------------------------------------------------------
# Step 3: Expanded query generation (template-based, no LLM)
# ---------------------------------------------------------------------------

class TestExpandedQueryGeneration:
    """get_expanded_queries generates novel PubMed queries from KG context."""

    def test_generates_queries_from_neighbors(self):
        from research.query_expansion import get_expanded_queries
        state = initial_state("traj:test")
        neighbors = [
            {"gene": "OPTN", "relationship": "associated_with", "confidence": 0.9},
            {"gene": "TBK1", "relationship": "targets", "confidence": 0.8},
        ]
        queries = get_expanded_queries("SOD1", neighbors, state)
        assert len(queries) > 0
        assert all(isinstance(q, str) for q in queries)

    def test_queries_contain_neighbor_genes(self):
        from research.query_expansion import get_expanded_queries
        state = initial_state("traj:test")
        neighbors = [
            {"gene": "OPTN", "relationship": "associated_with", "confidence": 0.9},
        ]
        queries = get_expanded_queries("SOD1", neighbors, state)
        # At least one query should mention the neighbor gene
        assert any("OPTN" in q for q in queries)

    def test_filters_queries_in_history(self):
        from research.query_expansion import get_expanded_queries, _normalize_query
        state = initial_state("traj:test")
        neighbors = [
            {"gene": "OPTN", "relationship": "associated_with", "confidence": 0.9},
        ]
        # Pre-fill history with the expected expanded queries
        queries_before = get_expanded_queries("SOD1", neighbors, state)
        if queries_before:
            state = replace(
                state,
                expansion_query_history=[_normalize_query(q) for q in queries_before],
            )
            queries_after = get_expanded_queries("SOD1", neighbors, state)
            # All queries should be filtered out since they're in history
            assert len(queries_after) < len(queries_before)

    def test_history_cap(self):
        from research.query_expansion import _cap_history
        history = [f"query_{i}" for i in range(600)]
        capped = _cap_history(history, max_size=500)
        assert len(capped) == 500
        # Should keep the most recent entries
        assert capped[-1] == "query_599"


# ---------------------------------------------------------------------------
# Step 3: Query normalization
# ---------------------------------------------------------------------------

class TestQueryNormalization:
    """_normalize_query must canonicalize queries for dedup."""

    def test_lowercase(self):
        from research.query_expansion import _normalize_query
        assert _normalize_query("ALS SOD1 Mutation") == _normalize_query("als sod1 mutation")

    def test_strips_year(self):
        from research.query_expansion import _normalize_query
        assert _normalize_query("ALS SOD1 2026") == _normalize_query("ALS SOD1 2025")

    def test_same_words_different_order(self):
        from research.query_expansion import _normalize_query
        # Sorted normalization means order doesn't matter
        assert _normalize_query("SOD1 ALS mutation") == _normalize_query("ALS mutation SOD1")


# ---------------------------------------------------------------------------
# Step 4: Policy integration
# ---------------------------------------------------------------------------

class TestPolicyIntegration:
    """Expansion must be transparent to action selection."""

    def test_expansion_disabled_returns_original(self):
        """When disabled, _build_acquisition_params returns standard rotation."""
        from research.policy import _build_acquisition_params
        from research.actions import ActionType
        state = initial_state("traj:test")
        state = replace(state, target_exhaustion={"TARDBP:query_clinvar": 100})
        # Even with exhaustion=100, disabled expansion should give original
        with patch("research.query_expansion.should_expand", return_value=False):
            _, params = _build_acquisition_params(ActionType.QUERY_CLINVAR, state, 0)
            # Should get one of the 16 canonical genes
            from targets.als_targets import ALS_TARGETS
            canonical_genes = {t["gene"] for t in ALS_TARGETS.values() if t.get("gene")}
            assert params.get("gene") in canonical_genes
