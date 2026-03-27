"""Tests for research intelligence — protocol gap analysis and targeted hypothesis generation."""
from __future__ import annotations

import pytest

from research.intelligence import (
    analyze_protocol_gaps,
    build_hypothesis_prompt,
    build_validation_query,
    plan_search_from_hypothesis,
)
from research.state import initial_state


class _MockStore:
    """Mock evidence store that returns configurable counts per layer."""

    def __init__(self, layer_counts=None):
        self._layer_counts = layer_counts or {}

    def query_by_protocol_layer(self, layer):
        count = self._layer_counts.get(layer, 5)
        return [{"id": f"evi:mock_{layer}_{i}", "body": {"protocol_layer": layer}} for i in range(count)]

    def query_by_intervention_ref(self, int_id):
        return []


class TestAnalyzeProtocolGaps:

    def test_identifies_sparse_layer(self):
        store = _MockStore({"root_cause_suppression": 5, "pathology_reversal": 25,
                           "circuit_stabilization": 20, "regeneration_reinnervation": 3,
                           "adaptive_maintenance": 15})
        state = initial_state(subject_ref="traj:draper_001")
        gaps = analyze_protocol_gaps(state, store)
        sparse_gaps = [g for g in gaps if g["gap_type"] == "sparse_layer"]
        assert len(sparse_gaps) >= 1
        assert sparse_gaps[0]["layer"] == "regeneration_reinnervation"

    def test_identifies_shallow_chains(self):
        store = _MockStore()
        state = initial_state(subject_ref="traj:draper_001")
        state.causal_chains = {"int:pridopidine": 1, "int:vtx002": 5}
        gaps = analyze_protocol_gaps(state, store)
        chain_gaps = [g for g in gaps if g["gap_type"] == "shallow_chain"]
        assert len(chain_gaps) >= 1
        assert chain_gaps[0]["intervention_id"] == "int:pridopidine"

    def test_identifies_missing_measurements(self):
        store = _MockStore()
        state = initial_state(subject_ref="traj:draper_001")
        gaps = analyze_protocol_gaps(state, store)
        missing = [g for g in gaps if g["gap_type"] == "missing_data"]
        assert len(missing) >= 1
        # Genetic testing should be highest priority missing measurement
        genetic_gaps = [g for g in missing if "genetic" in g["description"]]
        assert len(genetic_gaps) >= 1

    def test_gaps_sorted_by_priority(self):
        store = _MockStore()
        state = initial_state(subject_ref="traj:draper_001")
        state.causal_chains = {"int:a": 0}
        gaps = analyze_protocol_gaps(state, store)
        priorities = [g["priority"] for g in gaps]
        assert priorities == sorted(priorities, reverse=True)

    def test_gaps_have_search_queries(self):
        store = _MockStore({"regeneration_reinnervation": 2})
        state = initial_state(subject_ref="traj:draper_001")
        gaps = analyze_protocol_gaps(state, store)
        for gap in gaps:
            if gap["gap_type"] in ("sparse_layer", "shallow_chain"):
                assert "search_queries" in gap
                assert len(gap["search_queries"]) >= 1


class TestBuildHypothesisPrompt:

    def test_shallow_chain_prompt_includes_intervention(self):
        gap = {"gap_type": "shallow_chain", "description": "test", "intervention_id": "int:pridopidine"}
        state = initial_state(subject_ref="traj:draper_001")
        prompt = build_hypothesis_prompt(gap, state, [])
        assert "pridopidine" in prompt
        assert "Erik Draper" in prompt
        assert "motor neuron" in prompt.lower()
        assert "search_terms" in prompt

    def test_sparse_layer_prompt_mentions_layer(self):
        gap = {"gap_type": "sparse_layer", "description": "test", "layer": "regeneration_reinnervation"}
        state = initial_state(subject_ref="traj:draper_001")
        prompt = build_hypothesis_prompt(gap, state, [])
        assert "regeneration" in prompt.lower()
        assert "repurposed" in prompt.lower() or "overlooked" in prompt.lower()

    def test_missing_data_prompt_predicts_result(self):
        gap = {"gap_type": "missing_data", "description": "Missing measurement: genetic_testing"}
        state = initial_state(subject_ref="traj:draper_001")
        prompt = build_hypothesis_prompt(gap, state, [])
        assert "predicted_result" in prompt


class TestBuildValidationQuery:

    def test_extracts_biomedical_terms(self):
        query = build_validation_query("TDP-43 nuclear import may be restored by sigma-1R agonism in ALS motor neurons")
        assert "ALS" in query
        assert "TDP-43" in query or "sigma-1R" in query

    def test_filters_stop_words(self):
        query = build_validation_query("This hypothesis suggests that the specific mechanism would likely involve pathways")
        # Should not be full of stop words
        words = query.split()
        assert len(words) <= 10

    def test_handles_empty_statement(self):
        query = build_validation_query("")
        assert "ALS" in query


class TestPlanSearchFromHypothesis:

    def test_creates_pubmed_searches(self):
        hyp_result = {
            "search_terms": ["ALS TDP-43 intrabody", "sigma-1R ER calcium"],
            "target_genes": ["TARDBP"],
        }
        actions = plan_search_from_hypothesis(hyp_result)
        pubmed_actions = [a for a in actions if a["action"] == "search_pubmed"]
        assert len(pubmed_actions) >= 1

    def test_creates_ppi_queries_for_genes(self):
        hyp_result = {
            "search_terms": ["test"],
            "target_genes": ["TARDBP", "SIGMAR1"],
        }
        actions = plan_search_from_hypothesis(hyp_result)
        ppi_actions = [a for a in actions if a["action"] == "query_ppi_network"]
        assert len(ppi_actions) >= 1

    def test_handles_empty_hypothesis(self):
        actions = plan_search_from_hypothesis({})
        assert actions == []
