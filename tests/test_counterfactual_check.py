"""Tests for Stage 5 — Counterfactual verification of protocol layers."""
from __future__ import annotations

import pytest

from world_model.counterfactual_check import (
    CounterfactualResult,
    check_counterfactual,
    run_counterfactual_analysis,
)


# ---------------------------------------------------------------------------
# CounterfactualResult model
# ---------------------------------------------------------------------------

class TestCounterfactualResultModel:

    def test_basic_construction(self):
        cr = CounterfactualResult(
            layer="root_cause_suppression",
            removal_impact="critical",
            reasoning="Removing root-cause suppression eliminates the primary disease-modifying intervention.",
            is_load_bearing=True,
            weakest_evidence="evi:vtx002_fast_track",
            next_best_measurement="genetic_testing",
            cited_evidence=["evi:vtx002_fast_track"],
        )
        assert cr.is_load_bearing is True
        assert cr.removal_impact == "critical"
        assert cr.layer == "root_cause_suppression"

    def test_defaults(self):
        cr = CounterfactualResult(layer="pathology_reversal")
        assert cr.removal_impact == "uncertain"
        assert cr.is_load_bearing is False
        assert cr.reasoning == ""
        assert cr.cited_evidence == []


# ---------------------------------------------------------------------------
# check_counterfactual (with mock reasoning engine)
# ---------------------------------------------------------------------------

class TestCheckCounterfactual:

    class _MockEngine:
        def __init__(self, response):
            self._response = response

        def reason(self, **kwargs):
            return self._response

    def test_returns_result_on_success(self):
        mock_response = {
            "layer": "root_cause_suppression",
            "removal_impact": "critical",
            "reasoning": "Critical layer [evi:a].",
            "is_load_bearing": True,
            "weakest_evidence": "evi:a",
            "next_best_measurement": "genetic_testing",
            "cited_evidence": ["evi:a"],
        }
        engine = self._MockEngine(mock_response)
        result = check_counterfactual(
            protocol_json="{}",
            layer_name="root_cause_suppression",
            layer_interventions="int:vtx002",
            evidence_items=[{"id": "evi:a"}],
            reasoning_engine=engine,
        )
        assert result is not None
        assert result.is_load_bearing is True

    def test_returns_uncertain_on_none(self):
        engine = self._MockEngine(None)
        result = check_counterfactual(
            protocol_json="{}",
            layer_name="circuit_stabilization",
            layer_interventions="int:riluzole",
            evidence_items=[],
            reasoning_engine=engine,
        )
        assert result is not None
        assert result.removal_impact == "uncertain"
        assert "no response" in result.reasoning.lower() or "no response" in result.reasoning


# ---------------------------------------------------------------------------
# run_counterfactual_analysis (with mock protocol + engine)
# ---------------------------------------------------------------------------

class TestRunCounterfactualAnalysis:

    class _MockStore:
        def query_by_intervention_ref(self, int_id):
            return [{"id": f"evi:mock_{int_id}"}]

    class _MockEngine:
        def reason(self, **kwargs):
            return {
                "layer": "test",
                "removal_impact": "moderate",
                "reasoning": "Test [evi:mock].",
                "is_load_bearing": False,
                "weakest_evidence": "evi:mock",
                "next_best_measurement": "csf_biomarkers",
                "cited_evidence": ["evi:mock"],
            }

    def test_abstained_layers_get_low_impact(self):
        from ontology.enums import ProtocolLayer
        from ontology.protocol import CureProtocolCandidate, ProtocolLayerEntry

        protocol = CureProtocolCandidate(
            id="proto:test_v1",
            subject_ref="traj:test",
            objective="test",
            layers=[
                ProtocolLayerEntry(
                    layer=ProtocolLayer.root_cause_suppression,
                    intervention_refs=[],
                    notes="ABSTENTION: No eligible interventions",
                ),
                ProtocolLayerEntry(
                    layer=ProtocolLayer.circuit_stabilization,
                    intervention_refs=["int:riluzole"],
                    notes="Riluzole (score=0.70)",
                ),
            ],
        )
        results = run_counterfactual_analysis(
            protocol=protocol,
            evidence_store=self._MockStore(),
            reasoning_engine=self._MockEngine(),
        )
        assert len(results) == 2
        # Abstained layer should have low impact
        assert results[0].removal_impact == "low"
        assert results[0].is_load_bearing is False
