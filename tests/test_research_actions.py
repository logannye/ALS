"""Tests for research actions — types, results, and execution dispatch."""
from __future__ import annotations
import pytest
from research.actions import ActionType, ActionResult, build_action_params

class TestActionType:
    def test_all_16_actions_exist(self):
        assert len(ActionType) == 16

    def test_evidence_actions(self):
        assert ActionType.SEARCH_PUBMED.value == "search_pubmed"
        assert ActionType.SEARCH_TRIALS.value == "search_trials"
        assert ActionType.QUERY_CHEMBL.value == "query_chembl"
        assert ActionType.QUERY_OPENTARGETS.value == "query_opentargets"
        assert ActionType.CHECK_INTERACTIONS.value == "check_interactions"

    def test_reasoning_actions(self):
        assert ActionType.GENERATE_HYPOTHESIS.value == "generate_hypothesis"
        assert ActionType.DEEPEN_CAUSAL_CHAIN.value == "deepen_causal_chain"
        assert ActionType.VALIDATE_HYPOTHESIS.value == "validate_hypothesis"
        assert ActionType.SCORE_NEW_EVIDENCE.value == "score_new_evidence"
        assert ActionType.REGENERATE_PROTOCOL.value == "regenerate_protocol"

class TestActionResult:
    def test_default_construction(self):
        result = ActionResult(action=ActionType.SEARCH_PUBMED)
        assert result.evidence_items_added == 0
        assert result.success is True
        assert result.error is None

    def test_failed_result(self):
        result = ActionResult(action=ActionType.SEARCH_PUBMED, success=False, error="timeout")
        assert result.success is False

    def test_protocol_layer_field(self):
        """ActionResult should accept protocol_layer to track which layer evidence belongs to."""
        result = ActionResult(
            action=ActionType.SEARCH_PUBMED,
            protocol_layer="root_cause_suppression",
            evidence_items_added=5,
        )
        assert result.protocol_layer == "root_cause_suppression"

    def test_evidence_strength_field(self):
        """ActionResult should accept evidence_strength to classify evidence quality."""
        result = ActionResult(
            action=ActionType.SEARCH_PUBMED,
            evidence_strength="moderate",
            evidence_items_added=3,
        )
        assert result.evidence_strength == "moderate"

    def test_new_fields_default_none(self):
        result = ActionResult(action=ActionType.SEARCH_PUBMED)
        assert result.protocol_layer is None
        assert result.evidence_strength is None

class TestBuildActionParams:
    def test_pubmed_params(self):
        params = build_action_params(ActionType.SEARCH_PUBMED, query="TDP-43 sigma-1R neuroprotection")
        assert params["query"] == "TDP-43 sigma-1R neuroprotection"

    def test_chembl_params(self):
        params = build_action_params(ActionType.QUERY_CHEMBL, target_name="Sigma-1R")
        assert params["target_name"] == "Sigma-1R"

    def test_hypothesis_params(self):
        params = build_action_params(
            ActionType.GENERATE_HYPOTHESIS,
            topic="root_cause_suppression",
            uncertainty="Subtype posterior dominated by sporadic TDP-43 but genetics pending",
        )
        assert params["topic"] == "root_cause_suppression"
