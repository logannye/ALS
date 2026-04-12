"""Tests for Claude API client — rate limiting, spend tracking, fallback."""
import pytest
from unittest.mock import MagicMock, patch
from llm.claude_client import ClaudeClient, LLMSpendTracker


class TestLLMSpendTracker:
    def test_log_spend(self):
        tracker = LLMSpendTracker()
        tracker.log(model="claude-opus-4-6", phase="reasoning",
                    input_tokens=1000, output_tokens=500, cost_usd=0.05, prompt_cached=True)
        # Should not raise

    def test_monthly_spend(self):
        tracker = LLMSpendTracker()
        # Fresh tracker should report 0 or near-zero
        spend = tracker.monthly_spend_usd()
        assert isinstance(spend, float)
        assert spend >= 0.0


class TestClaudeClient:
    def test_init_with_config(self):
        client = ClaudeClient(
            api_key="test-key",
            reasoning_model="claude-opus-4-6",
            evaluation_model="claude-sonnet-4-6",
            max_opus_per_hour=30,
            max_sonnet_per_hour=60,
            monthly_budget_usd=100.0,
        )
        assert client._reasoning_model == "claude-opus-4-6"
        assert client._monthly_budget_usd == 100.0

    def test_budget_exceeded_blocks_calls(self):
        client = ClaudeClient(api_key="test-key", monthly_budget_usd=0.0)
        # With $0 budget, all calls should be blocked
        result = client.reason_about_edge(
            edge_context="test edge",
            supporting_evidence=["ev1"],
            contradicting_evidence=[],
        )
        assert result is None or result.get("budget_exceeded") is True

    @patch("llm.claude_client.ClaudeClient._call_api")
    def test_reason_about_edge_parses_response(self, mock_call):
        mock_call.return_value = {
            "confidence_assessment": 0.7,
            "mechanism": "Direct causal link supported by in vitro evidence",
            "open_questions": ["In vivo confirmation needed"],
            "confounders": [],
        }
        client = ClaudeClient(api_key="test-key", monthly_budget_usd=100.0)
        result = client.reason_about_edge(
            edge_context="TDP-43 aggregation causes STMN2 cryptic exon inclusion",
            supporting_evidence=["Paper A found direct correlation"],
            contradicting_evidence=[],
        )
        assert result["confidence_assessment"] == 0.7
        assert len(result["open_questions"]) == 1

    @patch("llm.claude_client.ClaudeClient._call_api")
    def test_cross_pathway_synthesis(self, mock_call):
        mock_call.return_value = {
            "proposed_edges": [
                {"source": "protein:tdp-43", "target": "process:microglial_activation",
                 "edge_type": "activates", "confidence": 0.3,
                 "rationale": "TDP-43 aggregates activate innate immune response"}
            ]
        }
        client = ClaudeClient(api_key="test-key", monthly_budget_usd=100.0)
        result = client.cross_pathway_synthesis(
            cluster_a="proteostasis", cluster_a_evidence=["ev1"],
            cluster_b="neuroinflammation", cluster_b_evidence=["ev2"],
        )
        assert len(result["proposed_edges"]) == 1
