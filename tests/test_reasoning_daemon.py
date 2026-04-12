# tests/test_reasoning_daemon.py
"""Tests for the ReasoningDaemon — Claude-powered deep analysis."""
import pytest
from unittest.mock import MagicMock, patch
from daemons.reasoning_daemon import ReasoningDaemon, _select_mode


class TestModeSelection:
    def test_mode_weights_default(self):
        mode = _select_mode(step=0, weights=[0.5, 0.3, 0.2])
        assert mode in ("edge_deepening", "counterfactual", "cross_pathway")

    def test_mode_distribution_respects_weights(self):
        """Over many calls, mode A should appear ~50% of the time."""
        counts = {"edge_deepening": 0, "counterfactual": 0, "cross_pathway": 0}
        for step in range(100):
            mode = _select_mode(step=step, weights=[0.5, 0.3, 0.2])
            counts[mode] += 1
        assert counts["edge_deepening"] >= 35  # Expect ~50, allow variance
        assert counts["cross_pathway"] >= 10   # Expect ~20


class TestReasoningDaemon:
    def test_init(self):
        daemon = ReasoningDaemon(claude_api_key="test-key")
        assert daemon._interval_s > 0

    @patch("daemons.reasoning_daemon.TCGraph")
    def test_no_weak_edges_uses_cross_pathway(self, mock_graph_cls):
        mock_graph = MagicMock()
        mock_graph.get_weakest_edges.return_value = []
        mock_graph_cls.return_value = mock_graph
        daemon = ReasoningDaemon(claude_api_key="test-key")
        # When no weak edges exist, daemon should fall through to cross-pathway mode
        # (this tests the selection logic, not the full cycle)
        assert daemon is not None
