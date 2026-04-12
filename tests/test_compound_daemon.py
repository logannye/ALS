"""Tests for the CompoundDaemon — drug candidate evaluation."""
import pytest
from unittest.mock import MagicMock, patch
from daemons.compound_daemon import CompoundDaemon


class TestCompoundDaemon:
    def test_init(self):
        daemon = CompoundDaemon(claude_api_key="test-key")
        assert daemon._interval_s >= 3600

    @patch("daemons.compound_daemon.TCGraph")
    def test_no_supported_hypotheses_is_noop(self, mock_graph_cls):
        mock_graph = MagicMock()
        mock_graph.get_hypotheses_by_status.return_value = []
        mock_graph_cls.return_value = mock_graph
        daemon = CompoundDaemon(claude_api_key="test-key")
        daemon._graph = mock_graph
        result = daemon.evaluate_once()
        assert result["action"] == "no_actionable_hypotheses"
