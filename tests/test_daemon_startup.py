"""Tests for daemon startup wiring in run_loop."""
import pytest
from unittest.mock import patch, MagicMock


class TestDaemonStartup:
    @patch("daemons.integration_daemon.IntegrationDaemon")
    @patch("daemons.reasoning_daemon.ReasoningDaemon")
    @patch("daemons.compound_daemon.CompoundDaemon")
    def test_start_daemons_creates_threads(self, mock_compound, mock_reasoning, mock_integration):
        from run_loop import _start_cognitive_daemons, _stop_cognitive_daemons

        daemons = _start_cognitive_daemons(claude_api_key="test-key")
        assert "integration" in daemons
        assert "reasoning" in daemons
        assert "compound" in daemons
        _stop_cognitive_daemons(daemons)
