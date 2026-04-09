"""Tests for the confidence updater — upgrades relationship confidence
based on evidence count and strength distribution."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from knowledge_quality.confidence_updater import (
    compute_confidence,
    update_relationship_confidences,
)


class TestComputeConfidence:
    def test_single_unknown_evidence_stays_low(self):
        """count=1, {'unknown': 1} -> 0.25-0.35"""
        result = compute_confidence(1, {"unknown": 1})
        assert 0.25 <= result <= 0.35

    def test_three_moderate_evidence_raises_confidence(self):
        """count=3, {'moderate': 3} -> >= 0.6"""
        result = compute_confidence(3, {"moderate": 3})
        assert result >= 0.6

    def test_five_strong_evidence_near_max(self):
        """count=5, {'strong': 5} -> >= 0.8"""
        result = compute_confidence(5, {"strong": 5})
        assert result >= 0.8

    def test_mixed_evidence_intermediate(self):
        """count=4, {'strong': 1, 'unknown': 3} -> 0.4-0.7"""
        result = compute_confidence(4, {"strong": 1, "unknown": 3})
        assert 0.4 <= result <= 0.7

    def test_zero_evidence_returns_minimum(self):
        """count=0, {} -> 0.3"""
        result = compute_confidence(0, {})
        assert result == 0.3

    def test_returns_stats(self):
        """update_relationship_confidences(dry_run=True) returns dict with right keys."""
        with patch("knowledge_quality.confidence_updater.get_connection") as mock_conn:
            ctx = MagicMock()
            cur = MagicMock()
            # Simulate empty scan (no relationships below 0.9)
            cur.fetchall.return_value = []
            ctx.__enter__ = MagicMock(return_value=ctx)
            ctx.__exit__ = MagicMock(return_value=False)
            ctx.cursor.return_value.__enter__ = MagicMock(return_value=cur)
            ctx.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.return_value = ctx

            stats = update_relationship_confidences(dry_run=True)
            assert "relationships_scanned" in stats
            assert "relationships_updated" in stats
            assert isinstance(stats["relationships_scanned"], int)
            assert isinstance(stats["relationships_updated"], int)
