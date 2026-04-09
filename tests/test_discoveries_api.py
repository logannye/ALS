"""Tests for the GET /api/discoveries endpoint."""
from __future__ import annotations

import sys
import os

# Ensure scripts/ is on the import path so `from db.pool import ...` resolves.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helper: call build_discoveries_response in dry_run mode (no DB needed)
# ---------------------------------------------------------------------------

def _get_dry_run_response(days: int = 14) -> dict:
    from api.routers.discoveries import build_discoveries_response
    return build_discoveries_response(days=days, dry_run=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDiscoveriesResponse:
    """Unit tests for the discoveries endpoint response shape."""

    def test_returns_dict_with_required_keys(self):
        result = _get_dry_run_response()
        assert isinstance(result, dict)
        assert "days" in result

    def test_highlights_is_list(self):
        result = _get_dry_run_response(days=1)
        day_entry = result["days"][0]
        assert isinstance(day_entry["highlights"], list)

    def test_each_highlight_has_text_and_category(self):
        result = _get_dry_run_response(days=1)
        for day_entry in result["days"]:
            for h in day_entry["highlights"]:
                assert "text" in h, "highlight missing 'text' key"
                assert "category" in h, "highlight missing 'category' key"

    def test_date_is_iso_string(self):
        result = _get_dry_run_response(days=1)
        day_entry = result["days"][0]
        # ISO date like "2026-04-09"
        date_str = day_entry["date"]
        assert isinstance(date_str, str)
        # Validate format: YYYY-MM-DD
        parts = date_str.split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 4  # year

    def test_days_list_returned(self):
        result = _get_dry_run_response(days=7)
        assert isinstance(result["days"], list)
        assert len(result["days"]) == 7

    def test_count_matches_requested_days(self):
        for n in (1, 3, 14):
            result = _get_dry_run_response(days=n)
            assert len(result["days"]) == n, f"Expected {n} days, got {len(result['days'])}"

    def test_reverse_chronological_order(self):
        result = _get_dry_run_response(days=5)
        dates = [d["date"] for d in result["days"]]
        assert dates == sorted(dates, reverse=True), (
            f"Days not in reverse chronological order: {dates}"
        )
