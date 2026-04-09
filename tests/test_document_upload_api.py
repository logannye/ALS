"""Tests for the document upload API — extract_structured_data and regex parsers."""
from __future__ import annotations

import sys
import os

# Ensure scripts/ is on the import path so module imports resolve.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


# ---------------------------------------------------------------------------
# Tests for extract_structured_data (dry_run mode, no DB needed)
# ---------------------------------------------------------------------------

class TestExtractStructuredData:
    """Unit tests for the extract_structured_data response shape."""

    def test_returns_dict_with_required_keys(self):
        from api.routers.document_upload import extract_structured_data
        result = extract_structured_data("", dry_run=True)
        assert isinstance(result, dict)
        for key in ("document_id", "document_type", "source", "date", "extracted", "uncertain_fields"):
            assert key in result, f"Missing key: {key}"

    def test_extracted_is_list(self):
        from api.routers.document_upload import extract_structured_data
        result = extract_structured_data("", dry_run=True)
        assert isinstance(result["extracted"], list)

    def test_dry_run_returns_empty_extraction(self):
        from api.routers.document_upload import extract_structured_data
        result = extract_structured_data("", dry_run=True)
        assert result["extracted"] == []
        assert result["document_type"] == "unknown"


# ---------------------------------------------------------------------------
# Tests for _parse_lab_patterns
# ---------------------------------------------------------------------------

class TestParseLabPatterns:
    """Unit tests for regex-based lab value extraction."""

    def test_extracts_nfl_value(self):
        from api.routers.document_upload import _parse_lab_patterns
        results = _parse_lab_patterns("Neurofilament light chain (NfL): 5.82 pg/mL")
        assert len(results) >= 1
        nfl = results[0]
        assert nfl["type"] == "lab"
        assert nfl["value"] == 5.82
        assert "nfl" in nfl["name"].lower() or "neurofilament" in nfl["name"].lower()

    def test_extracts_creatinine(self):
        from api.routers.document_upload import _parse_lab_patterns
        results = _parse_lab_patterns("Creatinine: 0.9 mg/dL")
        assert len(results) >= 1
        cr = results[0]
        assert cr["type"] == "lab"
        assert cr["value"] == 0.9

    def test_empty_text_returns_empty(self):
        from api.routers.document_upload import _parse_lab_patterns
        results = _parse_lab_patterns("")
        assert results == []

    def test_no_match_returns_empty(self):
        from api.routers.document_upload import _parse_lab_patterns
        results = _parse_lab_patterns("general note about the patient visit")
        assert results == []
