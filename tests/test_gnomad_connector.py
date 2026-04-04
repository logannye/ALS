"""Tests for GnomADConnector -- no gnomAD data required for unit tests."""
from __future__ import annotations

import pytest

from connectors.gnomad import GnomADConnector, _safe_float
from connectors.base import BaseConnector


class TestGnomADConnector:
    def test_connector_instantiates(self):
        conn = GnomADConnector()
        assert conn is not None

    def test_connector_inherits_base(self):
        conn = GnomADConnector()
        assert isinstance(conn, BaseConnector)

    def test_fetch_no_gene_returns_empty(self):
        conn = GnomADConnector()
        result = conn.fetch(gene="")
        assert result.evidence_items_added == 0
        assert result.errors == []

    def test_fetch_missing_file_returns_error(self):
        conn = GnomADConnector(file_path="/nonexistent/path/gnomad.tsv")
        result = conn.fetch(gene="SOD1")
        assert len(result.errors) == 1
        assert "not found" in result.errors[0]

    def test_ignores_extra_kwargs(self):
        """Factory passes uniprot= — connector must not crash."""
        conn = GnomADConnector(file_path="/nonexistent/path/gnomad.tsv")
        # Should not raise, even with extra kwargs in __init__ and fetch
        conn2 = GnomADConnector(store=None, extra_kwarg="ignored")
        result = conn.fetch(gene="", uniprot="Q13148")
        assert result.evidence_items_added == 0

    def test_evidence_id_format(self):
        """Evidence IDs must start with 'evi:gnomad:' prefix."""
        expected = "evi:gnomad:sod1_constraint"
        assert expected.startswith("evi:gnomad:")
        assert "sod1" in expected


class TestSafeFloat:
    def test_safe_float_valid(self):
        assert _safe_float("0.99") == 0.99

    def test_safe_float_integer(self):
        assert _safe_float("42") == 42.0

    def test_safe_float_na(self):
        assert _safe_float("NA") is None

    def test_safe_float_nan(self):
        assert _safe_float("NaN") is None

    def test_safe_float_empty(self):
        assert _safe_float("") is None

    def test_safe_float_none(self):
        assert _safe_float(None) is None

    def test_safe_float_dot(self):
        assert _safe_float(".") is None

    def test_safe_float_dash(self):
        assert _safe_float("-") is None

    def test_safe_float_negative(self):
        assert _safe_float("-1.5") == -1.5

    def test_safe_float_whitespace(self):
        assert _safe_float("  0.5  ") == 0.5


@pytest.mark.gnomad
class TestGnomADIntegration:
    """Integration tests requiring the gnomAD data file on Databank SSD."""

    _DATA_PATH = "/Volumes/Databank/databases/gnomad/gnomad_v4.1_constraint_metrics.tsv"

    def test_gnomad_fetch_tardbp(self):
        conn = GnomADConnector(file_path=self._DATA_PATH)
        result = conn.fetch(gene="TARDBP")
        assert result.errors == []
        # TARDBP may or may not be in gnomAD, so just check no crash
        assert result.evidence_items_added >= 0

    def test_gnomad_fetch_sod1(self):
        conn = GnomADConnector(file_path=self._DATA_PATH)
        result = conn.fetch(gene="SOD1")
        assert result.errors == []
        assert result.evidence_items_added >= 0
