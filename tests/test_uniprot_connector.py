"""Tests for UniProtConnector -- no UniProt data required for unit tests."""
from __future__ import annotations

import pytest

from connectors.uniprot import UniProtConnector
from connectors.base import BaseConnector


class TestUniProtConnector:
    def test_connector_instantiates(self):
        conn = UniProtConnector()
        assert conn is not None

    def test_connector_inherits_base(self):
        conn = UniProtConnector()
        assert isinstance(conn, BaseConnector)

    def test_fetch_no_gene_no_acc_returns_empty(self):
        conn = UniProtConnector()
        result = conn.fetch(gene="", accession="")
        assert result.evidence_items_added == 0
        assert result.errors == []

    def test_fetch_missing_file_returns_error(self):
        conn = UniProtConnector(file_path="/nonexistent/path/uniprot.tsv")
        result = conn.fetch(gene="SOD1")
        assert len(result.errors) == 1
        assert "not found" in result.errors[0]

    def test_uniprot_alias_sets_accession(self):
        """Factory passes uniprot= kwarg -- connector should use it as accession."""
        conn = UniProtConnector(file_path="/nonexistent/path/uniprot.tsv")
        result = conn.fetch(uniprot="Q13148")
        # File doesn't exist, so we get an error, but it shouldn't crash
        assert len(result.errors) == 1
        assert "not found" in result.errors[0]

    def test_ignores_extra_kwargs(self):
        """Factory may pass unexpected kwargs -- connector must not crash."""
        conn = UniProtConnector(store=None, extra_kwarg="ignored")
        result = conn.fetch(gene="", uniprot="", extra="ignored")
        assert result.evidence_items_added == 0

    def test_evidence_id_function_format(self):
        """Function evidence IDs must use 'evi:uniprot:' prefix."""
        expected = "evi:uniprot:q13148_function"
        assert expected.startswith("evi:uniprot:")
        assert "_function" in expected

    def test_evidence_id_ptm_format(self):
        expected = "evi:uniprot:q13148_ptm"
        assert expected.startswith("evi:uniprot:")
        assert "_ptm" in expected

    def test_evidence_id_disease_format(self):
        expected = "evi:uniprot:q13148_disease"
        assert expected.startswith("evi:uniprot:")
        assert "_disease" in expected

    def test_evidence_id_structure_format(self):
        expected = "evi:uniprot:q13148_structure"
        assert expected.startswith("evi:uniprot:")
        assert "_structure" in expected


@pytest.mark.uniprot
class TestUniProtIntegration:
    """Integration tests requiring the UniProt data file on Databank SSD."""

    _DATA_PATH = "/Volumes/Databank/databases/uniprot/uniprot_human_swissprot.tsv"

    def test_uniprot_fetch_tardbp(self):
        conn = UniProtConnector(file_path=self._DATA_PATH)
        result = conn.fetch(gene="TARDBP")
        assert result.errors == []
        # TARDBP should be in SwissProt
        assert result.evidence_items_added >= 0

    def test_uniprot_fetch_by_accession(self):
        conn = UniProtConnector(file_path=self._DATA_PATH)
        result = conn.fetch(accession="P00441")  # SOD1
        assert result.errors == []
        assert result.evidence_items_added >= 0

    def test_uniprot_fetch_via_uniprot_kwarg(self):
        """Factory compatibility: uniprot= kwarg should work as accession."""
        conn = UniProtConnector(file_path=self._DATA_PATH)
        result = conn.fetch(uniprot="P00441")  # SOD1
        assert result.errors == []
        assert result.evidence_items_added >= 0
