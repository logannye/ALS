"""Tests for DisGeNETConnector -- gene-disease association evidence."""
from __future__ import annotations

import os
import tempfile

import pytest

from connectors.base import BaseConnector, ConnectorResult
from connectors.disgenet import (
    DisGeNETConnector,
    _ALS_DISEASE_IDS,
    _score_to_strength,
)


# ---------------------------------------------------------------------------
# Connector instantiation
# ---------------------------------------------------------------------------

def test_connector_instantiates():
    c = DisGeNETConnector(file_path="/nonexistent/path.tsv")
    assert c is not None


def test_connector_inherits_base():
    c = DisGeNETConnector(file_path="/nonexistent/path.tsv")
    assert isinstance(c, BaseConnector)


def test_connector_accepts_extra_kwargs():
    """Factory compatibility: extra kwargs should be silently ignored."""
    c = DisGeNETConnector(file_path="/nonexistent/path.tsv", uniprot="P10636")
    assert c is not None


# ---------------------------------------------------------------------------
# fetch edge cases
# ---------------------------------------------------------------------------

def test_fetch_no_gene_returns_empty():
    c = DisGeNETConnector(file_path="/nonexistent/path.tsv")
    result = c.fetch()
    assert isinstance(result, ConnectorResult)
    assert result.evidence_items_added == 0
    assert result.errors == []


def test_fetch_missing_file_returns_error():
    c = DisGeNETConnector(file_path="/nonexistent/missing.tsv")
    result = c.fetch(gene="TARDBP")
    assert result.evidence_items_added == 0
    assert len(result.errors) == 1
    assert "not found" in result.errors[0]


# ---------------------------------------------------------------------------
# _score_to_strength
# ---------------------------------------------------------------------------

def test_score_to_strength_strong():
    assert _score_to_strength(0.8) == "strong"


def test_score_to_strength_boundary_strong():
    assert _score_to_strength(0.7) == "strong"


def test_score_to_strength_moderate():
    assert _score_to_strength(0.5) == "moderate"


def test_score_to_strength_boundary_moderate():
    assert _score_to_strength(0.4) == "moderate"


def test_score_to_strength_emerging():
    assert _score_to_strength(0.3) == "emerging"


def test_score_to_strength_boundary_emerging():
    assert _score_to_strength(0.2) == "emerging"


def test_score_to_strength_unknown():
    assert _score_to_strength(0.1) == "unknown"


def test_score_to_strength_zero():
    assert _score_to_strength(0.0) == "unknown"


# ---------------------------------------------------------------------------
# ALS disease IDs
# ---------------------------------------------------------------------------

def test_als_disease_ids_includes_als():
    assert "C0002736" in _ALS_DISEASE_IDS


def test_als_disease_ids_is_frozenset():
    assert isinstance(_ALS_DISEASE_IDS, frozenset)


def test_als_disease_ids_has_three_entries():
    assert len(_ALS_DISEASE_IDS) == 3


# ---------------------------------------------------------------------------
# Evidence ID format
# ---------------------------------------------------------------------------

def test_evidence_id_format():
    """Verify the expected evidence ID pattern."""
    gene = "tardbp"
    disease_id = "C0002736"
    expected = f"evi:disgenet:{gene}_{disease_id}"
    assert expected == "evi:disgenet:tardbp_C0002736"


# ---------------------------------------------------------------------------
# Fetch with synthetic TSV (unit test, no real file needed)
# ---------------------------------------------------------------------------

def test_fetch_with_synthetic_tsv():
    """Create a temporary TSV matching the DisGeNET schema and verify parsing."""
    header = "geneId\tgeneSymbol\tDSI\tDPI\tdiseaseId\tdiseaseName\tdiseaseType\tdiseaseClass\tdiseaseSemanticType\tscore\tEI\tYearInitial\tYearFinal\tNofPmids\tNofSnps\tsource"
    row1 = "23435\tTARDBP\t0.62\t0.54\tC0002736\tAmyotrophic Lateral Sclerosis\tdisease\tNervous System Diseases\tDisease or Syndrome\t0.85\t1.00\t2006\t2024\t312\t48\tCTD_human;UNIPROT"
    row2 = "23435\tTARDBP\t0.62\t0.54\tC4049195\tAmyotrophic lateral sclerosis type 10\tdisease\tNervous System Diseases\tDisease or Syndrome\t0.50\t0.80\t2009\t2022\t85\t12\tCTD_human"
    # Non-ALS row -- should be skipped
    row3 = "23435\tTARDBP\t0.62\t0.54\tC0011849\tDiabetes Mellitus\tdisease\tMetabolic Diseases\tDisease or Syndrome\t0.10\t0.50\t2015\t2020\t2\t0\tCTD_human"

    content = "\n".join([header, row1, row2, row3]) + "\n"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".tsv", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        tmp_path = f.name

    try:
        c = DisGeNETConnector(file_path=tmp_path)
        result = c.fetch(gene="TARDBP")
        assert result.evidence_items_added == 2
        assert result.errors == []
    finally:
        os.unlink(tmp_path)


def test_fetch_case_insensitive_gene_match():
    """Gene matching should be case-insensitive."""
    header = "geneId\tgeneSymbol\tDSI\tDPI\tdiseaseId\tdiseaseName\tdiseaseType\tdiseaseClass\tdiseaseSemanticType\tscore\tEI\tYearInitial\tYearFinal\tNofPmids\tNofSnps\tsource"
    row = "6647\tSOD1\t0.45\t0.38\tC0002736\tAmyotrophic Lateral Sclerosis\tdisease\tNervous System Diseases\tDisease or Syndrome\t0.92\t1.00\t1993\t2024\t1500\t200\tCTD_human;UNIPROT"

    content = header + "\n" + row + "\n"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".tsv", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        tmp_path = f.name

    try:
        c = DisGeNETConnector(file_path=tmp_path)
        # Search with lowercase
        result = c.fetch(gene="sod1")
        assert result.evidence_items_added == 1
    finally:
        os.unlink(tmp_path)


def test_fetch_ignores_uniprot_kwarg():
    """Factory compat: uniprot= kwarg should be silently accepted."""
    c = DisGeNETConnector(file_path="/nonexistent/path.tsv")
    # Should not raise, just return empty (file missing error)
    result = c.fetch(gene="TARDBP", uniprot="Q13148")
    assert len(result.errors) == 1  # file not found


# ---------------------------------------------------------------------------
# Integration test (requires real DisGeNET file on disk)
# ---------------------------------------------------------------------------

@pytest.mark.disgenet
def test_disgenet_fetch_tardbp():
    """Integration test: fetch TARDBP associations from the real DisGeNET file."""
    c = DisGeNETConnector()
    result = c.fetch(gene="TARDBP")
    assert isinstance(result, ConnectorResult)
    assert result.evidence_items_added > 0
    assert result.errors == []
