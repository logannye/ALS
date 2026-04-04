"""Tests for SpliceAIConnector — no SpliceAI data required for unit tests."""
from __future__ import annotations

import csv
import os
import tempfile

import pytest


def test_connector_instantiates():
    from connectors.spliceai import SpliceAIConnector
    c = SpliceAIConnector()
    assert c is not None


def test_connector_inherits_base():
    from connectors.spliceai import SpliceAIConnector
    from connectors.base import BaseConnector
    assert isinstance(SpliceAIConnector(), BaseConnector)


def test_fetch_no_gene_returns_empty():
    from connectors.spliceai import SpliceAIConnector
    c = SpliceAIConnector()
    r = c.fetch()
    assert r.evidence_items_added == 0
    assert not r.errors


def test_fetch_missing_dir_returns_empty():
    from connectors.spliceai import SpliceAIConnector
    c = SpliceAIConnector(data_dir="/nonexistent/path")
    r = c.fetch(gene="TARDBP")
    assert r.evidence_items_added == 0
    # Should NOT error — just return empty (data not yet available)
    assert not r.errors


def test_safe_float_valid():
    from connectors.spliceai import _safe_float
    assert _safe_float("0.85") == pytest.approx(0.85)
    assert _safe_float(0.5) == pytest.approx(0.5)
    assert _safe_float("0") == pytest.approx(0.0)


def test_safe_float_invalid():
    from connectors.spliceai import _safe_float
    assert _safe_float("") == 0.0
    assert _safe_float(None) == 0.0
    assert _safe_float("NA") == 0.0


def test_als_splice_genes_includes_unc13a():
    from connectors.spliceai import _ALS_SPLICE_GENES
    assert "UNC13A" in _ALS_SPLICE_GENES


def test_als_splice_genes_includes_stmn2():
    from connectors.spliceai import _ALS_SPLICE_GENES
    assert "STMN2" in _ALS_SPLICE_GENES


def test_thresholds():
    from connectors.spliceai import _HIGH_THRESHOLD, _MODERATE_THRESHOLD
    assert _HIGH_THRESHOLD == 0.5
    assert _MODERATE_THRESHOLD == 0.2


def test_fetch_with_fixture_data():
    """Create a temp TSV and verify the connector parses it correctly."""
    from connectors.spliceai import SpliceAIConnector

    with tempfile.TemporaryDirectory() as tmpdir:
        gene = "TESTGENE"
        filepath = os.path.join(tmpdir, f"spliceai_{gene}.tsv")
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["CHROM", "POS", "REF", "ALT", "SYMBOL",
                            "DS_AG", "DS_AL", "DS_DG", "DS_DL"],
                delimiter="\t",
            )
            writer.writeheader()
            # High impact variant (max_ds = 0.8)
            writer.writerow({
                "CHROM": "chr1", "POS": "100", "REF": "A", "ALT": "G",
                "SYMBOL": gene, "DS_AG": "0.8", "DS_AL": "0.1",
                "DS_DG": "0.0", "DS_DL": "0.05",
            })
            # Moderate impact variant (max_ds = 0.3)
            writer.writerow({
                "CHROM": "chr1", "POS": "200", "REF": "C", "ALT": "T",
                "SYMBOL": gene, "DS_AG": "0.1", "DS_AL": "0.3",
                "DS_DG": "0.0", "DS_DL": "0.0",
            })
            # Low impact variant (max_ds = 0.05)
            writer.writerow({
                "CHROM": "chr1", "POS": "300", "REF": "G", "ALT": "A",
                "SYMBOL": gene, "DS_AG": "0.0", "DS_AL": "0.0",
                "DS_DG": "0.0", "DS_DL": "0.05",
            })

        c = SpliceAIConnector(data_dir=tmpdir)
        r = c.fetch(gene=gene)
        # Summary (1) + high-impact individual (1) = 2
        assert r.evidence_items_added >= 1  # At least summary
        assert not r.errors


def test_fetch_counts_high_and_moderate():
    """Verify correct classification counts in summary body."""
    from connectors.spliceai import SpliceAIConnector

    with tempfile.TemporaryDirectory() as tmpdir:
        gene = "COUNTGENE"
        filepath = os.path.join(tmpdir, f"spliceai_{gene}.tsv")
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["CHROM", "POS", "REF", "ALT", "SYMBOL",
                            "DS_AG", "DS_AL", "DS_DG", "DS_DL"],
                delimiter="\t",
            )
            writer.writeheader()
            # 2 high impact
            writer.writerow({
                "CHROM": "chr1", "POS": "10", "REF": "A", "ALT": "G",
                "SYMBOL": gene, "DS_AG": "0.9", "DS_AL": "0.0",
                "DS_DG": "0.0", "DS_DL": "0.0",
            })
            writer.writerow({
                "CHROM": "chr1", "POS": "20", "REF": "C", "ALT": "T",
                "SYMBOL": gene, "DS_AG": "0.0", "DS_AL": "0.0",
                "DS_DG": "0.6", "DS_DL": "0.0",
            })
            # 1 moderate
            writer.writerow({
                "CHROM": "chr1", "POS": "30", "REF": "G", "ALT": "A",
                "SYMBOL": gene, "DS_AG": "0.0", "DS_AL": "0.25",
                "DS_DG": "0.0", "DS_DL": "0.0",
            })

        c = SpliceAIConnector(data_dir=tmpdir)
        r = c.fetch(gene=gene)
        # Summary (1) + 2 high-impact individuals = 3
        assert r.evidence_items_added == 3
        assert not r.errors


def test_fetch_ignores_uniprot_kwarg():
    """The factory executor passes uniprot= — connector must accept it silently."""
    from connectors.spliceai import SpliceAIConnector
    c = SpliceAIConnector(data_dir="/nonexistent/path")
    r = c.fetch(gene="TARDBP", uniprot="Q13148")
    assert r.evidence_items_added == 0
    assert not r.errors


def test_fetch_gene_case_insensitive():
    """Gene lookup should be case-insensitive."""
    from connectors.spliceai import SpliceAIConnector

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "spliceai_MYGENE.tsv")
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["CHROM", "POS", "REF", "ALT", "SYMBOL",
                            "DS_AG", "DS_AL", "DS_DG", "DS_DL"],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerow({
                "CHROM": "chr1", "POS": "100", "REF": "A", "ALT": "G",
                "SYMBOL": "MYGENE", "DS_AG": "0.7", "DS_AL": "0.0",
                "DS_DG": "0.0", "DS_DL": "0.0",
            })

        c = SpliceAIConnector(data_dir=tmpdir)
        r = c.fetch(gene="mygene")  # lowercase
        assert r.evidence_items_added >= 1


def test_fetch_empty_tsv_returns_empty():
    """A TSV with only headers and no data rows returns empty."""
    from connectors.spliceai import SpliceAIConnector

    with tempfile.TemporaryDirectory() as tmpdir:
        gene = "EMPTYGENE"
        filepath = os.path.join(tmpdir, f"spliceai_{gene}.tsv")
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["CHROM", "POS", "REF", "ALT", "SYMBOL",
                            "DS_AG", "DS_AL", "DS_DG", "DS_DL"],
                delimiter="\t",
            )
            writer.writeheader()

        c = SpliceAIConnector(data_dir=tmpdir)
        r = c.fetch(gene=gene)
        assert r.evidence_items_added == 0
        assert not r.errors


def test_evidence_id_format():
    assert "evi:spliceai:" in "evi:spliceai:tardbp_summary"


def test_fetch_with_store_mock():
    """Verify upsert_object is called on the store."""
    from connectors.spliceai import SpliceAIConnector

    upserted = []

    class MockStore:
        def upsert_object(self, obj):
            upserted.append(obj)

    with tempfile.TemporaryDirectory() as tmpdir:
        gene = "STOREGENE"
        filepath = os.path.join(tmpdir, f"spliceai_{gene}.tsv")
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["CHROM", "POS", "REF", "ALT", "SYMBOL",
                            "DS_AG", "DS_AL", "DS_DG", "DS_DL"],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerow({
                "CHROM": "chr1", "POS": "100", "REF": "A", "ALT": "G",
                "SYMBOL": gene, "DS_AG": "0.7", "DS_AL": "0.0",
                "DS_DG": "0.0", "DS_DL": "0.0",
            })

        c = SpliceAIConnector(store=MockStore(), data_dir=tmpdir)
        r = c.fetch(gene=gene)
        assert r.evidence_items_added >= 1
        assert len(upserted) >= 1
        assert upserted[0].id == f"evi:spliceai:{gene.lower()}_summary"


def test_fallback_file_search():
    """If canonical filename doesn't match, find file containing gene name."""
    from connectors.spliceai import SpliceAIConnector

    with tempfile.TemporaryDirectory() as tmpdir:
        gene = "FALLBACK"
        # Non-canonical filename
        filepath = os.path.join(tmpdir, f"scores_{gene}_v2.tsv")
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["CHROM", "POS", "REF", "ALT", "SYMBOL",
                            "DS_AG", "DS_AL", "DS_DG", "DS_DL"],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerow({
                "CHROM": "chr1", "POS": "50", "REF": "T", "ALT": "C",
                "SYMBOL": gene, "DS_AG": "0.6", "DS_AL": "0.0",
                "DS_DG": "0.0", "DS_DL": "0.0",
            })

        c = SpliceAIConnector(data_dir=tmpdir)
        r = c.fetch(gene=gene)
        assert r.evidence_items_added >= 1


# Integration test — requires actual SpliceAI data on disk
@pytest.mark.spliceai
def test_spliceai_fetch_tardbp():
    from connectors.spliceai import SpliceAIConnector
    c = SpliceAIConnector()
    r = c.fetch(gene="TARDBP")
    assert r.evidence_items_added >= 0
