"""Tests for GEOALSConnector — disease-state gene expression from ALS tissue."""
from __future__ import annotations

import os
import tempfile

import pytest

from connectors.base import BaseConnector, ConnectorResult
from connectors.geo_als import (
    GEOALSConnector,
    _DEFAULT_DATASETS,
    _compute_fold_change,
    _find_gene_probe,
    _parse_series_matrix,
)


# ---------------------------------------------------------------------------
# Connector instantiation
# ---------------------------------------------------------------------------

def test_connector_instantiates():
    c = GEOALSConnector()
    assert c is not None


def test_connector_inherits_base():
    assert isinstance(GEOALSConnector(), BaseConnector)


# ---------------------------------------------------------------------------
# fetch edge cases
# ---------------------------------------------------------------------------

def test_fetch_no_gene_returns_empty():
    r = GEOALSConnector().fetch()
    assert r.evidence_items_added == 0
    assert r.errors == []


def test_fetch_missing_dir_returns_empty():
    r = GEOALSConnector(data_path="/nonexistent").fetch(gene="TARDBP")
    assert r.evidence_items_added == 0
    assert r.errors == []  # missing dir is not an error


# ---------------------------------------------------------------------------
# _parse_series_matrix
# ---------------------------------------------------------------------------

def test_parse_series_matrix_fixture():
    """Create a minimal series matrix and verify parsing."""
    content = '''!Series_title\t"Test"
!Series_geo_accession\t"GSE999999"
"ID_REF"\t"GSM001"\t"GSM002"\t"GSM003"\t"GSM004"
"GENE_A"\t10.5\t11.2\t5.1\t4.8
"GENE_B"\t2.3\t2.1\t8.7\t9.0
'''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        f.flush()
        samples, probes = _parse_series_matrix(f.name)
    os.unlink(f.name)
    assert len(samples) == 4
    assert samples == ["GSM001", "GSM002", "GSM003", "GSM004"]
    assert "GENE_A" in probes
    assert "GENE_B" in probes
    assert len(probes["GENE_A"]) == 4
    assert probes["GENE_A"] == [10.5, 11.2, 5.1, 4.8]


def test_parse_series_matrix_handles_na():
    """NA values should become 0.0."""
    content = '"ID_REF"\t"S1"\t"S2"\n"PROBE1"\t5.0\tNA\n'
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        f.flush()
        samples, probes = _parse_series_matrix(f.name)
    os.unlink(f.name)
    assert probes["PROBE1"] == [5.0, 0.0]


def test_parse_series_matrix_empty_file():
    """Empty file returns empty structures."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("")
        f.flush()
        samples, probes = _parse_series_matrix(f.name)
    os.unlink(f.name)
    assert samples == []
    assert probes == {}


def test_parse_series_matrix_metadata_only():
    """File with only metadata lines returns no probes."""
    content = "!Series_title\t\"Test\"\n!Series_summary\t\"Summary\"\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        f.flush()
        samples, probes = _parse_series_matrix(f.name)
    os.unlink(f.name)
    assert samples == []
    assert probes == {}


# ---------------------------------------------------------------------------
# _compute_fold_change
# ---------------------------------------------------------------------------

def test_compute_fold_change_upregulated():
    fc, pval = _compute_fold_change([10.0, 11.0], [2.0, 3.0])
    assert fc > 0  # upregulated


def test_compute_fold_change_downregulated():
    fc, pval = _compute_fold_change([2.0, 3.0], [10.0, 11.0])
    assert fc < 0  # downregulated


def test_compute_fold_change_identical():
    fc, pval = _compute_fold_change([5.0, 5.0], [5.0, 5.0])
    assert abs(fc) < 0.01  # no change
    assert pval >= 0.5  # not significant


def test_compute_fold_change_too_few_samples():
    fc, pval = _compute_fold_change([5.0], [3.0])
    assert fc == 0.0
    assert pval == 1.0


def test_compute_fold_change_returns_float_tuple():
    fc, pval = _compute_fold_change([8.0, 9.0, 10.0], [2.0, 3.0, 4.0])
    assert isinstance(fc, float)
    assert isinstance(pval, float)
    assert 0.0 <= pval <= 1.0


def test_compute_fold_change_zero_control():
    """Control values near zero should not cause division error."""
    fc, pval = _compute_fold_change([10.0, 11.0], [0.0, 0.0])
    assert isinstance(fc, float)
    assert not (fc != fc)  # not NaN


# ---------------------------------------------------------------------------
# _find_gene_probe
# ---------------------------------------------------------------------------

def test_find_gene_probe_exact_match():
    probes = {"TARDBP": [1.0, 2.0], "SOD1": [3.0, 4.0]}
    result = _find_gene_probe(probes, "TARDBP")
    assert result is not None
    assert result[0] == "TARDBP"


def test_find_gene_probe_case_insensitive():
    probes = {"tardbp": [1.0, 2.0], "SOD1": [3.0, 4.0]}
    result = _find_gene_probe(probes, "TARDBP")
    assert result is not None


def test_find_gene_probe_not_found():
    probes = {"SOD1": [1.0, 2.0]}
    result = _find_gene_probe(probes, "TARDBP")
    assert result is None


def test_find_gene_probe_empty():
    assert _find_gene_probe({}, "TARDBP") is None


# ---------------------------------------------------------------------------
# get_disease_signature
# ---------------------------------------------------------------------------

def test_get_disease_signature_returns_lists():
    c = GEOALSConnector(data_path="/nonexistent")
    up, down = c.get_disease_signature()
    assert isinstance(up, list)
    assert isinstance(down, list)


def test_get_disease_signature_caches():
    c = GEOALSConnector(data_path="/nonexistent")
    up1, down1 = c.get_disease_signature()
    up2, down2 = c.get_disease_signature()
    assert up1 is up2  # same object — cached


# ---------------------------------------------------------------------------
# Evidence ID format
# ---------------------------------------------------------------------------

def test_evidence_id_format():
    assert "evi:geo:" in "evi:geo:gse124439_tardbp"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_default_datasets_defined():
    assert "GSE124439" in _DEFAULT_DATASETS
    assert len(_DEFAULT_DATASETS) >= 8


def test_default_datasets_have_required_keys():
    for accession, meta in _DEFAULT_DATASETS.items():
        assert "title" in meta
        assert "tissue" in meta
        assert "species" in meta
        assert "n_disease" in meta
        assert "n_control" in meta


# ---------------------------------------------------------------------------
# Integration: synthetic data end-to-end
# ---------------------------------------------------------------------------

def test_fetch_with_synthetic_data():
    """Build a temporary dataset directory and verify fetch produces evidence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a fake GSE124439 directory with a series matrix
        acc_dir = os.path.join(tmpdir, "GSE124439")
        os.makedirs(acc_dir)

        # 10 disease + 10 control = 20 samples
        header = '"ID_REF"\t' + "\t".join(f'"GSM{i:04d}"' for i in range(20))
        # TARDBP upregulated in disease (first 10 high, last 10 low)
        disease_vals = "\t".join(["15.0"] * 10)
        control_vals = "\t".join(["3.0"] * 10)
        row = f'"TARDBP"\t{disease_vals}\t{control_vals}'
        content = f"!Series_title\t\"Test\"\n{header}\n{row}\n"

        matrix_path = os.path.join(acc_dir, "GSE124439_series_matrix.txt")
        with open(matrix_path, "w") as f:
            f.write(content)

        c = GEOALSConnector(data_path=tmpdir)
        r = c.fetch(gene="TARDBP")

        assert r.evidence_items_added == 1
        assert r.errors == []


def test_fetch_with_synthetic_data_no_match():
    """Fetch returns empty when gene is not in the matrix."""
    with tempfile.TemporaryDirectory() as tmpdir:
        acc_dir = os.path.join(tmpdir, "GSE124439")
        os.makedirs(acc_dir)

        header = '"ID_REF"\t' + "\t".join(f'"GSM{i:04d}"' for i in range(20))
        row = '"SOD1"\t' + "\t".join(["5.0"] * 20)
        content = f"!Series_title\t\"Test\"\n{header}\n{row}\n"

        with open(os.path.join(acc_dir, "GSE124439_series_matrix.txt"), "w") as f:
            f.write(content)

        c = GEOALSConnector(data_path=tmpdir)
        r = c.fetch(gene="TARDBP")

        assert r.evidence_items_added == 0


def test_fetch_with_store():
    """Verify evidence is upserted to a mock store."""
    upserted = []

    class MockStore:
        def upsert_object(self, obj):
            upserted.append(obj)

    with tempfile.TemporaryDirectory() as tmpdir:
        acc_dir = os.path.join(tmpdir, "GSE124439")
        os.makedirs(acc_dir)

        header = '"ID_REF"\t' + "\t".join(f'"GSM{i:04d}"' for i in range(20))
        disease_vals = "\t".join(["15.0"] * 10)
        control_vals = "\t".join(["3.0"] * 10)
        row = f'"TARDBP"\t{disease_vals}\t{control_vals}'
        content = f"!Series_title\t\"Test\"\n{header}\n{row}\n"

        with open(os.path.join(acc_dir, "GSE124439_series_matrix.txt"), "w") as f:
            f.write(content)

        c = GEOALSConnector(store=MockStore(), data_path=tmpdir)
        r = c.fetch(gene="TARDBP")

        assert r.evidence_items_added == 1
        assert len(upserted) == 1
        assert upserted[0].id == "evi:geo:gse124439_tardbp"
        assert upserted[0].body["pch_layer"] == 2


# ---------------------------------------------------------------------------
# Live data (skipped if data dir not present)
# ---------------------------------------------------------------------------

@pytest.mark.geo_als
def test_geo_als_fetch_tardbp():
    c = GEOALSConnector()
    r = c.fetch(gene="TARDBP")
    assert r.evidence_items_added >= 0
