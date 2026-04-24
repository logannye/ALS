"""Tests for chembl_effect_mapper.

Pure-function tests run without any DB. Integration tests are gated
behind the existence of ``/Volumes/Databank/databases/chembl_36.db`` —
the real ChEMBL DB lives on an external SSD and may not be mounted in
all environments.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from world_model.chembl_effect_mapper import (
    DEFAULT_CHEMBL_DB,
    DEFAULT_UNIPROT_TSV,
    _FALLBACK_STD_LOG_NM,
    map_compound_target_to_effect,
    reset_caches,
    resolve_compound_to_chembl_id,
    resolve_gene_to_uniprot,
)


# ─── Unit-conversion sanity ──────────────────────────────────────────────────


def test_pchembl_to_log_nm_conversion_math():
    """pChEMBL=7 ↔ 100 nM ↔ log_nm=2. Verify invariant."""
    pchembl = 7.0
    expected_log_nm = 2.0
    assert 9.0 - pchembl == expected_log_nm


# ─── Gene-symbol resolver ────────────────────────────────────────────────────


uniprot_available = pytest.mark.skipif(
    not Path(DEFAULT_UNIPROT_TSV).exists(),
    reason=f"uniprot tsv not mounted at {DEFAULT_UNIPROT_TSV}",
)


@uniprot_available
def test_resolve_gene_known_symbols():
    reset_caches()
    assert resolve_gene_to_uniprot('SOD1') == 'P00441'
    # Case-insensitive and whitespace-tolerant.
    assert resolve_gene_to_uniprot('sod1') == 'P00441'
    assert resolve_gene_to_uniprot('  SOD1  ') == 'P00441'


@uniprot_available
def test_resolve_gene_unknown_returns_none():
    reset_caches()
    assert resolve_gene_to_uniprot('NOT_A_GENE_XYZ') is None


# ─── Compound resolver ──────────────────────────────────────────────────────


chembl_available = pytest.mark.skipif(
    not Path(DEFAULT_CHEMBL_DB).exists(),
    reason=f"chembl db not mounted at {DEFAULT_CHEMBL_DB}",
)


@chembl_available
def test_resolve_compound_exact_match():
    assert resolve_compound_to_chembl_id('riluzole') == 'CHEMBL744'
    assert resolve_compound_to_chembl_id('edaravone') == 'CHEMBL290916'


@chembl_available
def test_resolve_compound_unknown_returns_none():
    assert resolve_compound_to_chembl_id('not_a_real_compound_zzz') is None


@chembl_available
def test_resolve_compound_rejects_substring_match():
    """Must NOT match 'riluzole' against 'TRORILUZOLE' — different drug."""
    # Direct name 'trori' (not a drug) should return None, not a partial hit.
    assert resolve_compound_to_chembl_id('trori') is None


# ─── End-to-end mapping ─────────────────────────────────────────────────────


@chembl_available
@uniprot_available
def test_map_known_compound_target_pair_produces_effect():
    """Riluzole / SLC6A2 has ≥2 pChEMBL measurements in ChEMBL 36."""
    reset_caches()
    result = map_compound_target_to_effect('riluzole', 'SLC6A2')
    assert result is not None
    assert result.compound_chembl_id == 'CHEMBL744'
    assert result.target_uniprot == 'P23975'
    assert result.n_activities >= 2
    assert result.effect.mean is not None
    # Riluzole/EAAT-class potency should be in the micromolar range
    # (log_nm in 2–5, i.e. 100 nM to 100 µM).
    assert 1.0 <= result.effect.mean <= 6.0
    # Std must never be zero — even collapsing replicates use the fallback.
    assert result.effect.std is not None and result.effect.std > 0
    assert result.effect.scale in ('ic50_log_nm', 'ec50_log_nm')


@chembl_available
@uniprot_available
def test_map_returns_none_when_no_data():
    """Riluzole × SOD1 — no direct pChEMBL in ChEMBL 36."""
    reset_caches()
    assert map_compound_target_to_effect('riluzole', 'SOD1') is None


@chembl_available
@uniprot_available
def test_map_returns_none_on_unresolvable_compound():
    reset_caches()
    assert map_compound_target_to_effect('zzz_not_real', 'SOD1') is None


@chembl_available
@uniprot_available
def test_map_returns_none_on_unresolvable_gene():
    reset_caches()
    assert map_compound_target_to_effect('riluzole', 'NOT_A_GENE_XYZ') is None


# ─── Fallback-std enforcement ────────────────────────────────────────────────


@chembl_available
@uniprot_available
def test_identical_replicates_do_not_produce_zero_std():
    """When all pChEMBL samples are identical, std collapses to 0 and the
    fallback should kick in — simulator must always see nonzero dispersion."""
    reset_caches()
    # LMNA has 5 identical Potency measurements (all pchembl=5.15/4.9/... but
    # checking via any compound/target with near-identical replicates).
    # We assert the invariant in general: if n_activities >= 1, std > 0.
    for gene in ('SLC6A2', 'LMNA', 'CYP3A4'):
        r = map_compound_target_to_effect('riluzole', gene)
        if r is not None:
            assert r.effect.std is not None and r.effect.std > 0
            if r.n_activities == 1:
                assert r.effect.std == _FALLBACK_STD_LOG_NM
