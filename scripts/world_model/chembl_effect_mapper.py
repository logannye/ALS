"""ChEMBL → EffectDistribution mapper for the QuantitativeEffectEnricher.

Pure-function + cached module — no daemon state, no SCMWriter dependency.
Given a (compound_name, gene_symbol) pair, resolve to ChEMBL IDs and
return an EffectDistribution in ``ic50_log_nm`` units.

Unit conversion reminder:
    pChEMBL = -log10(activity_in_M)
    log_nm  = log10(activity_in_nM) = log10(activity_in_M * 1e9)
            = log10(activity_in_M) + 9
            = 9 - pChEMBL

So an IC50 of 100 nM → pChEMBL ≈ 7 → log_nm = 2, which is what the
simulator's ``ec50_log_nm`` / ``ic50_log_nm`` gates expect.

Design notes:
  * sqlite3 is the allowed exception — ChEMBL is a read-only external DB.
  * The target-side resolver reads UniProt SwissProt TSV once at import
    and caches gene symbol → accession. Worth ~5MB RSS; the daemon only
    runs periodically so this is fine.
  * Compound-side resolution is exact-first (`LOWER(pref_name) = %s`),
    synonym-fallback second. We deliberately avoid LIKE '%name%' fuzzy
    matching — it produces dangerous false positives (e.g. 'riluzole'
    matching 'TRORILUZOLE', which has different pharmacology).
"""
from __future__ import annotations

import logging
import math
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from world_model.scm_writer import EffectDistribution

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Defaults (overridable via ConfigLoader in the daemon).
# -----------------------------------------------------------------------------

DEFAULT_CHEMBL_DB = "/Volumes/Databank/databases/chembl_36.db"
DEFAULT_UNIPROT_TSV = "/Volumes/Databank/databases/uniprot/uniprot_human_swissprot.tsv"

# Activity standard types that we accept. IC50 is the canonical pharmacology
# measurement for antagonists; Potency / AC50 are commonly used in HTS
# screens and carry the same log_nm-convertible quantitative info.
_ACCEPTED_ACTIVITY_TYPES: tuple[str, ...] = (
    'IC50', 'Ki', 'Kd', 'EC50', 'Potency', 'AC50',
)

# Minimum number of activity data points to produce an EffectDistribution.
# pChEMBL measurements are sparse per (compound, target) pair — typical
# ALS-relevant targets have 1-6 values in ChEMBL 36. We accept N=1 with
# a fallback std (below) rather than skip useful data entirely.
_MIN_ACTIVITIES = 1

# Default std (in log_nm units) used when only one measurement exists.
# 0.5 log_nm ≈ a factor-of-3 uncertainty around the point estimate —
# deliberately wide so the simulator doesn't treat a single measurement
# as high-confidence.
_FALLBACK_STD_LOG_NM = 0.5


# -----------------------------------------------------------------------------
# UniProt gene-symbol → accession index.
# -----------------------------------------------------------------------------

_gene_to_uniprot_cache: Optional[dict[str, str]] = None
_gene_cache_lock = threading.Lock()


def _load_gene_to_uniprot(tsv_path: str) -> dict[str, str]:
    """Parse the SwissProt TSV and return a gene-symbol → primary accession dict.

    "Gene Names" column is space-delimited; the first token is canonical.
    We also index every alternative gene name to its primary accession so
    legacy symbols resolve.
    """
    out: dict[str, str] = {}
    path = Path(tsv_path)
    if not path.exists():
        logger.warning("uniprot tsv not found at %s — gene resolver returns empty", tsv_path)
        return out
    with path.open('r', encoding='utf-8') as fh:
        header = fh.readline().rstrip('\n').split('\t')
        try:
            entry_idx = header.index('Entry')
            gene_idx = header.index('Gene Names')
        except ValueError:
            logger.error("uniprot tsv header missing Entry/Gene Names columns")
            return out
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) <= max(entry_idx, gene_idx):
                continue
            accession = parts[entry_idx].strip()
            gene_names = parts[gene_idx].strip()
            if not accession or not gene_names:
                continue
            for symbol in gene_names.split():
                key = symbol.strip().upper()
                if not key:
                    continue
                # First occurrence wins — SwissProt order is generally by
                # protein significance, so this is a reasonable disambiguation.
                out.setdefault(key, accession)
    return out


def resolve_gene_to_uniprot(
    gene_symbol: str,
    uniprot_tsv: str = DEFAULT_UNIPROT_TSV,
) -> Optional[str]:
    """Return the UniProt accession for a HUGO gene symbol, or None."""
    global _gene_to_uniprot_cache
    if _gene_to_uniprot_cache is None:
        with _gene_cache_lock:
            if _gene_to_uniprot_cache is None:
                _gene_to_uniprot_cache = _load_gene_to_uniprot(uniprot_tsv)
    return _gene_to_uniprot_cache.get(gene_symbol.strip().upper())


# -----------------------------------------------------------------------------
# ChEMBL lookup.
# -----------------------------------------------------------------------------


@dataclass
class _ActivityRow:
    pchembl: float
    activity_type: str


def _connect_chembl(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path, check_same_thread=False)


def resolve_compound_to_chembl_id(
    compound_name: str,
    chembl_db: str = DEFAULT_CHEMBL_DB,
) -> Optional[str]:
    """Exact-match compound name → ChEMBL ID.

    Exact on LOWER(pref_name); if that fails, checks molecule_synonyms.
    Never uses wildcard LIKE — see module docstring rationale.
    """
    name = compound_name.strip().lower()
    if not name:
        return None
    conn = _connect_chembl(chembl_db)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT chembl_id FROM molecule_dictionary WHERE LOWER(pref_name) = ? LIMIT 1",
            (name,),
        )
        row = cur.fetchone()
        if row:
            return row[0]
        # Synonym fallback — exact match on the synonym text.
        cur.execute("""
            SELECT md.chembl_id
              FROM molecule_dictionary md
              JOIN molecule_synonyms ms ON md.molregno = ms.molregno
             WHERE LOWER(ms.synonyms) = ?
             LIMIT 1
        """, (name,))
        row = cur.fetchone()
        return row[0] if row else None
    except sqlite3.Error as e:
        logger.warning("ChEMBL lookup failed for compound=%s: %s", compound_name, e)
        return None
    finally:
        conn.close()


def fetch_activities(
    compound_chembl_id: str,
    target_uniprot: str,
    chembl_db: str = DEFAULT_CHEMBL_DB,
    max_rows: int = 200,
) -> list[_ActivityRow]:
    """Return pChEMBL values for a compound-target pair.

    Only activities with standard_relation='=' (exact measurements, not
    bounds) and pchembl_value IS NOT NULL. Deliberately narrow — we'd
    rather return zero rows than spurious effect data.
    """
    conn = _connect_chembl(chembl_db)
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT act.pchembl_value, act.standard_type
              FROM activities act
              JOIN assays ass ON act.assay_id = ass.assay_id
              JOIN target_dictionary td ON ass.tid = td.tid
              JOIN target_components tc ON td.tid = tc.tid
              JOIN component_sequences cs ON tc.component_id = cs.component_id
              JOIN molecule_dictionary md ON act.molregno = md.molregno
             WHERE md.chembl_id = ?
               AND cs.accession = ?
               AND act.pchembl_value IS NOT NULL
               AND act.standard_relation = '='
               AND act.standard_type IN ({})
             ORDER BY act.standard_type ASC
             LIMIT ?
        """.format(','.join('?' for _ in _ACCEPTED_ACTIVITY_TYPES)),
            (compound_chembl_id, target_uniprot, *_ACCEPTED_ACTIVITY_TYPES, max_rows),
        )
        return [
            _ActivityRow(pchembl=float(r[0]), activity_type=r[1])
            for r in cur.fetchall() if r[0] is not None
        ]
    except sqlite3.Error as e:
        logger.warning("ChEMBL activities failed for %s/%s: %s",
                       compound_chembl_id, target_uniprot, e)
        return []
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# Public entry point.
# -----------------------------------------------------------------------------


@dataclass
class MappingResult:
    """Success case: quantitative effect + provenance. None cases carry a reason."""
    effect: EffectDistribution
    compound_chembl_id: str
    target_uniprot: str
    n_activities: int
    dominant_activity_type: str


def map_compound_target_to_effect(
    compound_name: str,
    gene_symbol: str,
    chembl_db: str = DEFAULT_CHEMBL_DB,
    uniprot_tsv: str = DEFAULT_UNIPROT_TSV,
    min_activities: int = _MIN_ACTIVITIES,
) -> Optional[MappingResult]:
    """Resolve names → ChEMBL/UniProt → EffectDistribution.

    Returns None when any resolution step fails or there are fewer than
    ``min_activities`` data points. Never raises on missing data —
    returns None so the enricher daemon can skip cleanly.
    """
    target_uniprot = resolve_gene_to_uniprot(gene_symbol, uniprot_tsv)
    if not target_uniprot:
        return None
    compound_id = resolve_compound_to_chembl_id(compound_name, chembl_db)
    if not compound_id:
        return None
    activities = fetch_activities(compound_id, target_uniprot, chembl_db)
    if len(activities) < min_activities:
        return None

    # Convert pChEMBL → log_nm and summarise.
    log_nms = [9.0 - a.pchembl for a in activities]
    mean = sum(log_nms) / len(log_nms)
    if len(log_nms) >= 2:
        variance = sum((x - mean) ** 2 for x in log_nms) / len(log_nms)
        std = math.sqrt(variance)
        # Identical replicates produce std=0, which makes the simulator
        # treat the effect as deterministic. Floor to the N=1 fallback so
        # the downstream Gaussian sample has real dispersion.
        if std < 1e-6:
            std = _FALLBACK_STD_LOG_NM
    else:
        std = _FALLBACK_STD_LOG_NM
    sorted_vals = sorted(log_nms)
    k_lo = max(0, int(0.1 * (len(sorted_vals) - 1)))
    k_hi = max(0, int(0.9 * (len(sorted_vals) - 1)))

    # Activity type most frequently represented.
    type_counts: dict[str, int] = {}
    for a in activities:
        type_counts[a.activity_type] = type_counts.get(a.activity_type, 0) + 1
    dominant = max(type_counts.items(), key=lambda kv: kv[1])[0]

    scale = 'ic50_log_nm' if dominant == 'IC50' else 'ec50_log_nm'

    return MappingResult(
        effect=EffectDistribution(
            mean=mean,
            std=std,
            ci_lower=sorted_vals[k_lo],
            ci_upper=sorted_vals[k_hi],
            scale=scale,
        ),
        compound_chembl_id=compound_id,
        target_uniprot=target_uniprot,
        n_activities=len(activities),
        dominant_activity_type=dominant,
    )


def reset_caches() -> None:
    """Clear the gene→UniProt cache. Call in tests after changing the TSV."""
    global _gene_to_uniprot_cache
    with _gene_cache_lock:
        _gene_to_uniprot_cache = None
