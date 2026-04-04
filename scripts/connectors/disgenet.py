"""DisGeNETConnector -- gene-disease associations from curated sources.

Reads the DisGeNET curated_gene_disease_associations.tsv file and filters
for ALS-related disease IDs.  Each matching row becomes a BaseEnvelope
evidence item at PCH layer 1 (observational/associational).

Data file: /Volumes/Databank/databases/disgenet/curated_gene_disease_associations.tsv
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import BaseEnvelope

logger = logging.getLogger(__name__)

_DEFAULT_PATH = "/Volumes/Databank/databases/disgenet/curated_gene_disease_associations.tsv"

# UMLS CUI codes for ALS and closely related phenotypes
_ALS_DISEASE_IDS = frozenset({"C0002736", "C4049195", "C0393554"})


# ---------------------------------------------------------------------------
# Free function
# ---------------------------------------------------------------------------

def _score_to_strength(score: float) -> str:
    """Map a GDA score to a qualitative evidence strength label.

    >>> _score_to_strength(0.8)
    'strong'
    >>> _score_to_strength(0.5)
    'moderate'
    >>> _score_to_strength(0.3)
    'emerging'
    >>> _score_to_strength(0.1)
    'unknown'
    """
    if score >= 0.7:
        return "strong"
    if score >= 0.4:
        return "moderate"
    if score >= 0.2:
        return "emerging"
    return "unknown"


# ---------------------------------------------------------------------------
# DisGeNETConnector
# ---------------------------------------------------------------------------

class DisGeNETConnector(BaseConnector):
    """Connector for DisGeNET curated gene-disease associations (TSV file).

    Filters rows for ALS-related disease IDs and creates BaseEnvelope
    evidence items with GDA scores and supporting metadata.
    """

    def __init__(self, store: Any = None, file_path: str = _DEFAULT_PATH, **kwargs):
        self._store = store
        self._file_path = file_path

    # ------------------------------------------------------------------
    # BaseConnector contract
    # ------------------------------------------------------------------

    def fetch(self, *, gene: str = "", **kwargs) -> ConnectorResult:
        """Scan DisGeNET TSV for *gene* associations with ALS diseases.

        Parameters
        ----------
        gene:
            Gene symbol to search for (case-insensitive).
        **kwargs:
            Accepts and ignores ``uniprot=`` for factory compatibility.

        Returns
        -------
        ConnectorResult with one evidence item per matching row.
        """
        result = ConnectorResult()

        if not gene:
            return result

        if not Path(self._file_path).exists():
            result.errors.append(f"DisGeNET file not found: {self._file_path}")
            return result

        gene_upper = gene.upper()

        try:
            with open(self._file_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    gene_symbol = (row.get("geneSymbol") or "").strip()
                    disease_id = (row.get("diseaseId") or "").strip()

                    if gene_symbol.upper() != gene_upper:
                        continue
                    if disease_id not in _ALS_DISEASE_IDS:
                        continue

                    # Parse numeric fields safely
                    try:
                        score = float(row.get("score", 0))
                    except (ValueError, TypeError):
                        score = 0.0

                    try:
                        ei = float(row.get("EI", 0))
                    except (ValueError, TypeError):
                        ei = 0.0

                    try:
                        dsi = float(row.get("DSI", 0))
                    except (ValueError, TypeError):
                        dsi = 0.0

                    try:
                        dpi = float(row.get("DPI", 0))
                    except (ValueError, TypeError):
                        dpi = 0.0

                    try:
                        n_pmids = int(row.get("NofPmids", 0))
                    except (ValueError, TypeError):
                        n_pmids = 0

                    try:
                        n_snps = int(row.get("NofSnps", 0))
                    except (ValueError, TypeError):
                        n_snps = 0

                    disease_name = (row.get("diseaseName") or disease_id).strip()
                    source = (row.get("source") or "").strip()
                    gene_lower = gene.lower()

                    strength = _score_to_strength(score)

                    claim = (
                        f"DisGeNET: {gene} associated with {disease_name} "
                        f"(GDA score={score:.2f}, {n_pmids} PMIDs, EI={ei})"
                    )

                    evi = BaseEnvelope(
                        id=f"evi:disgenet:{gene_lower}_{disease_id}",
                        type="EvidenceItem",
                        status="active",
                        body={
                            "claim": claim,
                            "source": "disgenet",
                            "gene": gene,
                            "disease_id": disease_id,
                            "disease_name": disease_name,
                            "gda_score": score,
                            "evidence_index": ei,
                            "disease_specificity_index": dsi,
                            "disease_pleiotropy_index": dpi,
                            "n_pmids": n_pmids,
                            "n_snps": n_snps,
                            "source": source,
                            "evidence_strength": strength,
                            "pch_layer": 1,
                        },
                    )

                    if self._store:
                        self._store.upsert_object(evi)
                    result.evidence_items_added += 1

        except Exception as e:
            result.errors.append(f"DisGeNET scan failed for {gene}: {e}")

        return result
