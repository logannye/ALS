"""GTExAPIConnector — queries the GTEx REST API for gene expression data.

Replaces the local GTEx v10 SQLite database for Railway deployment.
Uses the public GTEx Portal API v2 at https://gtexportal.org/api/v2.

API endpoints used:
  - GET /expression/medianGeneExpression?gencodeId={gene}&datasetId=gtex_v8
      Fetch median gene expression (TPM) across tissues for a gene.
      The API accepts both GENCODE ID (ENSG...) and gene symbol for gencodeId.

ALS-relevant tissues filtered:
  - Brain_Spinal_cord_cervical_c-1  (primary ALS site — upper motor neurons)
  - Brain_Frontal_Cortex_BA9
  - Brain_Cortex
  - Nerve_Tibial
  - Muscle_Skeletal
  - Whole_Blood
"""
from __future__ import annotations

import logging

import requests

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import Provenance, Uncertainty
from ontology.enums import EvidenceDirection, EvidenceStrength, SourceSystem
from ontology.evidence import EvidenceItem

logger = logging.getLogger(__name__)

# ALS-relevant tissues to keep from the full GTEx v8 tissue set
_ALS_RELEVANT_TISSUES = frozenset({
    "Brain_Spinal_cord_cervical_c-1",
    "Brain_Frontal_Cortex_BA9",
    "Brain_Cortex",
    "Nerve_Tibial",
    "Muscle_Skeletal",
    "Whole_Blood",
})

# ---------------------------------------------------------------------------
# Free functions: API payload parsers
# ---------------------------------------------------------------------------


def _parse_tissue_expression(
    gene: str,
    tissue_id: str,
    median_tpm: float,
) -> EvidenceItem:
    """Parse one GTEx tissue expression record into an EvidenceItem.

    Parameters
    ----------
    gene:
        Gene symbol (used in evidence ID and claim).
    tissue_id:
        GTEx tissue ID string, e.g. "Brain_Spinal_cord_cervical_c-1".
    median_tpm:
        Median TPM expression value from GTEx.

    Returns
    -------
    EvidenceItem with pch_layer 1 (associational — observational expression).
    """
    # Classify expression level
    if median_tpm >= 10.0:
        expression_label = "highly expressed"
        strength = EvidenceStrength.strong
        confidence_score = 0.90
    elif median_tpm >= 1.0:
        expression_label = "moderately expressed"
        strength = EvidenceStrength.moderate
        confidence_score = 0.75
    elif median_tpm > 0.0:
        expression_label = "lowly expressed"
        strength = EvidenceStrength.emerging
        confidence_score = 0.60
    else:
        expression_label = "not detected"
        strength = EvidenceStrength.emerging
        confidence_score = 0.50

    tissue_display = tissue_id.replace("_", " ")
    claim = (
        f"{gene} is {expression_label} in {tissue_display} "
        f"(GTEx v8 median TPM = {median_tpm:.2f})"
    )

    item_id = f"evi:gtex:{gene.lower()}_{tissue_id.lower()}"

    return EvidenceItem(
        id=item_id,
        claim=claim,
        direction=EvidenceDirection.supports,
        strength=strength,
        provenance=Provenance(
            source_system=SourceSystem.database,
            asserted_by="gtex_api_connector",
        ),
        uncertainty=Uncertainty(confidence=confidence_score),
        body={
            "pch_layer": 1,
            "gene": gene,
            "tissue_id": tissue_id,
            "median_tpm": median_tpm,
            "expression_label": expression_label,
            "dataset": "gtex_v8",
            "data_source": "gtex_api",
            "erik_eligible": True,
        },
    )


# ---------------------------------------------------------------------------
# GTExAPIConnector
# ---------------------------------------------------------------------------


class GTExAPIConnector(BaseConnector):
    """Connector for the GTEx Portal REST API — replaces local GTEx SQLite DB.

    Queries the public GTEx v2 API for median gene expression across
    ALS-relevant tissues and converts each tissue result into an EvidenceItem.

    Rate limits:
        The GTEx Portal API is publicly accessible; this connector uses a
        30-second request timeout and exponential backoff (inherited from
        BaseConnector).
    """

    BASE_URL = "https://gtexportal.org/api/v2"
    DEFAULT_HEADERS = {"Accept": "application/json"}
    DATASET_ID = "gtex_v8"

    def __init__(self, store=None, **kwargs) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # BaseConnector contract
    # ------------------------------------------------------------------

    def fetch(self, gene: str = "", uniprot: str = "", **kwargs) -> ConnectorResult:
        """Fetch GTEx expression evidence for a gene in ALS-relevant tissues.

        Parameters
        ----------
        gene:
            Gene symbol, e.g. "TARDBP".  The GTEx API accepts gene symbols
            directly in the gencodeId parameter.
        uniprot:
            Not used for GTEx lookups (GTEx is gene-centric); ignored if
            provided without gene.

        Returns
        -------
        ConnectorResult with one EvidenceItem per ALS-relevant tissue and
        any errors encountered.
        """
        result = ConnectorResult()

        if not gene:
            return result

        return self._fetch_by_gene(gene)

    # ------------------------------------------------------------------
    # Private: fetch helpers
    # ------------------------------------------------------------------

    def _fetch_by_gene(self, gene: str) -> ConnectorResult:
        """Fetch expression data and produce per-tissue EvidenceItems."""
        result = ConnectorResult()

        try:
            data = self._retry_with_backoff(
                self._get_median_expression, gene
            )
        except Exception as e:
            result.errors.append(
                f"GTEx API lookup failed for gene={gene}: {e}"
            )
            return result

        tissue_records = data.get("data", [])
        if not tissue_records:
            return result

        for record in tissue_records:
            tissue_id = record.get("tissueSiteDetailId", "")
            if tissue_id not in _ALS_RELEVANT_TISSUES:
                continue

            try:
                median_tpm_raw = record.get("median", 0.0)
                median_tpm = float(median_tpm_raw) if median_tpm_raw is not None else 0.0
                item = _parse_tissue_expression(gene, tissue_id, median_tpm)
                if self._store:
                    self._store.upsert_evidence_item(item)
                result.evidence_items_added += 1
            except Exception as e:
                result.errors.append(
                    f"GTEx tissue parsing failed for gene={gene} tissue={tissue_id}: {e}"
                )

        return result

    # ------------------------------------------------------------------
    # Private: API call wrappers
    # ------------------------------------------------------------------

    def _get_median_expression(self, gene: str) -> dict:
        """GET /expression/medianGeneExpression?gencodeId={gene}&datasetId=gtex_v8"""
        url = f"{self.BASE_URL}/expression/medianGeneExpression"
        params = {
            "gencodeId": gene,
            "datasetId": self.DATASET_ID,
        }
        resp = requests.get(
            url,
            params=params,
            headers=self.DEFAULT_HEADERS,
            timeout=self.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
