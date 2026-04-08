"""AlphaFoldAPIConnector — queries the AlphaFold EBI API for protein structure data.

Replaces the local AlphaFold PDB file collection for Railway deployment.
Uses the public AlphaFold EBI API at https://alphafold.ebi.ac.uk/api.

API endpoints used:
  - GET /prediction/{uniprot_accession}
      Retrieve AlphaFold prediction metadata for a UniProt accession.
      Returns a list of prediction entries; the first is used.

Key fields extracted:
  - globalMetricValue  — Global pLDDT score (0–100, confidence metric)
  - entryId            — AlphaFold entry ID (e.g. AF-P00441-F1)
  - uniprotStart / uniprotEnd — Residue coverage
"""
from __future__ import annotations

import logging

import requests

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import Provenance, Uncertainty
from ontology.enums import EvidenceDirection, EvidenceStrength, SourceSystem
from ontology.evidence import EvidenceItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Free functions: API payload parsers
# ---------------------------------------------------------------------------


def _parse_prediction_entry(prediction: dict, uniprot: str) -> EvidenceItem:
    """Parse one AlphaFold prediction entry into an EvidenceItem.

    Parameters
    ----------
    prediction:
        Dict from the AlphaFold API prediction list.
    uniprot:
        UniProt accession used for the lookup (fallback if entry is missing it).

    Returns
    -------
    EvidenceItem with pch_layer 1 (associational — structural prediction).
    """
    entry_id = prediction.get("entryId", f"AF-{uniprot}-F1")
    plddt = prediction.get("globalMetricValue")
    uniprot_acc = prediction.get("uniprotAccession", uniprot)
    uniprot_start = prediction.get("uniprotStart", 1)
    uniprot_end = prediction.get("uniprotEnd", "?")
    model_url = prediction.get("pdbUrl", "")
    cif_url = prediction.get("cifUrl", "")

    try:
        plddt_float = float(plddt) if plddt is not None else 0.0
    except (TypeError, ValueError):
        plddt_float = 0.0

    # Confidence classification
    if plddt_float >= 90:
        confidence_label = "very high confidence (pLDDT ≥ 90) — suitable for drug design"
        strength = EvidenceStrength.strong
        confidence_score = 0.95
    elif plddt_float >= 70:
        confidence_label = "high confidence (pLDDT ≥ 70) — core structure reliable"
        strength = EvidenceStrength.strong
        confidence_score = 0.80
    elif plddt_float >= 50:
        confidence_label = "moderate confidence (pLDDT ≥ 50) — use with caution"
        strength = EvidenceStrength.moderate
        confidence_score = 0.60
    else:
        confidence_label = "low confidence (pLDDT < 50) — unreliable for drug design"
        strength = EvidenceStrength.emerging
        confidence_score = 0.40

    claim = (
        f"AlphaFold structure for {uniprot_acc} (entry {entry_id}): "
        f"global pLDDT = {plddt_float:.1f} — {confidence_label}. "
        f"Coverage: residues {uniprot_start}–{uniprot_end}."
    )

    item_id = f"evi:alphafold:{uniprot_acc.lower()}"

    return EvidenceItem(
        id=item_id,
        claim=claim,
        direction=EvidenceDirection.supports,
        strength=strength,
        provenance=Provenance(
            source_system=SourceSystem.database,
            asserted_by="alphafold_api_connector",
        ),
        uncertainty=Uncertainty(confidence=confidence_score),
        body={
            "pch_layer": 1,
            "uniprot": uniprot_acc,
            "entry_id": entry_id,
            "global_plddt": plddt_float,
            "uniprot_start": uniprot_start,
            "uniprot_end": uniprot_end,
            "pdb_url": model_url,
            "cif_url": cif_url,
            "data_source": "alphafold_api",
            "erik_eligible": True,
        },
    )


# ---------------------------------------------------------------------------
# AlphaFoldAPIConnector
# ---------------------------------------------------------------------------


class AlphaFoldAPIConnector(BaseConnector):
    """Connector for the AlphaFold EBI API — replaces local PDB file collection.

    Queries the public AlphaFold API for protein structure predictions and
    converts the global pLDDT confidence score into an EvidenceItem.

    Rate limits:
        The AlphaFold EBI API is publicly accessible; this connector uses a
        30-second request timeout and exponential backoff (inherited from
        BaseConnector).
    """

    BASE_URL = "https://alphafold.ebi.ac.uk/api"
    DEFAULT_HEADERS = {"Accept": "application/json"}

    def __init__(self, store=None, **kwargs) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # BaseConnector contract
    # ------------------------------------------------------------------

    def fetch(self, gene: str = "", uniprot: str = "", **kwargs) -> ConnectorResult:
        """Fetch AlphaFold structure evidence for a UniProt accession.

        Parameters
        ----------
        gene:
            Gene symbol — not used directly; provide uniprot instead.
        uniprot:
            UniProt accession, e.g. "P00441" (SOD1).

        Returns
        -------
        ConnectorResult with evidence_items_added count and any errors.

        Notes
        -----
        The AlphaFold API requires a UniProt accession for direct lookup.
        Gene names are not supported by the API itself.  If only ``gene``
        is supplied, the connector returns an empty result.
        """
        result = ConnectorResult()

        if not uniprot:
            return result

        return self._fetch_by_uniprot(uniprot)

    # ------------------------------------------------------------------
    # Private: fetch helpers
    # ------------------------------------------------------------------

    def _fetch_by_uniprot(self, uniprot: str) -> ConnectorResult:
        """Fetch prediction entry: GET /prediction/{uniprot_accession}"""
        result = ConnectorResult()

        try:
            predictions = self._retry_with_backoff(
                self._get_predictions, uniprot
            )
        except Exception as e:
            result.errors.append(
                f"AlphaFold API lookup failed for uniprot={uniprot}: {e}"
            )
            return result

        if not predictions:
            return result

        # Use the first (and typically only) prediction entry
        prediction = predictions[0]

        try:
            item = _parse_prediction_entry(prediction, uniprot)
            if self._store:
                self._store.upsert_evidence_item(item)
            result.evidence_items_added += 1
        except Exception as e:
            result.errors.append(
                f"AlphaFold evidence parsing failed for uniprot={uniprot}: {e}"
            )

        return result

    # ------------------------------------------------------------------
    # Private: API call wrappers
    # ------------------------------------------------------------------

    def _get_predictions(self, uniprot: str) -> list[dict]:
        """GET /prediction/{uniprot} — returns list of prediction dicts."""
        url = f"{self.BASE_URL}/prediction/{uniprot}"
        resp = requests.get(
            url,
            headers=self.DEFAULT_HEADERS,
            timeout=self.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        # API returns a list directly
        if isinstance(data, list):
            return data
        return []
