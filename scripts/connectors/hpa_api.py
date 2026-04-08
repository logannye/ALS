"""HPAAPIConnector — queries the Human Protein Atlas REST API for protein profiles.

Replaces local TSV file-based access for Railway deployment.
Uses the Human Protein Atlas API at https://www.proteinatlas.org.

API endpoints used:
  - GET /{gene}.json
      Fetch full protein profile for a gene: protein class, biological process,
      gene description, and subcellular location.

Returns EvidenceItem objects containing protein expression and localisation
data for ALS drug targets.
"""
from __future__ import annotations

import logging
from typing import Optional

import requests

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import Provenance, Uncertainty
from ontology.enums import EvidenceDirection, EvidenceStrength, SourceSystem
from ontology.evidence import EvidenceItem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Free functions: API payload parsers
# ---------------------------------------------------------------------------


def _parse_hpa_record(data: dict, gene: str) -> EvidenceItem:
    """Parse one HPA gene profile record into an EvidenceItem.

    Parameters
    ----------
    data:
        Dict from the HPA JSON endpoint for a gene.
    gene:
        Gene symbol queried.

    Returns
    -------
    EvidenceItem with PCH layer 1 (associational evidence — protein profiling).
    """
    # HPA JSON structure: top-level keys include proteinClasses, biologicalProcess,
    # geneSummary / proteinSummary, subcellularLocation, chromosome, uniprotId
    protein_classes = data.get("proteinClasses", [])
    if isinstance(protein_classes, list):
        protein_class_str = ", ".join(
            pc.get("name", "") for pc in protein_classes if pc.get("name")
        )
    else:
        protein_class_str = str(protein_classes)

    bio_processes = data.get("biologicalProcess", [])
    if isinstance(bio_processes, list):
        bio_process_str = ", ".join(
            bp.get("name", "") for bp in bio_processes[:5] if bp.get("name")
        )
    else:
        bio_process_str = str(bio_processes)

    subcellular_locations = data.get("subcellularLocation", [])
    if isinstance(subcellular_locations, list):
        subcellular_str = ", ".join(
            sl.get("name", "") for sl in subcellular_locations[:5] if sl.get("name")
        )
    else:
        subcellular_str = str(subcellular_locations)

    gene_description = data.get("geneSummary", "") or data.get("proteinSummary", "")

    # Build claim
    claim_parts: list[str] = [f"Human Protein Atlas profile for {gene}:"]
    if protein_class_str:
        claim_parts.append(f"protein class: {protein_class_str[:100]}.")
    if bio_process_str:
        claim_parts.append(f"biological process: {bio_process_str[:100]}.")
    if subcellular_str:
        claim_parts.append(f"subcellular location: {subcellular_str[:80]}.")
    if gene_description:
        claim_parts.append(f"{gene_description[:120]}.")

    claim = " ".join(claim_parts)

    item_id = f"evi:hpa:{gene.lower()}_profile"

    return EvidenceItem(
        id=item_id,
        claim=claim,
        direction=EvidenceDirection.supports,
        strength=EvidenceStrength.moderate,
        provenance=Provenance(
            source_system=SourceSystem.database,
            asserted_by="hpa_api_connector",
        ),
        uncertainty=Uncertainty(confidence=0.8),
        body={
            "protocol_layer": "root_cause_suppression",
            "pch_layer": 1,
            "applicable_subtypes": ["sporadic_tdp43", "unresolved"],
            "erik_eligible": True,
            "gene": gene,
            "protein_class": protein_class_str,
            "biological_process": bio_process_str,
            "subcellular_location": subcellular_str,
            "gene_description": gene_description[:200] if gene_description else "",
            "uniprot_id": data.get("uniprotIds", [None])[0]
            if isinstance(data.get("uniprotIds"), list)
            else data.get("uniprotId", ""),
            "chromosome": data.get("chromosome", ""),
            "data_source": "hpa_api",
        },
    )


# ---------------------------------------------------------------------------
# HPAAPIConnector
# ---------------------------------------------------------------------------


class HPAAPIConnector(BaseConnector):
    """Connector for the Human Protein Atlas REST API.

    Queries the public HPA JSON endpoint for protein class, biological process,
    gene description, and subcellular location data, converting results into
    EvidenceItem objects.

    Rate limits:
        The HPA API is freely accessible; this connector uses a
        30-second request timeout and exponential backoff (inherited from
        BaseConnector) to handle transient failures gracefully.
    """

    BASE_URL = "https://www.proteinatlas.org"
    DEFAULT_HEADERS = {"Accept": "application/json"}

    def __init__(self, store=None, **kwargs) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # BaseConnector contract
    # ------------------------------------------------------------------

    def fetch(self, gene: str = "", uniprot: str = "", **kwargs) -> ConnectorResult:
        """Fetch protein profile evidence for a gene from Human Protein Atlas.

        Parameters
        ----------
        gene:
            Gene symbol to query, e.g. "TARDBP".
        uniprot:
            Unused; accepted for interface compatibility.

        Returns
        -------
        ConnectorResult with evidence_items_added count and any errors.
        """
        result = ConnectorResult()

        if not gene:
            return result

        return self._fetch_profile(gene)

    # ------------------------------------------------------------------
    # Private: fetch helpers
    # ------------------------------------------------------------------

    def _fetch_profile(self, gene: str) -> ConnectorResult:
        """Fetch protein profile for a gene from the HPA JSON API."""
        result = ConnectorResult()
        url = f"{self.BASE_URL}/{gene}.json"

        try:
            resp = self._retry_with_backoff(
                requests.get,
                url,
                headers=self.DEFAULT_HEADERS,
                timeout=self.REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            result.errors.append(
                f"HPA profile fetch failed for gene={gene}: {e}"
            )
            return result

        if not data:
            return result

        try:
            item = _parse_hpa_record(data, gene)
            if self._store:
                self._store.upsert_evidence_item(item)
            result.evidence_items_added += 1
        except Exception as e:
            result.errors.append(
                f"Failed to parse HPA profile record for gene={gene}: {e}"
            )

        return result
