"""GWASCatalogAPIConnector — queries the GWAS Catalog REST API for SNP associations.

Replaces local TSV file-based access for Railway deployment.
Uses the EBI GWAS Catalog REST API at https://www.ebi.ac.uk/gwas/rest/api.

API endpoints used:
  - GET /singleNucleotidePolymorphisms/search/findByGene?geneName={gene}
      Search for SNPs associated with a gene name.

Returns EvidenceItem objects containing rsId, chromosome position, and
risk allele data for ALS-relevant genetic loci.
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


def _parse_snp_record(record: dict, gene: str = "") -> EvidenceItem:
    """Parse one GWAS Catalog SNP record into an EvidenceItem.

    Parameters
    ----------
    record:
        Dict from the GWAS Catalog SNP search response.
    gene:
        Gene symbol for which the SNP was queried.

    Returns
    -------
    EvidenceItem with PCH layer 1 (associational evidence — population GWAS).
    """
    rs_id = record.get("rsId", "")
    chromosome = record.get("chromosome_name", "")
    position = record.get("chromosome_position", "")

    # Extract risk alleles from associations list if present
    risk_alleles: list[str] = []
    associations = record.get("associations", {})
    if isinstance(associations, dict):
        embedded = associations.get("_embedded", {})
        assoc_list = embedded.get("associations", [])
        for assoc in assoc_list:
            for locus in assoc.get("loci", []):
                for ra in locus.get("strongestRiskAlleles", []):
                    allele = ra.get("riskAlleleName", "")
                    if allele:
                        risk_alleles.append(allele)

    location_str = f"chr{chromosome}:{position}" if chromosome and position else "unknown position"
    allele_str = f" risk alleles: {', '.join(risk_alleles[:3])}" if risk_alleles else ""

    claim = (
        f"GWAS Catalog SNP {rs_id} at {location_str}"
        + (f" for gene {gene}" if gene else "")
        + allele_str
        + "."
    )

    item_id = f"evi:gwas:{rs_id}_{gene.lower()}" if gene else f"evi:gwas:{rs_id}"

    return EvidenceItem(
        id=item_id,
        claim=claim,
        direction=EvidenceDirection.supports,
        strength=EvidenceStrength.emerging,
        provenance=Provenance(
            source_system=SourceSystem.database,
            asserted_by="gwas_catalog_api_connector",
        ),
        uncertainty=Uncertainty(confidence=0.7),
        body={
            "protocol_layer": "root_cause_suppression",
            "pch_layer": 1,
            "applicable_subtypes": ["sporadic_tdp43", "unresolved"],
            "erik_eligible": True,
            "rs_id": rs_id,
            "gene": gene,
            "chromosome": chromosome,
            "position": position,
            "risk_alleles": risk_alleles,
            "data_source": "gwas_catalog_api",
        },
    )


# ---------------------------------------------------------------------------
# GWASCatalogAPIConnector
# ---------------------------------------------------------------------------


class GWASCatalogAPIConnector(BaseConnector):
    """Connector for the GWAS Catalog REST API.

    Queries the public EBI GWAS Catalog API for SNP associations linked to
    a gene, extracting rsId, chromosome position, and risk allele data.

    Rate limits:
        The GWAS Catalog REST API is freely accessible; this connector uses a
        30-second request timeout and exponential backoff (inherited from
        BaseConnector) to handle transient failures gracefully.
    """

    BASE_URL = "https://www.ebi.ac.uk/gwas/rest/api"
    DEFAULT_HEADERS = {"Accept": "application/json"}

    def __init__(self, store=None, **kwargs) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # BaseConnector contract
    # ------------------------------------------------------------------

    def fetch(self, gene: str = "", uniprot: str = "", **kwargs) -> ConnectorResult:
        """Fetch SNP association evidence for a gene from GWAS Catalog.

        Parameters
        ----------
        gene:
            Gene symbol to search, e.g. "TARDBP".
        uniprot:
            Unused; accepted for interface compatibility.

        Returns
        -------
        ConnectorResult with evidence_items_added count and any errors.
        """
        result = ConnectorResult()

        if not gene:
            return result

        return self._fetch_by_gene(gene)

    # ------------------------------------------------------------------
    # Private: fetch helpers
    # ------------------------------------------------------------------

    def _fetch_by_gene(self, gene: str) -> ConnectorResult:
        """Fetch SNP associations for a gene symbol from GWAS Catalog API."""
        result = ConnectorResult()
        url = f"{self.BASE_URL}/singleNucleotidePolymorphisms/search/findByGene"
        params = {"geneName": gene}

        try:
            resp = self._retry_with_backoff(
                requests.get,
                url,
                params=params,
                headers=self.DEFAULT_HEADERS,
                timeout=self.REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            result.errors.append(
                f"GWAS Catalog SNP fetch failed for gene={gene}: {e}"
            )
            return result

        # Response is wrapped in _embedded.singleNucleotidePolymorphisms
        embedded = data.get("_embedded", {})
        snps = embedded.get("singleNucleotidePolymorphisms", [])

        if not snps:
            return result

        for record in snps:
            try:
                item = _parse_snp_record(record, gene)
                if self._store:
                    self._store.upsert_evidence_item(item)
                result.evidence_items_added += 1
            except Exception as e:
                rs_id = record.get("rsId", "unknown")
                result.errors.append(
                    f"Failed to parse GWAS SNP record for rsId={rs_id}: {e}"
                )

        return result
