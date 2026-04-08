"""UniProtAPIConnector — queries the UniProt REST API for protein annotations.

Replaces the local UniProt/SwissProt TSV for Railway deployment.
Uses the public UniProt REST API at https://rest.uniprot.org/uniprotkb.

API endpoints used:
  - GET /{accession}.json
      Fetch a protein entry by UniProt accession.
  - GET /search?query=gene_exact:{gene}+AND+organism_id:9606&format=json&size=1
      Search for a human protein by exact gene symbol.

Evidence aspects produced per protein entry:
  - function  — Functional comment (what the protein does)
  - disease   — Disease involvement annotations
  - location  — Subcellular localisation
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


def _extract_comment_text(entry: dict, comment_type: str) -> str:
    """Extract plain text from a UniProt comment section.

    Parameters
    ----------
    entry:
        Parsed UniProt JSON entry.
    comment_type:
        UniProt comment type string, e.g. "FUNCTION", "DISEASE", "SUBCELLULAR_LOCATION".

    Returns
    -------
    Concatenated text values, or empty string if absent.
    """
    comments = entry.get("comments", [])
    parts: list[str] = []
    for comment in comments:
        if comment.get("commentType", "") != comment_type:
            continue
        # FUNCTION / TISSUE_SPECIFICITY / INDUCTION etc. use "texts" list
        for text_obj in comment.get("texts", []):
            value = text_obj.get("value", "")
            if value:
                parts.append(value)
        # DISEASE uses nested "disease" dict
        disease = comment.get("disease", {})
        if disease:
            disease_name = disease.get("diseaseId", "") or disease.get("diseaseName", "")
            description = disease.get("description", "")
            if disease_name:
                parts.append(disease_name + (f": {description}" if description else ""))
        # SUBCELLULAR_LOCATION uses "subcellularLocations" list
        for subloc in comment.get("subcellularLocations", []):
            loc = subloc.get("location", {})
            value = loc.get("value", "")
            if value:
                parts.append(value)
    return " | ".join(parts)


def _parse_function_item(entry: dict, accession: str, gene: str) -> Optional[EvidenceItem]:
    """Extract function evidence from a UniProt JSON entry.

    Returns None if no function comment is present.
    """
    text = _extract_comment_text(entry, "FUNCTION")
    if not text:
        return None

    claim = f"{gene} ({accession}) function: {text[:500]}"
    item_id = f"evi:uniprot:{accession.lower()}_function"

    return EvidenceItem(
        id=item_id,
        claim=claim,
        direction=EvidenceDirection.supports,
        strength=EvidenceStrength.strong,
        provenance=Provenance(
            source_system=SourceSystem.database,
            asserted_by="uniprot_api_connector",
        ),
        uncertainty=Uncertainty(confidence=0.9),
        body={
            "pch_layer": 1,
            "gene": gene,
            "accession": accession,
            "aspect": "function",
            "function_text": text[:1000],
            "data_source": "uniprot_api",
            "erik_eligible": True,
        },
    )


def _parse_disease_item(entry: dict, accession: str, gene: str) -> Optional[EvidenceItem]:
    """Extract disease involvement evidence from a UniProt JSON entry.

    Returns None if no disease comment is present.
    """
    text = _extract_comment_text(entry, "DISEASE")
    if not text:
        return None

    claim = f"{gene} ({accession}) disease involvement: {text[:500]}"
    item_id = f"evi:uniprot:{accession.lower()}_disease"

    return EvidenceItem(
        id=item_id,
        claim=claim,
        direction=EvidenceDirection.supports,
        strength=EvidenceStrength.moderate,
        provenance=Provenance(
            source_system=SourceSystem.database,
            asserted_by="uniprot_api_connector",
        ),
        uncertainty=Uncertainty(confidence=0.85),
        body={
            "pch_layer": 1,
            "gene": gene,
            "accession": accession,
            "aspect": "disease",
            "disease_text": text[:1000],
            "data_source": "uniprot_api",
            "erik_eligible": True,
        },
    )


def _parse_location_item(entry: dict, accession: str, gene: str) -> Optional[EvidenceItem]:
    """Extract subcellular location evidence from a UniProt JSON entry.

    Returns None if no subcellular location comment is present.
    """
    text = _extract_comment_text(entry, "SUBCELLULAR_LOCATION")
    if not text:
        return None

    claim = f"{gene} ({accession}) subcellular location: {text[:400]}"
    item_id = f"evi:uniprot:{accession.lower()}_location"

    return EvidenceItem(
        id=item_id,
        claim=claim,
        direction=EvidenceDirection.supports,
        strength=EvidenceStrength.strong,
        provenance=Provenance(
            source_system=SourceSystem.database,
            asserted_by="uniprot_api_connector",
        ),
        uncertainty=Uncertainty(confidence=0.9),
        body={
            "pch_layer": 1,
            "gene": gene,
            "accession": accession,
            "aspect": "location",
            "location_text": text[:1000],
            "data_source": "uniprot_api",
            "erik_eligible": True,
        },
    )


# ---------------------------------------------------------------------------
# UniProtAPIConnector
# ---------------------------------------------------------------------------


class UniProtAPIConnector(BaseConnector):
    """Connector for the UniProt REST API — replaces local SwissProt TSV.

    Queries the public UniProt REST API for protein function, disease
    involvement, and subcellular localisation data and converts each aspect
    into an EvidenceItem.

    Rate limits:
        UniProt REST API is publicly accessible; this connector uses a 30-second
        request timeout and exponential backoff (inherited from BaseConnector).
    """

    BASE_URL = "https://rest.uniprot.org/uniprotkb"
    DEFAULT_HEADERS = {"Accept": "application/json"}

    def __init__(self, store=None, **kwargs) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # BaseConnector contract
    # ------------------------------------------------------------------

    def fetch(self, gene: str = "", uniprot: str = "", **kwargs) -> ConnectorResult:
        """Fetch protein annotation evidence for a gene or UniProt accession.

        Parameters
        ----------
        gene:
            Gene symbol, e.g. "TARDBP". Used if uniprot is not provided.
        uniprot:
            UniProt accession, e.g. "Q13148".

        Returns
        -------
        ConnectorResult with evidence_items_added count and any errors.
        """
        result = ConnectorResult()

        if not gene and not uniprot:
            return result

        if uniprot:
            return self._fetch_by_accession(uniprot)
        else:
            return self._fetch_by_gene(gene)

    # ------------------------------------------------------------------
    # Private: fetch helpers
    # ------------------------------------------------------------------

    def _fetch_by_accession(self, accession: str) -> ConnectorResult:
        """Fetch entry by UniProt accession: GET /{accession}.json"""
        result = ConnectorResult()

        try:
            entry = self._retry_with_backoff(self._get_entry_by_accession, accession)
        except Exception as e:
            result.errors.append(
                f"UniProt API lookup failed for accession={accession}: {e}"
            )
            return result

        if not entry:
            return result

        gene = self._primary_gene(entry) or accession
        return self._extract_evidence(entry, accession, gene)

    def _fetch_by_gene(self, gene: str) -> ConnectorResult:
        """Fetch entry by gene symbol: GET /search?query=gene_exact:{gene}..."""
        result = ConnectorResult()

        try:
            entry, accession = self._retry_with_backoff(
                self._search_by_gene, gene
            )
        except Exception as e:
            result.errors.append(
                f"UniProt API search failed for gene={gene}: {e}"
            )
            return result

        if not entry or not accession:
            return result

        return self._extract_evidence(entry, accession, gene)

    def _extract_evidence(
        self, entry: dict, accession: str, gene: str
    ) -> ConnectorResult:
        """Parse all evidence aspects from a UniProt entry."""
        result = ConnectorResult()

        parsers = [
            _parse_function_item,
            _parse_disease_item,
            _parse_location_item,
        ]

        for parser in parsers:
            try:
                item = parser(entry, accession, gene)
                if item is None:
                    continue
                if self._store:
                    self._store.upsert_evidence_item(item)
                result.evidence_items_added += 1
            except Exception as e:
                result.errors.append(
                    f"UniProt evidence parsing failed for {accession} ({parser.__name__}): {e}"
                )

        return result

    # ------------------------------------------------------------------
    # Private: API call wrappers
    # ------------------------------------------------------------------

    def _get_entry_by_accession(self, accession: str) -> dict:
        """GET /{accession}.json — fetch a single entry by accession."""
        url = f"{self.BASE_URL}/{accession}.json"
        resp = requests.get(
            url,
            headers=self.DEFAULT_HEADERS,
            timeout=self.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def _search_by_gene(self, gene: str) -> tuple[dict, str]:
        """GET /search?query=gene_exact:{gene}+AND+organism_id:9606&format=json&size=1

        Returns
        -------
        (entry_dict, accession) — first match, or ({}, "") if no results.
        """
        url = f"{self.BASE_URL}/search"
        params = {
            "query": f"gene_exact:{gene} AND organism_id:9606",
            "format": "json",
            "size": 1,
        }
        resp = requests.get(
            url,
            params=params,
            headers=self.DEFAULT_HEADERS,
            timeout=self.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return {}, ""
        entry = results[0]
        accession = entry.get("primaryAccession", "")
        return entry, accession

    @staticmethod
    def _primary_gene(entry: dict) -> str:
        """Extract the primary gene name from a UniProt entry dict."""
        genes = entry.get("genes", [])
        if not genes:
            return ""
        primary = genes[0]
        gene_name = primary.get("geneName", {})
        return gene_name.get("value", "")
