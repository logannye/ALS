"""GalenKGAPIConnector — queries Galen's REST API for ALS cross-disease knowledge.

Replaces the direct PostgreSQL connection to galen_kg for Railway deployment.
Calls Galen's HTTP API endpoints instead of connecting to the MacBook-local DB.

API endpoints used:
  - GET {GALEN_API_URL}/api/erik-bridge/kg/neighbors?gene={gene}&max_results=10&min_confidence=0.4
      Fetch neighboring entities for a gene from the Galen KG.
  - GET {GALEN_API_URL}/api/erik-bridge/kg/search?query={gene}&entity_type=gene&limit=20
      Fallback: search the Galen KG for entities matching a gene name.

NOTE: The Galen-side endpoints do not yet exist. This connector is forward-compatible —
it will return an empty result with an error entry when the endpoints are unavailable,
allowing Railway deployment to proceed without blocking on Galen connectivity.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import requests

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import Provenance, Uncertainty
from ontology.enums import EvidenceDirection, EvidenceStrength, SourceSystem
from ontology.evidence import EvidenceItem

logger = logging.getLogger(__name__)

DEFAULT_GALEN_API_URL = "http://localhost:8000"


class GalenKGAPIConnector(BaseConnector):
    """Connector for Galen's REST API — replaces direct PostgreSQL for Railway.

    Queries the Galen cancer knowledge graph via HTTP endpoints and converts
    results into EvidenceItem objects representing ALS/cancer cross-disease
    relationships.

    Environment:
        GALEN_API_URL: Base URL of Galen's REST API (default: http://localhost:8000).
    """

    DEFAULT_HEADERS = {"Accept": "application/json"}

    def __init__(self, store=None, **kwargs) -> None:
        self._store = store
        self._base_url = os.environ.get("GALEN_API_URL", DEFAULT_GALEN_API_URL).rstrip("/")

    # ------------------------------------------------------------------
    # BaseConnector contract
    # ------------------------------------------------------------------

    def fetch(self, gene: str = "", uniprot: str = "", **kwargs) -> ConnectorResult:
        """Fetch Galen KG neighbor evidence for a gene.

        Parameters
        ----------
        gene:
            Gene symbol to look up, e.g. "SIGMAR1".
        uniprot:
            Unused — present for interface compatibility with other connectors.

        Returns
        -------
        ConnectorResult with evidence_items_added count and any errors.
        """
        result = ConnectorResult()

        if not gene:
            return result

        # Try primary endpoint: /api/erik-bridge/kg/neighbors
        primary_result = self._fetch_neighbors(gene)
        if primary_result.evidence_items_added > 0 or (
            primary_result.errors and "404" not in primary_result.errors[0]
            and "Not Found" not in primary_result.errors[0]
        ):
            return primary_result

        # Fallback to /api/erik-bridge/kg/search
        fallback_result = self._fetch_search(gene)
        if primary_result.errors:
            fallback_result.errors = primary_result.errors + fallback_result.errors
        return fallback_result

    # ------------------------------------------------------------------
    # Private: endpoint helpers
    # ------------------------------------------------------------------

    def _fetch_neighbors(self, gene: str) -> ConnectorResult:
        """Call GET /api/erik-bridge/kg/neighbors for neighbor relationships."""
        result = ConnectorResult()
        url = f"{self._base_url}/api/erik-bridge/kg/neighbors"
        params = {
            "gene": gene,
            "max_results": 10,
            "min_confidence": 0.4,
        }

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
                f"GalenKG neighbors fetch failed for gene={gene}: {e}"
            )
            return result

        neighbors = data.get("neighbors") or data.get("results") or []
        for record in neighbors:
            try:
                item = _parse_neighbor_record(record, gene)
                if self._store:
                    self._store.upsert_evidence_item(item)
                result.evidence_items_added += 1
            except Exception as e:
                result.errors.append(
                    f"Failed to parse neighbor record for gene={gene}: {e}"
                )

        return result

    def _fetch_search(self, gene: str) -> ConnectorResult:
        """Call GET /api/erik-bridge/kg/search as fallback."""
        result = ConnectorResult()
        url = f"{self._base_url}/api/erik-bridge/kg/search"
        params = {
            "query": gene,
            "entity_type": "gene",
            "limit": 20,
        }

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
                f"GalenKG search fetch failed for gene={gene}: {e}"
            )
            return result

        results_list = data.get("results") or data.get("entities") or []
        for record in results_list:
            try:
                item = _parse_search_record(record, gene)
                if self._store:
                    self._store.upsert_evidence_item(item)
                result.evidence_items_added += 1
            except Exception as e:
                result.errors.append(
                    f"Failed to parse search record for gene={gene}: {e}"
                )

        return result


# ---------------------------------------------------------------------------
# Free functions: API payload parsers
# ---------------------------------------------------------------------------


def _make_evidence_id(name: str, gene: str) -> str:
    """Create a deterministic evidence ID from a result name and query gene.

    Format: evi:galen_kg:{name}_{gene} (lowercased, spaces replaced with underscores)
    """
    safe_name = name.lower().replace(" ", "_")
    safe_gene = gene.lower().replace(" ", "_")
    return f"evi:galen_kg:{safe_name}_{safe_gene}"


def _parse_neighbor_record(record: dict, gene: str) -> EvidenceItem:
    """Parse one neighbor record from /api/erik-bridge/kg/neighbors into EvidenceItem.

    Parameters
    ----------
    record:
        Dict with keys like name, entity_type, relationship_type, confidence.
    gene:
        The query gene symbol (used for evidence ID and claim construction).

    Returns
    -------
    EvidenceItem with PCH layer 1 (observational cross-disease link).
    """
    name = record.get("name") or record.get("target_name") or record.get("entity_name", "")
    entity_type = record.get("entity_type") or record.get("target_type", "")
    rel_type = record.get("relationship_type") or record.get("relation", "related_to")
    confidence = float(record.get("confidence", 0.6))

    claim = (
        f"[Galen cross-reference] {gene} {rel_type} {name}"
        + (f" ({entity_type})" if entity_type else "")
    )

    item_id = _make_evidence_id(name, gene)

    return EvidenceItem(
        id=item_id,
        claim=claim,
        direction=EvidenceDirection.supports,
        strength=EvidenceStrength.emerging,
        provenance=Provenance(
            source_system=SourceSystem.database,
            asserted_by="galen_kg_api_connector",
        ),
        uncertainty=Uncertainty(confidence=confidence),
        body={
            "protocol_layer": "root_cause_suppression",
            "pch_layer": 1,
            "applicable_subtypes": ["sporadic_tdp43", "unresolved"],
            "erik_eligible": True,
            "query_gene": gene,
            "neighbor_name": name,
            "neighbor_type": entity_type,
            "relationship_type": rel_type,
            "data_source": "galen_kg_api",
        },
    )


def _parse_search_record(record: dict, gene: str) -> EvidenceItem:
    """Parse one search result from /api/erik-bridge/kg/search into EvidenceItem.

    Parameters
    ----------
    record:
        Dict with keys like name, entity_type, description.
    gene:
        The query gene symbol.

    Returns
    -------
    EvidenceItem with PCH layer 1.
    """
    name = record.get("name") or record.get("entity_name", "")
    entity_type = record.get("entity_type") or record.get("type", "")
    description = record.get("description") or record.get("summary", "")
    confidence = float(record.get("confidence", 0.6))

    claim = (
        f"[Galen cross-reference] {name}"
        + (f" ({entity_type})" if entity_type else "")
        + (f": {description}" if description else f" is related to {gene}")
    )

    item_id = _make_evidence_id(name, gene)

    return EvidenceItem(
        id=item_id,
        claim=claim,
        direction=EvidenceDirection.supports,
        strength=EvidenceStrength.emerging,
        provenance=Provenance(
            source_system=SourceSystem.database,
            asserted_by="galen_kg_api_connector",
        ),
        uncertainty=Uncertainty(confidence=confidence),
        body={
            "protocol_layer": "root_cause_suppression",
            "pch_layer": 1,
            "applicable_subtypes": ["sporadic_tdp43", "unresolved"],
            "erik_eligible": True,
            "query_gene": gene,
            "entity_name": name,
            "entity_type": entity_type,
            "data_source": "galen_kg_api",
        },
    )
