"""GnomADAPIConnector — queries the gnomAD GraphQL API for gene constraint metrics.

Replaces local TSV file-based access for Railway deployment.
Uses the gnomAD GraphQL API at https://gnomad.broadinstitute.org/api.

API:
  - POST https://gnomad.broadinstitute.org/api (GraphQL)
      Query gene constraint metrics: pLI, LOEUF, missense z-score, oe_lof, oe_mis.

Returns EvidenceItem objects containing gene constraint metrics critical for
interpreting variant pathogenicity and loss-of-function intolerance.
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

# GraphQL query for gene constraint metrics
_CONSTRAINT_QUERY = """
query GeneConstraint($geneSymbol: String!) {
  gene(gene_symbol: $geneSymbol, reference_genome: GRCh38) {
    gnomad_constraint {
      pLI
      oe_lof
      oe_lof_upper
      oe_mis
      mis_z
    }
  }
}
"""


# ---------------------------------------------------------------------------
# Free functions: API payload parsers
# ---------------------------------------------------------------------------


def _parse_constraint_record(constraint: dict, gene: str) -> EvidenceItem:
    """Parse a gnomAD constraint record into an EvidenceItem.

    Parameters
    ----------
    constraint:
        Dict from gnomAD GraphQL response gene.gnomad_constraint.
    gene:
        Gene symbol queried.

    Returns
    -------
    EvidenceItem with PCH layer 1 (associational evidence — population constraint).
    """
    pli = constraint.get("pLI")
    oe_lof = constraint.get("oe_lof")
    loeuf = constraint.get("oe_lof_upper")  # LOEUF = oe_lof_upper
    oe_mis = constraint.get("oe_mis")
    mis_z = constraint.get("mis_z")

    # Build a human-readable claim
    parts: list[str] = [f"gnomAD constraint metrics for {gene}:"]

    if pli is not None:
        try:
            pli_f = float(pli)
            if pli_f > 0.9:
                parts.append(f"pLI={pli_f:.3f} (highly constrained — intolerant to LoF)")
            elif pli_f > 0.5:
                parts.append(f"pLI={pli_f:.3f} (moderately constrained)")
            else:
                parts.append(f"pLI={pli_f:.3f} (LoF-tolerant)")
        except (TypeError, ValueError):
            parts.append(f"pLI={pli}")

    if loeuf is not None:
        try:
            loeuf_f = float(loeuf)
            if loeuf_f < 0.35:
                parts.append(f"LOEUF={loeuf_f:.3f} (strong LoF intolerance)")
            else:
                parts.append(f"LOEUF={loeuf_f:.3f}")
        except (TypeError, ValueError):
            parts.append(f"LOEUF={loeuf}")

    if mis_z is not None:
        try:
            mis_z_f = float(mis_z)
            if mis_z_f > 3.09:
                parts.append(f"mis_z={mis_z_f:.2f} (missense-constrained)")
            else:
                parts.append(f"mis_z={mis_z_f:.2f}")
        except (TypeError, ValueError):
            parts.append(f"mis_z={mis_z}")

    if oe_mis is not None:
        parts.append(f"oe_mis={oe_mis}")

    claim = " ".join(parts) + "."

    # Derive strength from pLI if available
    try:
        pli_f = float(pli) if pli is not None else 0.0
    except (TypeError, ValueError):
        pli_f = 0.0

    if pli_f > 0.9:
        strength = EvidenceStrength.strong
        confidence = 0.9
    elif pli_f > 0.5:
        strength = EvidenceStrength.moderate
        confidence = 0.7
    else:
        strength = EvidenceStrength.emerging
        confidence = 0.5

    item_id = f"evi:gnomad:{gene.lower()}_constraint"

    return EvidenceItem(
        id=item_id,
        claim=claim,
        direction=EvidenceDirection.supports,
        strength=strength,
        provenance=Provenance(
            source_system=SourceSystem.database,
            asserted_by="gnomad_api_connector",
        ),
        uncertainty=Uncertainty(confidence=confidence),
        body={
            "protocol_layer": "root_cause_suppression",
            "pch_layer": 1,
            "applicable_subtypes": ["sporadic_tdp43", "unresolved"],
            "erik_eligible": True,
            "gene": gene,
            "pLI": pli,
            "oe_lof": oe_lof,
            "LOEUF": loeuf,
            "oe_mis": oe_mis,
            "mis_z": mis_z,
            "data_source": "gnomad_api",
        },
    )


# ---------------------------------------------------------------------------
# GnomADAPIConnector
# ---------------------------------------------------------------------------


class GnomADAPIConnector(BaseConnector):
    """Connector for the gnomAD GraphQL API.

    Queries the public gnomAD GraphQL endpoint for gene constraint metrics
    (pLI, LOEUF, missense z-score) and converts results into EvidenceItem
    objects.

    Uses POST requests with a JSON body (GraphQL), not GET.

    Rate limits:
        The gnomAD API is freely accessible; this connector uses a
        30-second request timeout and exponential backoff (inherited from
        BaseConnector) to handle transient failures gracefully.
    """

    BASE_URL = "https://gnomad.broadinstitute.org/api"
    DEFAULT_HEADERS = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    def __init__(self, store=None, **kwargs) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # BaseConnector contract
    # ------------------------------------------------------------------

    def fetch(self, gene: str = "", uniprot: str = "", **kwargs) -> ConnectorResult:
        """Fetch gene constraint evidence from gnomAD GraphQL API.

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

        return self._fetch_constraint(gene)

    # ------------------------------------------------------------------
    # Private: fetch helpers
    # ------------------------------------------------------------------

    def _fetch_constraint(self, gene: str) -> ConnectorResult:
        """Fetch constraint metrics for a gene via gnomAD GraphQL API."""
        result = ConnectorResult()

        payload = {
            "query": _CONSTRAINT_QUERY,
            "variables": {"geneSymbol": gene},
        }

        try:
            resp = self._retry_with_backoff(
                requests.post,
                self.BASE_URL,
                json=payload,
                headers=self.DEFAULT_HEADERS,
                timeout=self.REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            result.errors.append(
                f"gnomAD constraint fetch failed for gene={gene}: {e}"
            )
            return result

        # Navigate GraphQL response: data.gene.gnomad_constraint
        gql_data = data.get("data", {})
        gene_data = gql_data.get("gene")
        if not gene_data:
            return result

        constraint = gene_data.get("gnomad_constraint")
        if not constraint:
            return result

        try:
            item = _parse_constraint_record(constraint, gene)
            if self._store:
                self._store.upsert_evidence_item(item)
            result.evidence_items_added += 1
        except Exception as e:
            result.errors.append(
                f"Failed to parse gnomAD constraint record for gene={gene}: {e}"
            )

        return result
