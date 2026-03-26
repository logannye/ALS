"""OpenTargetsConnector — queries the OpenTargets GraphQL API for ALS targets.

Uses the public GraphQL endpoint at api.platform.opentargets.org to fetch
target-disease association scores for ALS (EFO_0000253).
"""
from __future__ import annotations

import logging
from typing import Optional

import requests

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import Provenance
from ontology.enums import EvidenceDirection, EvidenceStrength, SourceSystem
from ontology.evidence import EvidenceItem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Free function: parse a single association row from the GraphQL response
# ---------------------------------------------------------------------------

def _parse_target_association(row: dict) -> EvidenceItem:
    """Parse one OpenTargets associatedTargets row into an EvidenceItem.

    Parameters
    ----------
    row:
        Dict with keys: target (id, approvedSymbol), score, datatypeScores.

    Returns
    -------
    EvidenceItem with PCH layer 1 (associational evidence).
    """
    target_info = row.get("target", {})
    ensembl_id = target_info.get("id", "")
    symbol = target_info.get("approvedSymbol", ensembl_id)
    score = float(row.get("score", 0.0))

    # Extract datatype scores into a lookup dict
    datatype_scores: dict[str, float] = {}
    for ds in row.get("datatypeScores", []):
        ds_id = ds.get("id", "")
        ds_score = float(ds.get("score", 0.0))
        datatype_scores[ds_id] = ds_score

    genetic_association = datatype_scores.get("genetic_association", 0.0)
    known_drug_count = datatype_scores.get("known_drug", 0.0)

    claim = f"{symbol} is associated with ALS (overall score={score:.2f})"

    from ontology.base import Uncertainty
    item = EvidenceItem(
        id=f"evi:ot:{ensembl_id}_als",
        claim=claim,
        direction=EvidenceDirection.insufficient,
        strength=EvidenceStrength.unknown,
        provenance=Provenance(
            source_system=SourceSystem.database,
            asserted_by="opentargets_connector",
        ),
        uncertainty=Uncertainty(confidence=score),
        body={
            "protocol_layer": "",
            "mechanism_target": symbol,
            "applicable_subtypes": ["sporadic_tdp43", "unresolved"],
            "erik_eligible": True,
            "pch_layer": 1,
            "ensembl_id": ensembl_id,
            "gene_symbol": symbol,
            "association_score": score,
            "genetic_association": genetic_association,
            "known_drug_count": known_drug_count,
        },
    )
    return item


def _make_evidence_item(row: dict) -> EvidenceItem:
    """Build an EvidenceItem from an OpenTargets association row."""
    return _parse_target_association(row)


# ---------------------------------------------------------------------------
# OpenTargetsConnector
# ---------------------------------------------------------------------------

class OpenTargetsConnector(BaseConnector):
    """Connector for the OpenTargets Platform GraphQL API.

    Retrieves ALS target-disease associations and converts them into
    EvidenceItem objects with PCH layer 1 (associational evidence).
    """

    GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
    ALS_EFO_ID = "EFO_0000253"

    GRAPHQL_QUERY = """
        query ALSTargets($efoId: String!, $size: Int!) {
          disease(efoId: $efoId) {
            associatedTargets(page: {size: $size, index: 0}) {
              rows {
                target {
                  id
                  approvedSymbol
                }
                score
                datatypeScores {
                  id
                  score
                }
              }
            }
          }
        }
    """

    def __init__(self, *, store=None) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # BaseConnector contract
    # ------------------------------------------------------------------

    def fetch(self, **kwargs) -> ConnectorResult:
        """Top-level fetch: delegates to fetch_als_targets."""
        return self.fetch_als_targets()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_als_targets(
        self,
        min_score: float = 0.1,
        max_results: int = 100,
    ) -> ConnectorResult:
        """Query OpenTargets for ALS-associated targets and build EvidenceItems.

        Parameters
        ----------
        min_score:
            Minimum overall association score to include.
        max_results:
            Maximum number of rows to request from the API (page size).

        Returns
        -------
        ConnectorResult with evidence_items_added count and any errors.
        """
        result = ConnectorResult()

        variables = {
            "efoId": self.ALS_EFO_ID,
            "size": max_results,
        }

        try:
            resp = self._retry_with_backoff(
                requests.post,
                self.GRAPHQL_URL,
                json={"query": self.GRAPHQL_QUERY, "variables": variables},
                timeout=self.REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            result.errors.append(f"OpenTargets GraphQL request failed: {e}")
            return result

        try:
            rows = (
                data["data"]["disease"]["associatedTargets"]["rows"]
            )
        except (KeyError, TypeError) as e:
            result.errors.append(f"Unexpected OpenTargets response shape: {e}")
            return result

        for row in rows:
            try:
                score = float(row.get("score", 0.0))
                if score < min_score:
                    continue
                item = _make_evidence_item(row)
                if self._store:
                    self._store.upsert_evidence_item(item)
                result.evidence_items_added += 1
            except Exception as e:
                symbol = row.get("target", {}).get("approvedSymbol", "unknown")
                result.errors.append(f"Failed to parse row for {symbol}: {e}")

        return result
