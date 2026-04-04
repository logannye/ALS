"""CMap / LINCS Connector — drug repurposing via transcriptomic signature reversal.

Queries the Connectivity Map (clue.io) API to find compounds that reverse
an ALS disease gene expression signature. If no explicit gene lists are
provided, the connector loads a disease signature from GEOALSConnector.

API: https://api.clue.io/api (may require user_key authentication)
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import requests

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import BaseEnvelope

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.clue.io/api"


# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------

def _normalize_compound(name: str) -> str:
    """Normalize a compound name to a canonical form.

    Lowercases and replaces spaces with underscores.

    Parameters
    ----------
    name:
        Compound name (e.g. ``"Riluzole Hydrochloride"``).

    Returns
    -------
    Normalized string (e.g. ``"riluzole_hydrochloride"``).
    """
    return name.lower().replace(" ", "_")


def _build_reversal_query_hash(up_genes: list[str], down_genes: list[str]) -> str:
    """Build a deterministic 12-char hex hash for a gene signature query.

    Genes are sorted before hashing so order does not matter.

    Parameters
    ----------
    up_genes:
        List of upregulated gene symbols.
    down_genes:
        List of downregulated gene symbols.

    Returns
    -------
    First 12 hex characters of the MD5 hash.
    """
    up_sorted = sorted(g.upper() for g in up_genes)
    dn_sorted = sorted(g.upper() for g in down_genes)
    payload = "UP:" + ",".join(up_sorted) + "|DN:" + ",".join(dn_sorted)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# CMapConnector
# ---------------------------------------------------------------------------

class CMapConnector(BaseConnector):
    """Connector for the Connectivity Map (CMap / LINCS) at clue.io.

    Two query modes:
    1. **By compound**: look up perturbation signatures for a named compound.
    2. **By gene signature**: find compounds that reverse a disease signature.

    The clue.io API may require authentication. All API failures are handled
    gracefully and reported as errors in the ConnectorResult.
    """

    def __init__(
        self,
        store: Any = None,
        api_key: str | None = None,
        min_connectivity_score: float = -90.0,
    ) -> None:
        self._store = store
        self._api_key = api_key
        self._min_connectivity_score = min_connectivity_score

    # ------------------------------------------------------------------
    # BaseConnector contract
    # ------------------------------------------------------------------

    def fetch(
        self,
        *,
        compound: str = "",
        up_genes: list[str] | None = None,
        down_genes: list[str] | None = None,
        **kwargs: Any,
    ) -> ConnectorResult:
        """Fetch CMap perturbation or reversal evidence.

        Parameters
        ----------
        compound:
            Compound name for Mode 1 (perturbation lookup).
        up_genes:
            Upregulated genes for Mode 2 (signature reversal query).
        down_genes:
            Downregulated genes for Mode 2.
        **kwargs:
            Silently ignored.

        Returns
        -------
        ConnectorResult with evidence_items_added count and any errors.
        """
        if compound:
            return self._fetch_by_compound(compound)
        return self._fetch_by_signature(up_genes, down_genes)

    # ------------------------------------------------------------------
    # Mode 1: By compound
    # ------------------------------------------------------------------

    def _fetch_by_compound(self, compound: str) -> ConnectorResult:
        """Look up perturbation signatures for a named compound."""
        result = ConnectorResult()
        compound_norm = _normalize_compound(compound)

        headers = self._build_headers()
        where_clause = json.dumps({"pert_iname": compound})
        fields = json.dumps(["pert_iname", "cell_id", "pert_dose", "pert_time", "score"])
        url = f"{_BASE_URL}/sigs"
        params = {"where": where_clause, "fields": fields}

        try:
            resp = self._retry_with_backoff(
                requests.get,
                url,
                params=params,
                headers=headers,
                timeout=self.REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            msg = f"CMap API unavailable: {e}"
            logger.warning(msg)
            result.errors.append(msg)
            return result

        if not isinstance(data, list):
            result.errors.append("CMap API returned unexpected format (expected list)")
            return result

        for sig in data:
            try:
                score = float(sig.get("score", 0))
                if abs(score) <= 80:
                    continue

                cell_id = sig.get("cell_id", "unknown")
                dose = sig.get("pert_dose", "")
                time_val = sig.get("pert_time", "")

                evi_id = f"evi:cmap:{compound_norm}_{cell_id.lower()}_perturbation"
                direction = "positive" if score > 0 else "negative"

                claim = (
                    f"CMap perturbation: {compound} in {cell_id} shows "
                    f"{direction} connectivity (score={score:.1f}"
                )
                if dose:
                    claim += f", dose={dose}"
                if time_val:
                    claim += f", time={time_val}"
                claim += ")"

                evi = BaseEnvelope(
                    id=evi_id,
                    type="EvidenceItem",
                    status="active",
                    body={
                        "claim": claim,
                        "data_source": "cmap",
                        "compound": compound,
                        "cell_id": cell_id,
                        "connectivity_score": score,
                        "pert_dose": dose,
                        "pert_time": time_val,
                        "pch_layer": 2,
                    },
                )

                if self._store:
                    self._store.upsert_object(evi)
                result.evidence_items_added += 1

            except Exception as e:
                logger.debug("Failed to parse CMap signature: %s", e)
                result.errors.append(f"CMap signature parse error: {e}")

        return result

    # ------------------------------------------------------------------
    # Mode 2: By gene signature
    # ------------------------------------------------------------------

    def _fetch_by_signature(
        self,
        up_genes: list[str] | None,
        down_genes: list[str] | None,
    ) -> ConnectorResult:
        """Find compounds that reverse a disease gene expression signature."""
        result = ConnectorResult()

        # If no gene lists provided, try to load from GEO ALS
        if not up_genes or not down_genes:
            try:
                from connectors.geo_als import GEOALSConnector
                geo = GEOALSConnector()
                up_genes, down_genes = geo.get_disease_signature(top_n=100)
            except Exception:
                pass

        if not up_genes or not down_genes:
            return result  # no signature available

        query_hash = _build_reversal_query_hash(up_genes, down_genes)
        headers = self._build_headers()
        url = f"{_BASE_URL}/queryl1000"
        payload = {
            "up": up_genes[:150],
            "dn": down_genes[:150],
        }

        try:
            resp = self._retry_with_backoff(
                requests.post,
                url,
                json=payload,
                headers=headers,
                timeout=self.REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            msg = f"CMap API unavailable: {e}"
            logger.warning(msg)
            result.errors.append(msg)
            return result

        # Parse results — expect a list of compound hits
        hits = data if isinstance(data, list) else data.get("results", [])

        for hit in hits:
            try:
                score = float(hit.get("score", hit.get("connectivity_score", 0)))
                if score > self._min_connectivity_score:
                    continue  # not reversing strongly enough

                compound_name = hit.get("pert_iname", hit.get("name", ""))
                if not compound_name:
                    continue

                compound_norm = _normalize_compound(compound_name)
                evi_id = f"evi:cmap:reversal_{query_hash}_{compound_norm}"

                claim = (
                    f"CMap reversal: {compound_name} reverses ALS transcriptomic "
                    f"signature (connectivity={score:.1f})"
                )

                evi = BaseEnvelope(
                    id=evi_id,
                    type="EvidenceItem",
                    status="active",
                    body={
                        "claim": claim,
                        "data_source": "cmap",
                        "compound": compound_name,
                        "connectivity_score": score,
                        "query_hash": query_hash,
                        "drug_repurposing_candidate": True,
                        "pch_layer": 2,
                    },
                )

                if self._store:
                    self._store.upsert_object(evi)
                result.evidence_items_added += 1

            except Exception as e:
                logger.debug("Failed to parse CMap reversal hit: %s", e)
                result.errors.append(f"CMap reversal parse error: {e}")

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers, including API key if available."""
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._api_key:
            headers["user_key"] = self._api_key
        return headers
