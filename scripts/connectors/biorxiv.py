"""BiorxivConnector — fetches preprint evidence from bioRxiv and medRxiv.

Uses the official bioRxiv/medRxiv API:
  https://api.biorxiv.org/details/{server}/{start_date}/{end_date}/0/{max}

Preprints are ALWAYS assigned EvidenceStrength.emerging because they have
not undergone peer review.  The DOI is required for a stable canonical ID;
preprints without a DOI are silently dropped.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import Provenance
from ontology.enums import (
    EvidenceDirection,
    EvidenceStrength,
    SourceSystem,
)
from ontology.evidence import EvidenceItem

logger = logging.getLogger(__name__)

_SERVERS = ("biorxiv", "medrxiv")
_BASE_URL = "https://api.biorxiv.org/details"


# ---------------------------------------------------------------------------
# Free function: parse a single API result dict
# ---------------------------------------------------------------------------

def _parse_preprint(raw: dict) -> Optional[EvidenceItem]:
    """Parse a bioRxiv/medRxiv API result dict into an EvidenceItem.

    Parameters
    ----------
    raw:
        A single entry from the ``collection`` list returned by the API.

    Returns
    -------
    EvidenceItem if the entry has a valid DOI, else None.
    """
    doi = raw.get("doi") or ""
    if not doi:
        return None

    server = (raw.get("server") or "biorxiv").lower()
    title = raw.get("title") or ""
    abstract = (raw.get("abstract") or "")[:2000]
    authors = raw.get("authors") or ""
    date = raw.get("date") or ""
    category = raw.get("category") or ""

    # Canonical ID: evi:{server}:{doi} — lowercased, / and spaces → _
    doi_clean = doi.lower().replace("/", "_").replace(" ", "_")
    item_id = f"evi:{server}:{doi_clean}"

    # Claim: "[Preprint] {title}" — truncate entire string to 500 chars
    claim = f"[Preprint] {title}"[:500]

    return EvidenceItem(
        id=item_id,
        claim=claim,
        direction=EvidenceDirection.insufficient,
        strength=EvidenceStrength.emerging,  # ALWAYS emerging — not peer-reviewed
        source_refs=[f"doi:{doi}"],
        provenance=Provenance(
            source_system=SourceSystem.literature,
            asserted_by="biorxiv_connector",
            source_artifact_id=doi,
        ),
        body={
            "doi": doi,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "date": date,
            "server": server,
            "category": category,
            "peer_reviewed": False,
            "strength": EvidenceStrength.emerging.value,
        },
    )


# ---------------------------------------------------------------------------
# BiorxivConnector
# ---------------------------------------------------------------------------

class BiorxivConnector(BaseConnector):
    """Connector for the bioRxiv/medRxiv preprint API.

    Searches both servers for preprints relevant to a query and upserts
    matching EvidenceItem objects into the Erik evidence store.

    All results are assigned ``EvidenceStrength.emerging`` because preprints
    are not peer-reviewed.
    """

    def __init__(
        self,
        store=None,
        *,
        enabled: bool = True,
        lookback_days: int = 90,
        max_results: int = 15,
    ) -> None:
        self._store = store
        self._enabled = enabled
        self._lookback_days = lookback_days
        self._max_results = max_results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, query: str = "ALS motor neuron", **kwargs) -> ConnectorResult:
        """Search bioRxiv and medRxiv for preprints relevant to *query*.

        Filters results by checking whether any word from the query appears in
        the title or abstract (case-insensitive).  Uses ``_retry_with_backoff``
        for HTTP calls.

        Parameters
        ----------
        query:
            Free-text search string.  Results are filtered by word presence in
            title+abstract.

        Returns
        -------
        ConnectorResult with counts of added items, duplicates, and errors.
        """
        result = ConnectorResult()

        if not self._enabled:
            return result

        # Compute date window
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=self._lookback_days)
        start_date = start_dt.strftime("%Y-%m-%d")
        end_date = end_dt.strftime("%Y-%m-%d")

        # Query words for relevance filtering
        query_words = [w.lower() for w in query.split() if w]

        for server in _SERVERS:
            self._fetch_server(
                server=server,
                start_date=start_date,
                end_date=end_date,
                query_words=query_words,
                result=result,
            )

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_server(
        self,
        server: str,
        start_date: str,
        end_date: str,
        query_words: list[str],
        result: ConnectorResult,
    ) -> None:
        """Fetch preprints from one server and merge into *result*."""
        url = f"{_BASE_URL}/{server}/{start_date}/{end_date}/0/{self._max_results}"
        try:
            resp = self._retry_with_backoff(
                requests.get, url, timeout=self.REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            result.errors.append(f"{server} fetch failed: {exc}")
            return

        collection = data.get("collection") or []
        for raw in collection:
            # Relevance filter: any query word must appear in title or abstract
            title = (raw.get("title") or "").lower()
            abstract = (raw.get("abstract") or "").lower()
            combined = title + " " + abstract
            if query_words and not any(w in combined for w in query_words):
                continue

            try:
                item = _parse_preprint(raw)
                if item is None:
                    continue
                if self._store:
                    self._store.upsert_evidence_item(item)
                result.evidence_items_added += 1
            except Exception as exc:
                doi = raw.get("doi", "unknown")
                result.errors.append(f"Parse/upsert failed for DOI {doi}: {exc}")
