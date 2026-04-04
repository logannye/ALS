"""FAERSConnector -- FDA Adverse Event Reporting System via openFDA API.

Queries the openFDA drug/event endpoint for post-marketing safety signals.
Uses the count endpoint to retrieve adverse reaction terms and their report
counts for a given drug.  All evidence is PCH layer 1 (observational).

API docs: https://open.fda.gov/apis/drug/event/
"""
from __future__ import annotations

import logging
import re
from typing import Any

import requests

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import BaseEnvelope

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Free functions (for testability)
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    """Lowercase, replace spaces/hyphens with underscores, strip non-alnum.

    >>> _normalize_name("Riluzole Hydrochloride")
    'riluzole_hydrochloride'
    """
    name = name.lower().strip()
    name = name.replace(" ", "_").replace("-", "_")
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name


def _parse_count_response(data: dict, drug_name: str) -> list[dict]:
    """Extract reaction/count pairs from the openFDA count response.

    Parameters
    ----------
    data:
        JSON response from the openFDA count endpoint.
        Expected shape: ``{"results": [{"term": "NAUSEA", "count": 245}, ...]}``
    drug_name:
        The drug name (used for logging only).

    Returns
    -------
    List of dicts with ``reaction`` and ``count`` keys.
    """
    results = data.get("results", [])
    return [
        {"reaction": r.get("term", ""), "count": r.get("count", 0)}
        for r in results
        if r.get("term")
    ]


def _build_safety_profile(reactions: list[dict], drug_name: str) -> dict:
    """Build an aggregated safety profile from reaction counts.

    Parameters
    ----------
    reactions:
        List of ``{"reaction": str, "count": int}`` dicts.
    drug_name:
        Drug name for the profile.

    Returns
    -------
    Dict with total_reports, unique_reactions, top_reactions (top 20 by count),
    and the drug_name.
    """
    total_reports = sum(r["count"] for r in reactions)
    unique_reactions = len(reactions)
    sorted_reactions = sorted(reactions, key=lambda r: r["count"], reverse=True)
    top_reactions = sorted_reactions[:20]

    return {
        "drug_name": drug_name,
        "total_reports": total_reports,
        "unique_reactions": unique_reactions,
        "top_reactions": top_reactions,
    }


def _assess_safety_signal(profile: dict, drug_name: str) -> tuple[bool, str]:
    """Assess whether a safety signal warrants concern.

    With the count-only endpoint we cannot distinguish serious from
    non-serious reactions, so we always return ``is_safe=True`` and let
    downstream reasoning (e.g. protocol generator) interpret the profile.

    Returns
    -------
    (is_safe, reasoning) tuple.
    """
    total = profile.get("total_reports", 0)
    unique = profile.get("unique_reactions", 0)
    return (True, f"{drug_name}: {total} total FAERS reports across {unique} reaction types")


# ---------------------------------------------------------------------------
# FAERSConnector
# ---------------------------------------------------------------------------

class FAERSConnector(BaseConnector):
    """Connector for FDA Adverse Event Reporting System (openFDA API).

    Fetches post-marketing adverse event counts for a given drug and
    creates evidence items for the safety profile and top reactions.
    """

    BASE_URL = "https://api.fda.gov/drug/event.json"

    def __init__(self, store: Any = None, min_report_count: int = 10):
        self._store = store
        self._min_report_count = min_report_count

    # ------------------------------------------------------------------
    # BaseConnector contract
    # ------------------------------------------------------------------

    def fetch(self, *, drug_name: str = "", **kwargs) -> ConnectorResult:
        """Fetch FAERS adverse event counts for *drug_name*.

        Steps:
        1. Query openFDA count endpoint for reaction terms.
        2. Filter reactions below ``min_report_count``.
        3. Create aggregated safety-profile evidence.
        4. Create per-reaction evidence for the top 10 reactions.
        """
        result = ConnectorResult()

        if not drug_name:
            return result

        drug_norm = _normalize_name(drug_name)

        # Build request
        params = {
            "search": f'patient.drug.medicinalproduct:"{drug_name}"',
            "count": "patient.reaction.reactionmeddrapt.exact",
            "limit": "100",
        }

        try:
            resp = self._retry_with_backoff(
                requests.get,
                self.BASE_URL,
                params=params,
                timeout=self.REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            result.errors.append(f"FAERS API request failed for {drug_name}: {e}")
            return result

        # Parse and filter
        reactions = _parse_count_response(data, drug_name)
        reactions = [r for r in reactions if r["count"] >= self._min_report_count]

        if not reactions:
            return result

        # Build safety profile
        profile = _build_safety_profile(reactions, drug_name)

        # --- Aggregated safety-profile evidence ---
        top_20 = profile["top_reactions"]
        claim = (
            f"FAERS safety profile for {drug_name}: "
            f"{profile['total_reports']} total adverse event reports across "
            f"{profile['unique_reactions']} reaction types. "
            f"Top reactions: {', '.join(r['reaction'] for r in top_20[:5])}."
        )

        profile_evi = BaseEnvelope(
            id=f"evi:faers:{drug_norm}_safety_profile",
            type="EvidenceItem",
            status="active",
            body={
                "claim": claim,
                "source": "faers",
                "drug_name": drug_name,
                "total_reports": profile["total_reports"],
                "unique_reactions": profile["unique_reactions"],
                "top_reactions": top_20,
                "pch_layer": 1,
                "data_source": "fda_faers",
                "evidence_strength": "moderate",
            },
        )

        if self._store:
            self._store.upsert_object(profile_evi)
        result.evidence_items_added += 1

        # --- Per-reaction evidence (top 10) ---
        for reaction_entry in top_20[:10]:
            reaction = reaction_entry["reaction"]
            count = reaction_entry["count"]
            reaction_norm = _normalize_name(reaction)

            reaction_claim = (
                f"FAERS: {drug_name} associated with {reaction} "
                f"({count} reports)"
            )
            reaction_evi = BaseEnvelope(
                id=f"evi:faers:{drug_norm}_{reaction_norm}",
                type="EvidenceItem",
                status="active",
                body={
                    "claim": reaction_claim,
                    "source": "faers",
                    "drug_name": drug_name,
                    "reaction": reaction,
                    "report_count": count,
                    "pch_layer": 1,
                    "data_source": "fda_faers",
                },
            )

            if self._store:
                self._store.upsert_object(reaction_evi)
            result.evidence_items_added += 1

        return result

    # ------------------------------------------------------------------
    # Convenience method
    # ------------------------------------------------------------------

    def fetch_safety_profile(self, drug_name: str) -> tuple[ConnectorResult, bool]:
        """Fetch FAERS data and return (ConnectorResult, is_safe).

        Calls ``fetch(drug_name=...)`` then runs ``_assess_safety_signal``
        to produce the safety assessment.
        """
        cr = self.fetch(drug_name=drug_name)

        # Rebuild profile from the fetch result for assessment
        # (we don't persist intermediate state, so re-derive)
        profile = {"total_reports": 0, "unique_reactions": 0}
        if cr.evidence_items_added > 0:
            # Rough reconstruction: 1 profile + N reaction items
            profile["total_reports"] = cr.evidence_items_added  # placeholder
            profile["unique_reactions"] = max(cr.evidence_items_added - 1, 0)

        # Actually re-fetch the raw data for an accurate assessment
        drug_norm = _normalize_name(drug_name)
        params = {
            "search": f'patient.drug.medicinalproduct:"{drug_name}"',
            "count": "patient.reaction.reactionmeddrapt.exact",
            "limit": "100",
        }
        try:
            resp = self._retry_with_backoff(
                requests.get,
                self.BASE_URL,
                params=params,
                timeout=self.REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            reactions = _parse_count_response(data, drug_name)
            profile = _build_safety_profile(reactions, drug_name)
        except Exception:
            pass  # Use placeholder profile

        _, reasoning = _assess_safety_signal(profile, drug_name)
        is_safe = True  # Count endpoint cannot distinguish serious vs non-serious
        return (cr, is_safe)
