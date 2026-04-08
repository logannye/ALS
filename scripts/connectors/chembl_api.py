"""ChEMBLAPIConnector — queries the ChEMBL REST API for bioactivity data.

Replaces the 30GB local ChEMBL SQLite database for Railway deployment.
Uses the EBI ChEMBL REST API at https://www.ebi.ac.uk/chembl/api/data/.

API endpoints used:
  - GET /target.json?target_components__accession={uniprot}&limit=1
      Look up a ChEMBL target by UniProt accession.
  - GET /target/search.json?q={gene}&limit=5
      Search for a target by gene name.
  - GET /activity.json?target_chembl_id={id}&standard_type__in=IC50,EC50,Ki,Kd&pchembl_value__gte=5&limit=20
      Fetch bioactivity records for a ChEMBL target.
  - GET /mechanism.json?target_components__accession={uniprot}&limit=10
      Fetch mechanism-of-action records for a UniProt target.
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


def _parse_bioactivity_record(record: dict, target_name: str = "") -> EvidenceItem:
    """Parse one ChEMBL activity record into an EvidenceItem.

    Parameters
    ----------
    record:
        Dict from /activity.json `activities` list.
    target_name:
        Human-readable target name for the claim string.

    Returns
    -------
    EvidenceItem with PCH layer 2 (interventional evidence — measured binding).
    """
    mol_id = record.get("molecule_chembl_id", "")
    mol_name = record.get("molecule_pref_name") or mol_id
    target_id = record.get("target_chembl_id", "")
    activity_type = record.get("standard_type", "")
    activity_value = record.get("standard_value", "")
    activity_units = record.get("standard_units", "nM")
    pchembl = record.get("pchembl_value", "")

    display_target = target_name or target_id
    claim = (
        f"{mol_name} has {activity_type} = {activity_value} {activity_units} "
        f"against {display_target}"
        + (f" (pChEMBL={pchembl})" if pchembl else "")
    )

    # pChEMBL ≥ 7 → strong, ≥ 5 → moderate, else emerging
    try:
        pchembl_float = float(pchembl) if pchembl else 0.0
    except (TypeError, ValueError):
        pchembl_float = 0.0

    if pchembl_float >= 7.0:
        strength = EvidenceStrength.strong
    elif pchembl_float >= 5.0:
        strength = EvidenceStrength.moderate
    else:
        strength = EvidenceStrength.emerging

    # Confidence from pChEMBL (normalise to [0, 1] from typical range 3–10)
    confidence = min(1.0, max(0.0, (pchembl_float - 3.0) / 7.0)) if pchembl_float else 0.5

    item_id = f"evi:chembl_api:{mol_id}_{target_id}_{activity_type.lower()}"

    return EvidenceItem(
        id=item_id,
        claim=claim,
        direction=EvidenceDirection.supports,
        strength=strength,
        provenance=Provenance(
            source_system=SourceSystem.database,
            asserted_by="chembl_api_connector",
        ),
        uncertainty=Uncertainty(confidence=confidence),
        body={
            "protocol_layer": "root_cause_suppression",
            "pch_layer": 2,
            "mechanism_target": display_target,
            "applicable_subtypes": ["sporadic_tdp43", "unresolved"],
            "erik_eligible": True,
            "molecule_chembl_id": mol_id,
            "molecule_name": mol_name,
            "target_chembl_id": target_id,
            "activity_type": activity_type,
            "activity_value": activity_value,
            "activity_units": activity_units,
            "pchembl_value": pchembl,
            "data_source": "chembl_api",
        },
    )


def _parse_mechanism_record(record: dict) -> EvidenceItem:
    """Parse one ChEMBL mechanism record into an EvidenceItem.

    Parameters
    ----------
    record:
        Dict from /mechanism.json `mechanisms` list.

    Returns
    -------
    EvidenceItem with PCH layer 2.
    """
    mol_id = record.get("molecule_chembl_id", "")
    mol_name = record.get("molecule_pref_name") or mol_id
    target_id = record.get("target_chembl_id", "")
    moa = record.get("mechanism_of_action", "")
    action_type = record.get("action_type", "")

    claim = (
        f"{mol_name} acts on target {target_id} via {moa}"
        + (f" (action_type={action_type})" if action_type else "")
    )

    item_id = f"evi:chembl_api_moa:{mol_id}_{target_id}"

    return EvidenceItem(
        id=item_id,
        claim=claim,
        direction=EvidenceDirection.supports,
        strength=EvidenceStrength.preclinical,
        provenance=Provenance(
            source_system=SourceSystem.database,
            asserted_by="chembl_api_connector",
        ),
        uncertainty=Uncertainty(confidence=0.7),
        body={
            "protocol_layer": "root_cause_suppression",
            "pch_layer": 2,
            "mechanism_target": target_id,
            "applicable_subtypes": ["sporadic_tdp43", "unresolved"],
            "erik_eligible": True,
            "molecule_chembl_id": mol_id,
            "molecule_name": mol_name,
            "target_chembl_id": target_id,
            "mechanism_of_action": moa,
            "action_type": action_type,
            "data_source": "chembl_api",
        },
    )


# ---------------------------------------------------------------------------
# ChEMBLAPIConnector
# ---------------------------------------------------------------------------


class ChEMBLAPIConnector(BaseConnector):
    """Connector for the ChEMBL REST API — replaces local 30GB SQLite DB.

    Queries the public EBI ChEMBL API for bioactivity and mechanism-of-action
    data and converts results into EvidenceItem objects.

    Rate limits:
        The ChEMBL REST API is freely accessible; this connector uses a
        30-second request timeout and exponential backoff (inherited from
        BaseConnector) to handle transient failures gracefully.
    """

    BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
    DEFAULT_HEADERS = {"Accept": "application/json"}

    def __init__(self, store=None, **kwargs) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # BaseConnector contract
    # ------------------------------------------------------------------

    def fetch(self, gene: str = "", uniprot: str = "", **kwargs) -> ConnectorResult:
        """Fetch bioactivity evidence for a gene or UniProt accession.

        Parameters
        ----------
        gene:
            Gene symbol to search, e.g. "SIGMAR1".
        uniprot:
            UniProt accession, e.g. "Q99720".

        Returns
        -------
        ConnectorResult with evidence_items_added count and any errors.
        """
        result = ConnectorResult()

        # Require at least one identifier
        if not gene and not uniprot:
            return result

        if uniprot:
            return self._fetch_by_uniprot(uniprot)
        else:
            return self._fetch_by_gene(gene)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_full_profile(
        self,
        uniprot_id: str = "",
        **kwargs,
    ) -> ConnectorResult:
        """Fetch a full profile: bioactivity + mechanism of action.

        Parameters
        ----------
        uniprot_id:
            UniProt accession, e.g. "Q99720" (SIGMAR1).

        Returns
        -------
        Aggregated ConnectorResult across bioactivity + mechanism queries.
        """
        combined = ConnectorResult()

        if not uniprot_id:
            return combined

        # 1) Look up ChEMBL target ID for this UniProt
        try:
            target_chembl_id, target_name = self._retry_with_backoff(
                self._lookup_target_by_uniprot, uniprot_id
            )
        except Exception as e:
            combined.errors.append(
                f"ChEMBL target lookup failed for uniprot={uniprot_id}: {e}"
            )
            return combined

        if not target_chembl_id:
            return combined

        # 2) Bioactivity
        bio_result = self._fetch_bioactivity(target_chembl_id, target_name)
        combined.evidence_items_added += bio_result.evidence_items_added
        combined.skipped_duplicates += bio_result.skipped_duplicates
        combined.errors.extend(bio_result.errors)

        # 3) Mechanism of action
        moa_result = self._fetch_mechanisms(uniprot_id)
        combined.evidence_items_added += moa_result.evidence_items_added
        combined.skipped_duplicates += moa_result.skipped_duplicates
        combined.errors.extend(moa_result.errors)

        return combined

    # ------------------------------------------------------------------
    # Private: high-level fetch helpers
    # ------------------------------------------------------------------

    def _fetch_by_uniprot(self, uniprot_id: str) -> ConnectorResult:
        """Fetch evidence via UniProt accession → ChEMBL target → bioactivity."""
        result = ConnectorResult()

        try:
            target_chembl_id, target_name = self._retry_with_backoff(
                self._lookup_target_by_uniprot, uniprot_id
            )
        except Exception as e:
            result.errors.append(
                f"ChEMBL target lookup failed for uniprot={uniprot_id}: {e}"
            )
            return result

        if not target_chembl_id:
            return result

        return self._fetch_bioactivity(target_chembl_id, target_name)

    def _fetch_by_gene(self, gene: str) -> ConnectorResult:
        """Fetch evidence via gene name search → ChEMBL target → bioactivity."""
        result = ConnectorResult()

        try:
            target_chembl_id, target_name = self._retry_with_backoff(
                self._lookup_target_by_gene, gene
            )
        except Exception as e:
            result.errors.append(
                f"ChEMBL target search failed for gene={gene}: {e}"
            )
            return result

        if not target_chembl_id:
            return result

        return self._fetch_bioactivity(target_chembl_id, target_name)

    # ------------------------------------------------------------------
    # Private: ChEMBL API call wrappers
    # ------------------------------------------------------------------

    def _lookup_target_by_uniprot(self, uniprot_id: str) -> tuple[str, str]:
        """Return (target_chembl_id, target_name) for a UniProt accession.

        Calls: GET /target.json?target_components__accession={uniprot}&limit=1
        """
        url = f"{self.BASE_URL}/target.json"
        params = {
            "target_components__accession": uniprot_id,
            "limit": 1,
        }
        resp = requests.get(
            url,
            params=params,
            headers=self.DEFAULT_HEADERS,
            timeout=self.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        targets = data.get("targets", [])
        if not targets:
            return "", ""
        first = targets[0]
        return first.get("target_chembl_id", ""), first.get("pref_name", "")

    def _lookup_target_by_gene(self, gene: str) -> tuple[str, str]:
        """Return (target_chembl_id, target_name) for a gene symbol.

        Calls: GET /target/search.json?q={gene}&limit=5
        Selects first SINGLE PROTEIN hit; falls back to first result.
        """
        url = f"{self.BASE_URL}/target/search.json"
        params = {"q": gene, "limit": 5}
        resp = requests.get(
            url,
            params=params,
            headers=self.DEFAULT_HEADERS,
            timeout=self.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        targets = data.get("targets", [])
        if not targets:
            return "", ""

        # Prefer SINGLE PROTEIN hits
        for t in targets:
            if t.get("target_type", "").upper() == "SINGLE PROTEIN":
                return t.get("target_chembl_id", ""), t.get("pref_name", "")

        first = targets[0]
        return first.get("target_chembl_id", ""), first.get("pref_name", "")

    def _fetch_bioactivity(
        self,
        target_chembl_id: str,
        target_name: str = "",
        limit: int = 20,
    ) -> ConnectorResult:
        """Fetch bioactivity records for a ChEMBL target ID.

        Calls: GET /activity.json?target_chembl_id={id}&...
        """
        result = ConnectorResult()
        url = f"{self.BASE_URL}/activity.json"
        params = {
            "target_chembl_id": target_chembl_id,
            "standard_type__in": "IC50,EC50,Ki,Kd",
            "pchembl_value__gte": 5,
            "limit": limit,
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
                f"ChEMBL bioactivity fetch failed for target={target_chembl_id}: {e}"
            )
            return result

        activities = data.get("activities", [])
        if activities is None:
            return result

        for record in activities:
            try:
                item = _parse_bioactivity_record(record, target_name)
                if self._store:
                    self._store.upsert_evidence_item(item)
                result.evidence_items_added += 1
            except Exception as e:
                mol_id = record.get("molecule_chembl_id", "unknown")
                result.errors.append(
                    f"Failed to parse bioactivity record for mol={mol_id}: {e}"
                )

        return result

    def _fetch_mechanisms(
        self,
        uniprot_id: str,
        limit: int = 10,
    ) -> ConnectorResult:
        """Fetch mechanism-of-action records for a UniProt target.

        Calls: GET /mechanism.json?target_components__accession={uniprot}&limit=10
        """
        result = ConnectorResult()
        url = f"{self.BASE_URL}/mechanism.json"
        params = {
            "target_components__accession": uniprot_id,
            "limit": limit,
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
                f"ChEMBL mechanism fetch failed for uniprot={uniprot_id}: {e}"
            )
            return result

        mechanisms = data.get("mechanisms", [])
        if mechanisms is None:
            return result

        for record in mechanisms:
            try:
                item = _parse_mechanism_record(record)
                if self._store:
                    self._store.upsert_evidence_item(item)
                result.evidence_items_added += 1
            except Exception as e:
                mol_id = record.get("molecule_chembl_id", "unknown")
                result.errors.append(
                    f"Failed to parse mechanism record for mol={mol_id}: {e}"
                )

        return result
