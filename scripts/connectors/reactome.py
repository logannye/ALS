"""Reactome Content Service connector — curated biological pathways."""
from __future__ import annotations
import requests
from typing import Any, Optional
from connectors.base import BaseConnector, ConnectorResult
from ontology.base import Provenance
from ontology.enums import EvidenceDirection, EvidenceStrength, SourceSystem
from ontology.evidence import EvidenceItem

BASE_URL = "https://reactome.org/ContentService"

def _parse_pathway_response(raw: list[dict], uniprot_id: str) -> list[dict]:
    results = []
    for entry in raw:
        if entry.get("speciesName") != "Homo sapiens":
            continue
        results.append({"pathway_id": entry["stId"], "pathway_name": entry.get("displayName", ""), "uniprot_id": uniprot_id})
    return results

def _parse_contained_events(raw: list[dict], pathway_id: str) -> list[dict]:
    return [{"reaction_id": e["stId"], "reaction_name": e.get("displayName", ""), "reaction_type": e.get("className", ""), "pathway_id": pathway_id} for e in raw]

class ReactomeConnector(BaseConnector):
    def __init__(self, evidence_store=None, base_url: str = BASE_URL):
        self._store = evidence_store
        self._base_url = base_url

    def fetch(self, *, uniprot_id: str = "", gene_symbol: str = "", **kwargs) -> ConnectorResult:
        result = ConnectorResult()
        if not uniprot_id:
            result.errors.append("uniprot_id is required")
            return result
        try:
            pathways = self._fetch_pathways(uniprot_id)
        except Exception as e:
            result.errors.append(f"Reactome API error: {e}")
            return result
        for pw in pathways[:10]:
            try:
                events = self._fetch_contained_events(pw["pathway_id"])
                num_reactions = len(events)
            except Exception:
                num_reactions = 0
            item = self._build_evidence_item(pw["pathway_id"], pw["pathway_name"], uniprot_id, gene_symbol, num_reactions)
            if self._store:
                self._store.upsert_evidence_item(item)
            result.evidence_items_added += 1
        return result

    def _fetch_pathways(self, uniprot_id: str) -> list[dict]:
        url = f"{self._base_url}/data/pathways/low/entity/{uniprot_id}"
        resp = self._retry_with_backoff(requests.get, url, headers={"Accept": "application/json"}, timeout=self.REQUEST_TIMEOUT)
        resp.raise_for_status()
        return _parse_pathway_response(resp.json(), uniprot_id)

    def _fetch_contained_events(self, pathway_id: str) -> list[dict]:
        url = f"{self._base_url}/data/pathway/{pathway_id}/containedEvents"
        resp = self._retry_with_backoff(requests.get, url, headers={"Accept": "application/json"}, timeout=self.REQUEST_TIMEOUT)
        resp.raise_for_status()
        return _parse_contained_events(resp.json(), pathway_id)

    def _build_evidence_item(self, pathway_id, pathway_name, uniprot_id, gene_symbol, num_reactions):
        return EvidenceItem(
            id=f"evi:reactome:{pathway_id}_{uniprot_id}",
            claim=f"{gene_symbol or uniprot_id} participates in pathway: {pathway_name} ({num_reactions} reactions)",
            direction=EvidenceDirection.supports, strength=EvidenceStrength.strong,
            source_refs=[f"reactome:{pathway_id}"],
            provenance=Provenance(source_system=SourceSystem.database, asserted_by="reactome_connector", source_artifact_id=pathway_id),
            body={"pathway": pathway_id, "pathway_name": pathway_name, "uniprot_id": uniprot_id, "gene_symbol": gene_symbol, "num_reactions": num_reactions, "pch_layer": 1, "data_source": "reactome"},
        )
