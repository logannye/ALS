# scripts/connectors/kegg.py
"""KEGG REST API connector — pathway ontology and gene-pathway mapping."""
from __future__ import annotations
import requests
from typing import Optional
from connectors.base import BaseConnector, ConnectorResult
from ontology.base import Provenance
from ontology.enums import EvidenceDirection, EvidenceStrength, SourceSystem
from ontology.evidence import EvidenceItem

BASE_URL = "https://rest.kegg.jp"

def _parse_link_response(text: str) -> list[tuple[str, str]]:
    links = []
    for line in text.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split("\t")
        if len(parts) == 2:
            links.append((parts[0], parts[1]))
    return links

def _parse_pathway_entry(text: str) -> dict:
    name = ""
    for line in text.split("\n"):
        if line.startswith("NAME"):
            name = line[12:].strip().split(" - Homo sapiens")[0].strip()
            break
    return {"name": name}

class KEGGConnector(BaseConnector):
    def __init__(self, evidence_store=None, base_url: str = BASE_URL):
        self._store = evidence_store
        self._base_url = base_url

    def fetch(self, *, gene_id: str = "", gene_symbol: str = "", **kwargs) -> ConnectorResult:
        result = ConnectorResult()
        if not gene_id and not gene_symbol:
            result.errors.append("gene_id or gene_symbol required")
            return result
        kegg_gene = f"hsa:{gene_id}" if gene_id else ""
        try:
            links = self._fetch_gene_pathways(kegg_gene)
        except Exception as e:
            result.errors.append(f"KEGG API error: {e}")
            return result
        for _, pathway_ref in links[:10]:
            pathway_id = pathway_ref.replace("path:", "")
            try:
                entry = self._fetch_pathway_info(pathway_id)
                pathway_name = entry["name"]
            except Exception:
                pathway_name = pathway_id
            item = self._build_evidence_item(pathway_id, pathway_name, gene_id, gene_symbol)
            if self._store:
                self._store.upsert_evidence_item(item)
            result.evidence_items_added += 1
        return result

    def _fetch_gene_pathways(self, kegg_gene):
        url = f"{self._base_url}/link/pathway/{kegg_gene}"
        resp = self._retry_with_backoff(requests.get, url, timeout=self.REQUEST_TIMEOUT)
        resp.raise_for_status()
        return _parse_link_response(resp.text)

    def _fetch_pathway_info(self, pathway_id):
        url = f"{self._base_url}/get/{pathway_id}"
        resp = self._retry_with_backoff(requests.get, url, timeout=self.REQUEST_TIMEOUT)
        resp.raise_for_status()
        return _parse_pathway_entry(resp.text)

    def _build_evidence_item(self, pathway_id, pathway_name, gene_id, gene_symbol):
        return EvidenceItem(
            id=f"evi:kegg:{pathway_id}_{gene_symbol or gene_id}",
            claim=f"{gene_symbol or gene_id} participates in KEGG pathway: {pathway_name}",
            direction=EvidenceDirection.supports, strength=EvidenceStrength.strong,
            source_refs=[f"kegg:{pathway_id}"],
            provenance=Provenance(source_system=SourceSystem.database, asserted_by="kegg_connector", source_artifact_id=pathway_id),
            body={"pathway": pathway_id, "pathway_name": pathway_name, "gene_id": gene_id, "gene_symbol": gene_symbol, "pch_layer": 1, "data_source": "kegg"},
        )
