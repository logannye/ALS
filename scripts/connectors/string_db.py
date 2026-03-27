# scripts/connectors/string_db.py
"""STRING protein-protein interaction connector."""
from __future__ import annotations
import requests
from connectors.base import BaseConnector, ConnectorResult
from ontology.base import Provenance
from ontology.enums import EvidenceDirection, EvidenceStrength, SourceSystem
from ontology.evidence import EvidenceItem

BASE_URL = "https://string-db.org/api"

def _parse_network_response(raw: list[dict], query_gene: str) -> list[dict]:
    return [{"gene_a": e.get("preferredName_A", ""), "gene_b": e.get("preferredName_B", ""),
             "combined_score": e.get("score", 0), "experimental_score": e.get("escore", 0)} for e in raw]

class STRINGConnector(BaseConnector):
    def __init__(self, evidence_store=None, base_url: str = BASE_URL, min_score: int = 400):
        self._store = evidence_store
        self._base_url = base_url
        self._min_score = min_score

    def fetch(self, *, gene_symbol: str = "", limit: int = 20, **kwargs) -> ConnectorResult:
        result = ConnectorResult()
        if not gene_symbol:
            result.errors.append("gene_symbol required")
            return result
        try:
            interactions = self._fetch_network(gene_symbol, limit)
        except Exception as e:
            result.errors.append(f"STRING API error: {e}")
            return result
        for ix in interactions:
            if ix["combined_score"] < self._min_score:
                continue
            item = self._build_evidence_item(ix["gene_a"], ix["gene_b"], ix["combined_score"], ix["experimental_score"])
            if self._store:
                self._store.upsert_evidence_item(item)
            result.evidence_items_added += 1
        return result

    def _fetch_network(self, gene_symbol, limit):
        url = f"{self._base_url}/json/network"
        params = {"identifiers": gene_symbol, "species": 9606, "limit": limit, "required_score": self._min_score, "caller_identity": "erik_als_engine"}
        resp = self._retry_with_backoff(requests.get, url, params=params, timeout=self.REQUEST_TIMEOUT)
        resp.raise_for_status()
        return _parse_network_response(resp.json(), gene_symbol)

    def _build_evidence_item(self, gene_a, gene_b, combined_score, experimental_score):
        sorted_pair = sorted([gene_a, gene_b])
        eid = f"evi:string:{sorted_pair[0]}_{sorted_pair[1]}"
        if combined_score >= 700:
            strength = EvidenceStrength.strong
        elif combined_score >= 400:
            strength = EvidenceStrength.moderate
        else:
            strength = EvidenceStrength.emerging
        return EvidenceItem(
            id=eid,
            claim=f"Protein interaction: {gene_a} - {gene_b} (STRING score {combined_score}/1000)",
            direction=EvidenceDirection.supports, strength=strength,
            source_refs=[f"string:{gene_a}_{gene_b}"],
            provenance=Provenance(source_system=SourceSystem.database, asserted_by="string_connector"),
            body={"gene_a": gene_a, "gene_b": gene_b, "combined_score": combined_score, "experimental_score": experimental_score, "pch_layer": 1, "data_source": "string"},
        )
